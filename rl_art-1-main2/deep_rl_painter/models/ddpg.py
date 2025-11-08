"""
DDPG Agent Module

Defines the DDPGAgent class for training and action inference in a goal-conditioned painting task.

The agent:
    - Uses dual CNN encoders to process canvas and target images.
    - Takes previous actions into account to maintain temporal coherence.
    - Trains Actor and Critic networks using transitions from a replay buffer.

Key Inputs:
    - canvas: current image
    - target_image: goal image to paint
    - prev_action: the last action applied to canvas
    - action: the action to evaluate or apply

Outputs:
    - Actor: next action to apply (x, y, r, g, b, width)
    - Critic: Q-value estimating expected return from a (state, action) pair
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import time
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import math
import time
import torch, torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb

def _sync_min_int(val: int, device): # for shared step -> global_step
    if not dist.is_initialized():
        return val
    t = torch.tensor([val], device=device, dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())

def _allr_min_bool(flag: bool, device):
    if not dist.is_initialized():
        return flag
    t = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(int(t.item()))

class DDPGAgent:
    def __init__(self, actor, critic, actor_target, critic_target,
                 actor_optimizer, critic_optimizer, replay_buffer, noise, config):
        self.actor = actor
        self.critic = critic
        self.actor_target = actor_target
        self.critic_target = critic_target
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.replay_buffer = replay_buffer
        self.noise = noise
        self.config = config
        self.channels = config["canvas_channels"]
        self.device = config["device"]

        # Initialize target networks with the same weights as the main networks
        #self.actor_target.load_state_dict(actor.state_dict())
        #self.critic_target.load_state_dict(critic.state_dict())

        # handle both plain nn.Module and DDP-wrapped modules
        actor_sd  = actor.module.state_dict()  if hasattr(actor, "module")  else actor.state_dict()
        critic_sd = critic.module.state_dict() if hasattr(critic, "module") else critic.state_dict()

        self.actor_target.load_state_dict(actor_sd)
        self.critic_target.load_state_dict(critic_sd)


        # keep track of epiosde and step numbers - never used
        self.episode = 0
        self.step = 0

        # move networks to GPU
        
        self.resized_target = None  # cache for resized target image

        # train.py may set this to a function that logs grad-flow to W&B
        self.viz_after_backward = None

        # decides gumbel or softmax in select_action
        self.use_gumbel_in_select = True       # runtime choice; doesn't affect training
        self.gumbel_tau = config["gumbel_tau"]

        self.scaler = GradScaler()  #  (for mixed precision)

        # --- hack actor ---
        self.use_script = False   # set False to go back to normal behavior
        self.script_ptr = 0
        script_path = os.path.join(os.path.dirname(__file__), "index.txt")
        with open(script_path, "r") as f:
            txt = f.read()
        # assume comma-separated integers; convert to 0-based once
        self.script = [int(x.strip()) - 1 for x in txt.split(",") if x.strip()]

        #update actor every 2 steps - never used
        self.update_actor = 0

        # reducing time
        self.is_main = bool(self.config.get("is_main", True))
        self.global_step = int(self.config.get("global_step_start", 0))
        self.episode = 0
        self.step_in_ep = 0

        torch.set_float32_matmul_precision("high")   # enable TF32 kernels on Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True
        cudnn.benchmark = True  # pick best conv algos for static shapes

        self.actor_target.eval()
        self.critic_target.eval()
        for p in self.actor_target.parameters():
            p.requires_grad_(False)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.gf_critic_mean = None 
        self.last_critic_loss = None  
        self.gf_actor_mean = None
        self.last_actor_loss = None
        self.last_entropy    = None
        self.last_tau        = None
        self.last_alpha      = None

    def set_progress(self, episode: int, step_in_ep: int):
        self.episode = int(episode)
        self.step_in_ep = int(step_in_ep)
        steps_per_ep = int(self.config.get("max_strokes", 5000))
        self.global_step = self.episode * steps_per_ep + self.step_in_ep

    # resize canvas and target to 224x224 before passing into networks
    # runs on gpu because all inputs given later are tensors
    # and normalises acc to Image Net values
    def resize_input(self, x, size=(224, 224)):
        # x: (B,C,H,W) in [0,255] or [0,1]
        # resize
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        # scale to [0,1] if appears to be [0,255]
        if x.max() > 1.5:
            x = x / 255.0
        #mean = x.new_tensor([0.449]).view(1,1,1,1)  # avg ImageNet RGB means
        #std  = x.new_tensor([0.226]).view(1,1,1,1)  # avg ImageNet RGB stds
        # repeat: BCHW -> repeat channel dim (canvas channel is always 1)
        # encoder needs 3 channels
        x = x.repeat(1, 3, 1, 1)  
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = x.new_tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        # imagenet norm
        x = (x - mean) / std
        return x

    def select_action(self, canvas, target_image, prev_action_onehot):
        """
        Use the actor network to select the best action (nail index) given the current state.

        Args:
            canvas: (1, C, H, W)
            target_image: (1, C, H, W)
            prev_soft: (nails,) float distribution used as 'prev action' input
        Returns:
            action_idx (int)
            action_soft (Tensor, (nails,))
        """

        # canvas = target_image = (B, C, H, W) = already tensors from train.py

        # ensure shape (1, nails)
        """if prev_soft.dim() == 1:
            prev_soft_in = prev_soft.unsqueeze(0)  # (1, nails)
        else:
            prev_soft_in = prev_soft  # assume already (1, nails)"""

        self.actor.eval()
        with torch.no_grad(), autocast(dtype=torch.float16):
            canvas_resized = self.resize_input(canvas)  # (1, C, 224, 224)
            target_resized = self.resize_input(target_image)
            nail_logits, stroke_params = self.actor(canvas_resized, target_resized, prev_action_onehot)  # (1, nails)) 
            
            """if self.use_gumbel_in_select:
                # Sample a soft vector (hard=False), then pick the argmax index
                a_soft = F.gumbel_softmax(nail_logits, tau=self.gumbel_tau, hard=False)  # (1, nails)
                action_idx = a_soft.argmax(dim=-1).item()
            # not being used, only applies gumbel as of now
            else:
                # Deterministic: softmax -> argmax
                nail_probs = torch.softmax(nail_logits, dim=-1)                           # (1, nails)
                action_idx = nail_probs.argmax(dim=-1).item()"""
            
            #a_soft = F.gumbel_softmax(nail_logits, tau=self.gumbel_tau, hard=False)
            #action_idx = a_soft.argmax(dim=-1).item()    
            action_idx = nail_logits.argmax(dim=-1).item()

        self.actor.train()
        #print ("Action idx(argmax)", action_idx)
        return action_idx

    def act(self, canvas, target_image, prev_action_idx, noise_scale=0.01,episode=0, step=0):
        """"
        Behavior policy used in train.py.
        Choose action with warmup + epsilon exploration.
        Args:
            prev_soft: (nails,) Gumbel/softmax vector from previous step.
        Returns:
            action_idx (int), action_soft (nails,)
        """
        if self.use_script:
            # reset for each new episode so we replay from the start
            if step == 0:
                self.script_ptr = 0
            
            if self.script_ptr >= len(self.script):
                self.script_ptr = len(self.script) - 1  # or: raise RuntimeError("script finished")
            
            idx = int(self.script[self.script_ptr])
            self.script_ptr += 1
            return idx

        nails = self.config["nails"]
        # for the first episode, the first 500 steps are just exploration (p=1)
        """if episode < 1 and step < 500:
            action_idx = np.random.randint(0, self.config["nails"])
        else:"""
        p = np.random.rand()  # uniform in [0, 1]
        #p = 0
        # make it dynamic later (threshold=0.8 now)
        # more exploration in initial stages, more exploitation later on
        # dynamic exploration - decays    
        global_step = int(self.global_step)  # calculated above
        eps_start, eps_end, eps_decay_steps = 0.6, 0.05, 80_000
        eps = eps_end + (eps_start - eps_end) * math.exp(-global_step / eps_decay_steps)
        if p < eps:
            # exploration: pick a random nail index 
            action_idx = np.random.randint(0, self.config["nails"])
        else:
            #exploitation: use policy network (prev_soft is already a gumbel vector)
            one_hot_prev = F.one_hot(torch.tensor([prev_action_idx]), num_classes=nails).float().to(self.device)
            action_idx = self.select_action(canvas, target_image, one_hot_prev)
        return action_idx

    def update_actor_critic(self, target_image):
        """
        One DDP-friendly update step.
        - Uses local batch size (split across ranks by train.py)
        - Caches target image resize
        - Mixed precision for big memory+speed wins
        """
        #min-across-ranks gate:
        B = int(self.config.get("batch_size_local", self.config["batch_size"]))
        local_can_train = (len(self.replay_buffer) >= B)
        if not _allr_min_bool(local_can_train, self.device):
            return
        
        # might collapse if ranks are at diff stages
        #B = int(self.config.get("batch_size_local", self.config["batch_size"]))   # <<< local per-GPU batch
        #if len(self.replay_buffer) < B:
        #    return

        # ---- cadence gating: train only every N env steps ----
        TRAIN_EVERY = int(self.config.get("train_every", 1))
        # ignore warmup for now, i needed add value in config (train for the first n steps always)
        WARMUP = int(self.config.get("train_warmup_steps", 0))
        # Make cadence decisions from a SHARED step across ranks
        shared_step = _sync_min_int(self.global_step, self.device)
        # Use shared_step for all cadence logic
        if shared_step >= WARMUP and (shared_step % max(1, TRAIN_EVERY)) != 0:
            if self.is_main:
                print("[time] train_total_s: 0.000000 (train skipped)")  
            return
        
        nails = self.config["nails"]

        t_total = time.time()

        # ----- Sample replay buffer -----
        t = time.time()
        canvas, current_idx, action_idx, next_canvas, rewards, dones = self.replay_buffer.sample(B)
        current_idx = current_idx.view(-1)
        action_idx  = action_idx.view(-1)
        if self.is_main: print(f"[time] sample_s: {time.time() - t:.6f}")

        t = time.time()
        prev_onehot   = F.one_hot(current_idx, nails).float().to(self.device)  # (B, nails) at s_t
        action_onehot = F.one_hot(action_idx,  nails).float().to(self.device)  # (B, nails) a_t (buffer)
        if self.is_main: print(f"[time] onehot_s: {time.time() - t:.6f}")

        # freeze BatchNorm in targets - needs to be in eval mode
        # use the frozen stats instead of updating with every sampled batch.
        # because hidden layers has batch norm
        self.actor_target.eval()
        self.critic_target.eval()

        # ----- Resize & target caching -----
        t = time.time()
        if self.resized_target is None:
            with torch.no_grad():
                self.resized_target = self.resize_input(target_image)   # (1, C, 224, 224)
        target_resized      = self.resized_target.expand(B, -1, -1, -1)  # <<< zero-copy expand along batch
        canvas_resized      = self.resize_input(canvas)
        next_canvas_resized = self.resize_input(next_canvas)
        if self.is_main: print(f"[time] resize_s: {time.time() - t:.6f}")

        # ===================== Target Critic/ Q value =====================
        #with torch.no_grad(), autocast(dtype=torch.float16):
        t = time.time()
        with torch.no_grad():
            # Next action for target: use argmax ONE-HOT (matches discrete env) w/o gumbel
            next_logits, _ = self.actor_target(next_canvas_resized, target_resized, action_onehot)  # (B, nails)
            next_idx = next_logits.argmax(dim=-1)
            next_onehot  = F.one_hot(next_idx, nails).float()
            #next_onehot = F.gumbel_softmax(next_logits, tau=self.gumbel_tau, hard=False)             # (B, nails), straight-through
            target_Q = self.critic_target(next_canvas_resized, target_resized, next_onehot)
            target_Q = rewards + self.config["gamma"] * target_Q * (1 - dones)
        if self.is_main: print(f"[time] targetQ_s: {time.time() - t:.6f}")

        # ===================== Critic Update =====================

        #with autocast(dtype=torch.float16):
            # Current Q for the action that was actually executed
            # detach = the onehots are from replay buffer, so the actor shouldn't be optimized optimized to reproduce past replay actions
        t = time.time()
        self.critic_optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=torch.float16):
            current_Q   = self.critic(canvas_resized, target_resized, action_onehot.detach()) 
            critic_loss = F.mse_loss(current_Q, target_Q)
        #critic_loss.backward()
        self.scaler.scale(critic_loss).backward()
        # --- gradient viz before grads are cleared ---
        if self.is_main and (shared_step % 50 == 0) and (self.viz_after_backward is not None):
            try:
                self.viz_after_backward()
            except Exception as e:
                print(f"[viz critic error]: {e}")
        self.scaler.unscale_(self.critic_optimizer)
        clip_grad_norm_(self.critic.parameters(), self.config.get("clip_grad_norm", 1.0))  # ← comment out to disable
        #self.critic_optimizer.step()
        self.scaler.step(self.critic_optimizer)
        self.scaler.update()
        if self.is_main: print(f"[time] critic_s: {time.time() - t:.6f}")

        # store loss for logging
        self.last_critic_loss = float(critic_loss.detach())

        # ===================== Actor Update =====================
        # ---- actor cadence within training steps ----
        #now - actor forward+loss+backward() every training call, 
        #but step the optimizer only every second training call (4th env step).

        RUN_ACTOR_EVERY = int(self.config.get("actor_every", 2))
        # Count “train steps” using the shared step
        shared_train_idx = shared_step // max(1, TRAIN_EVERY)
        
        # start-of-actor-cycle on the first training call in each 2-call cycle
        is_actor_cycle_start = (shared_train_idx % RUN_ACTOR_EVERY) == 0
        # actor_step only on the second training call in the cycle (every 4 env steps)
        actor_step = (shared_train_idx % RUN_ACTOR_EVERY) == (RUN_ACTOR_EVERY - 1)

        for p in self.critic.parameters(): 
            p.requires_grad_(False)

        # use critic in eval mode to freeze BN running stats
        self.critic.eval()    

        # zero actor grads only at the start of the 2-call accumulation window
        if is_actor_cycle_start:
            self.actor_optimizer.zero_grad(set_to_none=True)

        t = time.time()
        #self.actor_optimizer.zero_grad(set_to_none=True)
        """with autocast(dtype=torch.float16): 
            nail_logits, _ = self.actor(canvas_resized, target_resized, prev_onehot)  # (B, nails)          
            a_soft = F.gumbel_softmax(nail_logits, tau=self.gumbel_tau, hard=False)   # (B, nails)
            #Q_sa = self.critic(canvas_resized, target_resized, a_soft)                # (B,1)
            Q_sa = self.critic(canvas_resized.detach(), target_resized.detach(), a_soft)           
            # 1) "pre-entropy" actor loss: just the Q term (negated mean Q)
            q_only_loss = -Q_sa.mean()
            # 2) ---- entropy bonus on the soft distribution BEFORE straight-through ----
            # use same temperature as Gumbel; compute entropy in fp32 for stability
            tau = max(float(self.gumbel_tau), 1e-6)
            p = F.softmax((nail_logits / tau).float(), dim=-1)                        # (B, nails)
            # Shanon entropy H(p) = - Σ_i p_i log p_i
            entropy = -(p * (p.clamp_min(1e-8).log())).sum(dim=-1).mean()             # scalar (nats)           
            # 3) alpha and entropy term
            alpha = float(self.config.get("entropy_alpha", 0.05)) # tune 0.01–0.05 (if policy collapses early,increase)
            entropy_term = alpha * entropy
            # 4) final actor loss
            #actor_loss = -Q_sa.mean() - alpha * entropy
            actor_loss = q_only_loss - entropy_term"""

        with autocast(dtype=torch.float16):
            nail_logits, _ = self.actor(canvas_resized, target_resized, prev_onehot)

            tau_loss = 1.3  # warmer ONLY for loss path
            a_soft   = F.gumbel_softmax(nail_logits, tau=tau_loss, hard=False)

            Q_sa = self.critic(canvas_resized.detach(), target_resized.detach(), a_soft)

            # --- Q normalization so entropy can compete ---
            # normalising, centering the q value/ actor loss so make sure alpha has an affect
            q_mean = Q_sa.detach().mean()
            q_std  = Q_sa.detach().std().clamp_min(1e-3)
            Q_norm = (Q_sa - q_mean) / q_std

            # --- Entropy from logits/τ (numerically stable) ---
            # same p*log(p) but more stable
            ent = torch.distributions.Categorical(logits=(nail_logits / tau_loss)).entropy().mean()

            alpha = float(self.config.get("entropy_alpha", 0.10))
            actor_loss = -(Q_norm.mean() + alpha * ent)
        
        
        #computes gradients
        self.scaler.scale(actor_loss).backward()
        # --- gradient viz before grads are cleared ---
        if self.is_main and (shared_step % 50 == 0) and (self.viz_after_backward is not None):
            try:
                self.viz_after_backward()
            except Exception as e:
                print(f"[viz actor error]: {e}")
        if actor_step:
            self.scaler.unscale_(self.actor_optimizer)
            clip_grad_norm_(self.actor.parameters(), self.config.get("clip_grad_norm", 1.0))   # ← comment out to disable
            #updates the model weights acc to the gradients
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
            # ready for next accumulation window
            self.actor_optimizer.zero_grad(set_to_none=True)

        # store loss + policy stats (optional)
        self.last_actor_loss = float(actor_loss.detach())
        #self.last_entropy    = float(entropy.detach())
        #self.last_tau        = float(tau)
        #self.last_alpha      = float(alpha)

        # ---------- W&B logging (main rank only; once per actor step) ----------
        """if self.is_main and actor_step:
            # env-step on x-axis (“global step” on plots)
            step_val = int(shared_step)
            wandb.log({
                "actor loss/q_only":          float(q_only_loss.detach()),   # before entropy
                "actor loss/alpha":           float(alpha),                  # alpha value
                "actor loss/entropy":         float(entropy.detach()),       # entropy value
                "actor loss/alpha_x_entropy": float(entropy_term.detach()),
                "actor loss/final":           float(actor_loss.detach()),    # final actor loss
            }, step=step_val)"""

        if self.is_main and actor_step:
            step_val = int(shared_step)
            wandb.log({
                # Actor loss terms
                "actor loss/q_norm_mean":     float(Q_norm.mean().detach()),
                "actor loss/q_mean_raw":      float(q_mean),
                "actor loss/q_std_raw":       float(q_std),
                "actor loss/alpha":           float(alpha),
                "actor loss/entropy":         float(ent.detach()),
                "actor loss/alpha_x_entropy": float(alpha * ent.detach().mean()),
                "actor loss/final":           float(actor_loss.detach()),

                # Policy health (no extra compute: derived from a_soft)
                "policy/top1_prob":           float(a_soft.max(dim=-1).values.mean().detach()),
                #"policy/tau_loss":            float(tau_loss),
            }, step=step_val)


    

        if self.is_main:
            wandb.log({
                "dbg/shared_step": int(shared_step),
                "dbg/shared_train_idx": int(shared_train_idx),
                "dbg/is_actor_cycle_start": int(is_actor_cycle_start),
                "dbg/actor_step_flag": int(actor_step),
                "dbg/TRAIN_EVERY": int(TRAIN_EVERY),
                "dbg/RUN_ACTOR_EVERY": int(RUN_ACTOR_EVERY),
                "dbg/critic_training_mode": int(self.critic.training),
                "dbg/any_critic_requires_grad": int(any(p.requires_grad for p in self.critic.parameters())),
                "dbg/scaler_scale": float(self.scaler.get_scale()),
            }, step=int(shared_step))
    

        if self.is_main:
            if actor_step: 
                print(f"[time] actor_s: {time.time() - t:.6f}")
            else:
                print(f"[time] actor_s: 0.000000 (accumulating) ")    

        # restore critic for the next critic update
        self.critic.train()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        # ===================== Soft target updates =====================
        t = time.time()
        if (shared_step % 5) == 0:
            self.soft_update(self.critic, self.critic_target, self.config["tau"])
            self.soft_update(self.actor,  self.actor_target,  self.config["tau"])
        if self.is_main: print(f"[time] soft_s: {time.time() - t:.6f}")

        # ===================== Hooks / viz =====================
        """t = time.time()
        if (self.viz_after_backward is not None) and (shared_step % 50 == 0):
            try:
                self.viz_after_backward()
            except Exception:
                pass
        if self.is_main: print(f"[time] hooks_s: {time.time() - t:.6f}")"""
        
        if self.is_main:
            print(f"[time] train_total_s: {time.time() - t_total:.6f}")           


    def train(self, target_image):
        """Wrapper to update actor and critic networks."""
        self.update_actor_critic(target_image)

    """def soft_update(self, local_model, target_model, tau):
        
        Perform soft target network update.

        Args:
            local_model: Actor or Critic main network.
            target_model: Corresponding target network.
            tau (float): Soft update coefficient (0 < tau << 1).
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)"""

    # faster approach
    @torch.no_grad()
    def soft_update(self, local_model, target_model, tau: float):
        tps = [p.data for p in target_model.parameters()]
        lps = [p.data for p in local_model.parameters()]
        # tp = (1-tau)*tp + tau*lp
        torch._foreach_mul_(tps, 1.0 - tau)
        torch._foreach_add_(tps, lps, alpha=tau)

if __name__ == "__main__":
    """
    Quick diagnostic test for DDPGAgent setup and API:
    - Builds dummy Actor/Critic
    - Runs .act() and .train() with fake canvas data
    - Verifies integration works without crashing
    """
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    import numpy as np
    from .actor import Actor
    from .critic import Critic
    from ..utils.replay_buffer import ReplayBuffer
    from ..utils.noise import OUNoise

    print("Running DDPGAgent standalone integration test...")

    # Config setup
    config = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "image_size": (224, 224),
        "batch_size": 4,
        "gamma": 0.99,
        "tau": 0.005
    }

    B, C, H, W = config["batch_size"], 3, *config["image_size"]
    action_dim = 6

    # Models
    actor = Actor("resnet18", "resnet18", pretrained=False, out_neurons=action_dim, in_channels=C)
    critic = Critic("resnet18", "resnet18", pretrained=False, out_neurons=1, in_channels=C)
    actor_target = Actor("resnet18", "resnet18", pretrained=False, out_neurons=action_dim, in_channels=C)
    critic_target = Critic("resnet18", "resnet18", pretrained=False, out_neurons=1, in_channels=C)

    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # Agent and utils
    replay_buffer = ReplayBuffer(capacity=100)
    noise = OUNoise(action_dim)

    agent = DDPGAgent(
        actor=actor,
        critic=critic,
        actor_target=actor_target,
        critic_target=critic_target,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        replay_buffer=replay_buffer,
        noise=noise,
        config=config,
        channels=C
    )

    # Dummy tensors
    canvas = np.random.rand(C, H, W).astype(np.float32)
    target = np.random.rand(C, H, W).astype(np.float32)
    prev_action = np.zeros(action_dim, dtype=np.float32)

    # Test .act()
    try:
        action = agent.act(canvas, target, prev_action)
        print("DDPGAgent.act() output:", action)
    except Exception as e:
        print("DDPGAgent.act() failed:", e)

    # Populate buffer
    for _ in range(B):
        next_canvas = np.random.rand(C, H, W).astype(np.float32)
        reward = 1.0
        done = 0.0
        replay_buffer.store(canvas, prev_action, action, next_canvas, reward, done)

    # Test .train()
    try:
        agent.train(target)
        print("DDPGAgent.train() ran without error.")
    except Exception as e:
        print("DDPGAgent.train() failed:", e)
