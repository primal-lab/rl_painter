"""
Unit Tests for Deep RL Painter

This file includes automated unit tests for the key components of the reinforcement learning
painting pipeline, including the environment, DDPG agent, and reward functions.

Tests:
- Environment reset and step
- Agent action selection
- Agent parameter updates
- Reward function outputs
"""

import torch
import unittest
import os
import numpy as np
from env.environment import PaintingEnv
from models.actor import Actor
from models.critic import Critic
from models.ddpg import DDPGAgent
from utils.replay_buffer import ReplayBuffer
from utils.noise import OUNoise
from config import config
import lpips


class TestDeepRLPainter(unittest.TestCase):
    def setUp(self):
        """
        Set up a minimal environment and agent configuration before each test.
        """
        self.device = config["device"]
        self.target_image_path = config["target_image_path"]
        self.canvas_size = config["image_size"]
        self.action_dim = 6
        self.in_channels = 3
        self.batch_size = 4

        self.env = PaintingEnv(
            target_image_path=self.target_image_path,
            canvas_size=self.canvas_size,
            max_strokes=5,
            device=self.device
        )

        self.target_image = self.env.get_target_tensor().to(self.device)

        self.actor = Actor(
            image_encoder_model=config["model_name"],
            image_encoder_model_2=config["model_name"],
            pretrained=False,
            out_neurons=self.action_dim,
            in_channels=self.in_channels
        )

        self.critic = Critic(
            image_encoder_model=config["model_name"],
            image_encoder_model_2=config["model_name"],
            pretrained=False,
            out_neurons=1,
            in_channels=self.in_channels
        )

        self.agent = DDPGAgent(
            actor=self.actor,
            critic=self.critic,
            actor_target=self.actor,
            critic_target=self.critic,
            actor_optimizer=torch.optim.Adam(self.actor.parameters(), lr=1e-4),
            critic_optimizer=torch.optim.Adam(self.critic.parameters(), lr=1e-3),
            replay_buffer=ReplayBuffer(capacity=100),
            noise=OUNoise(self.action_dim),
            config=config,
            channels=self.in_channels
        )

    def test_env_reset(self):
        """
        Ensure that environment reset returns a valid canvas tensor.
        """
        canvas = self.env.reset()
        self.assertIsInstance(canvas, np.ndarray)

    def test_env_step(self):
        """
        Check that stepping through the environment returns valid outputs.
        """
        canvas = self.env.reset()
        dummy_action = np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
        next_canvas, reward, done = self.env.step(dummy_action)
        self.assertEqual(canvas.shape, next_canvas.shape)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_agent_select_action(self):
        """
        Verify that the agent produces a valid action within [-1, 1] bounds.
        """
        canvas = self.env.reset()
        prev_action = np.zeros(self.action_dim, dtype=np.float32)
        action = self.agent.select_action(canvas, self.target_image.cpu().numpy(), prev_action)
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertTrue(np.all(action >= -1) and np.all(action <= 1))

    def test_agent_update(self):
        """
        Populate replay buffer and verify that training step runs without error.
        """
        for _ in range(self.batch_size):
            canvas = self.env.reset()
            prev_action = np.zeros(self.action_dim, dtype=np.float32)
            action = np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
            next_canvas, reward, done = self.env.step(action)
            self.agent.replay_buffer.store(canvas, prev_action, action, next_canvas, reward, done)

        try:
            self.agent.train(self.target_image.cpu().numpy())
        except Exception as e:
            self.fail(f"Agent training raised an exception: {e}")

    def test_reward_functions(self):
        """
        Test SSIM, MSE, and LPIPS reward functions for output shape and execution.
        """
        from env.reward import calculate_ssim_reward, calculate_mse_reward, calculate_lpips_reward

        B, C, H, W = 2, 3, *self.canvas_size
        prev_canvas = torch.rand(B, C, H, W).to(self.device)
        curr_canvas = torch.rand(B, C, H, W).to(self.device)
        target = torch.rand(B, C, H, W).to(self.device)

        ssim = calculate_ssim_reward(prev_canvas, curr_canvas, target)
        mse = calculate_mse_reward(prev_canvas, curr_canvas, target)
        lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
        lpips_val = calculate_lpips_reward(prev_canvas, curr_canvas, target, lpips_fn)

        self.assertEqual(ssim.shape, (B, 1))
        self.assertEqual(mse.shape, (B, 1))
        self.assertEqual(lpips_val.shape, (B, 1))


if __name__ == '__main__':
    unittest.main()
