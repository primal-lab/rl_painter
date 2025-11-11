import numpy as np

class OUNoise:
    """
    Ornstein-Uhlenbeck process for generating noise, often used in reinforcement learning
    to facilitate exploration by adding temporally correlated noise to actions.
    """
    # in train.py
    # theta = 0.15, sigma = 0.2
    def __init__(self, size, mu=0.0, theta=0.05, sigma=0.4):
        # action dim
        self.size = size 
        # kind of like a target value - noise stays around this value
        self.mu = mu 
        # controls how quickly the noise gets pulled back toward mu
        # how persistent the exploration can be
        # higher the theta, lesser time needed => less exploration, more stable
        self.theta = theta
        # controls how strong (how big) the random part of the noise is
        # big sigma → noise is large → actions jump around more → more exploration
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    # in act() in ddpg.py
    def sample(self):
        # new noise = "pull toward mu" + "randomness"
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state

# since actions are very close to each other:

# increase sigma (how much noise can help explore diff actions) -> more exploration
# decrease theta (how strongly the noise gets pulled back to mu) -> noise will wander more freely

# increase noise_scale (how much of the OUNoise is actually added to the action)
# decrease noise_decay (how much the noise_scale decays per episode)