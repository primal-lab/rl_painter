# needs to be reviewed - add proper comments 
import numpy as np

class OUNoise:
    """
    Ornstein-Uhlenbeck process for generating noise, often used in reinforcement learning
    to facilitate exploration by adding temporally correlated noise to actions.
    """
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state
