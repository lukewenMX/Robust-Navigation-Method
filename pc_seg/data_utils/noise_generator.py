'''
Random Process Noise Generation for random controller
'''
import numpy as np
import matplotlib.pyplot as plt

class OrnsteinUhlenbeckNoise():
    def __init__(self, mu, sigma, theta=0.15, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        # self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self, dt):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * dt + self.sigma * np.sqrt(dt) * np.random.normal()
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else 0

if __name__ == "__main__":
    noise = OrnsteinUhlenbeckNoise(mu=0, sigma=0.01)
    x = range(10000)
    y = [noise(dt=0.01) for i in x]
    plt.title("OU Noise")
    plt.plot(x, y)
    plt.show()
