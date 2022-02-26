import numpy as np
import matplotlib.pyplot as plt
gt = np.load("whole_nanyanglink.npy")
# traj = np.load("test.npy")

# print(traj.shape)
plt.plot(gt[:,0],gt[:,1])
# plt.plot(traj[:,0],traj[:,1])
plt.xlim((-100,300))
plt.ylim((-100,300))
plt.show()
