from matplotlib import transforms
import numpy as np
import matplotlib.pyplot as plt

cloud_predict_w = np.load("cloud_predict.npy")
# img_predict_w = np.load("img_predict.npy")
gt_w = np.load("gt.npy")
random_w = np.load("random.npy")

x_axis = list(range(random_w.shape[0]))

plt.title("Result Analysis")
plt.plot(x_axis, cloud_predict_w, label = "cloud predicted angular vel")
# plt.plot(x_axis, img_predict_w, label = "image predicted angular vel")
# plt.plot(x_axis, random_w, label = "random angular vel")
plt.plot(x_axis, gt_w, label = "ground truth angular vel")
plt.legend()

plt.xlabel("timestamp")
plt.ylabel(r"$\mathrm{rad}^{-1}$")
plt.show()