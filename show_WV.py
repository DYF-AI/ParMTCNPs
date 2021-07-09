import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 只能显示两个和三个目标的向量
data = np.loadtxt('wv_test_M2.csv')
print(data.shape[0])
print('data:', data)  # 二维数组

fig = plt.figure()
ax = Axes3D(fig)

x, y, z = data[:, 0], data[:, 1], data[:, 2]

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.scatter(x, y, z, marker='.', s=50, label='',color='r')

VecStart_x = np.zeros(data.shape[0])
VecStart_y = np.zeros(data.shape[0])
VecStart_z = np.zeros(data.shape[0])
VecEnd_x = data[:, 0]
VecEnd_y = data[:, 1]
VecEnd_z = data[:, 2]

for i in range(VecStart_x.shape[0]):
    ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i], VecEnd_y[i]], zs=[VecStart_z[i], VecEnd_z[i]])

plt.show()