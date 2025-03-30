import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

points = [
    (0, 2, 73),
    (1, 2, 129),
    (2, 2, 101),
    (0, 1, 194),
    (1, 1, 43),
    (2, 1, 252),
    (0, 0, 179),
    (1, 0, 172),
    (2, 0, 229)
]

points2 = [
    (0, 2, 110),
    (1, 2, 132),
    (2, 2, 150),
    (0, 1, 132),
    (1, 1, 152),
    (2, 1, 154),
    (0, 0, 147),
    (1, 0, 178),
    (2, 0, 174)
]

x, y, z = zip(*points2)
x, y, z = np.array(x), np.array(y), np.array(z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

dx = dy = 0.5  
dz = z

ax.bar3d(x, y, np.zeros_like(z), dx, dy, dz, shade=True)


plt.show()