from kinematics import NLinkArm
from visualize import plot_cartesian_trajectory
import numpy as np

dh_params = [
    [0.0,        0.0 / 1000,   243.3 / 1000,  0.0],
    [np.pi/2,    0.0 / 1000,    30.0 / 1000,  np.pi/2],
    [np.pi,    280.0 / 1000,    20.0 / 1000,  np.pi/2],
    [np.pi/2,    0.0 / 1000,   245.0 / 1000,  np.pi/2],
    [np.pi/2,    0.0 / 1000,    57.0 / 1000,  0.0],
    [-np.pi/2,   0.0 / 1000,   235.0 / 1000, -np.pi/2]
]

robot = NLinkArm(dh_params)

path = np.array([
    [3.00055741,  0.42024106,  3.80467259, -0.08676108,  6.28295253,  4.76474886], 
    [5.6204529,  -0.0588321,   2.35204525,  3.6034775,   6.28663955,  4.76474886]
])
plot_cartesian_trajectory(path, robot)