import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# -------------------------------
# 1. Load odometry control inputs
# -------------------------------
data = pd.read_csv("odom_data.csv")

# Make v positive (if encoders gave negative velocities)
data['v'] = -data['v']

# Estimate time step automatically
dt = np.mean(np.diff(data['time']))

# -------------------------------
# 2. EKF Initialization
# -------------------------------
x_est = np.zeros((3, 1))   # [x, y, theta]
P = np.eye(3) * 0.01       # Initial covariance
sigma_v, sigma_w = 0.05, 0.02
Q = np.diag([sigma_v**2, sigma_w**2])
x_hist, P_hist = [], []

# -------------------------------
# 3. EKF Prediction Loop
# -------------------------------
for i in range(len(data)):
    v = data.loc[i, 'v']
    w = data.loc[i, 'w']
    theta = x_est[2, 0]

    # Nonlinear motion model
    x_pred = x_est + np.array([
        [v * dt * np.cos(theta)],
        [v * dt * np.sin(theta)],
        [w * dt]
    ])

    # Jacobians
    F = np.array([
        [1, 0, -v * dt * np.sin(theta)],
        [0, 1,  v * dt * np.cos(theta)],
        [0, 0, 1]
    ])
    G = np.array([
        [dt * np.cos(theta), 0],
        [dt * np.sin(theta), 0],
        [0, dt]
    ])

    # Covariance prediction
    P = F @ P @ F.T + G @ Q @ G.T

    # Save
    x_est = x_pred
    x_hist.append(x_est.flatten())
    P_hist.append(P.copy())

x_hist = np.array(x_hist)
P_hist = np.array(P_hist)

# -------------------------------
# 4. Helper: plot covariance ellipse
# -------------------------------
def plot_cov_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    eigvals, eigvecs = np.linalg.eig(cov[:2, :2])
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(xy=mean[:2], width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

# -------------------------------
# 5. Plot the trajectory
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_hist[:, 0], x_hist[:, 1], 'b-', linewidth=2, label='EKF Odometry Prediction')

# Plot heading arrows every 10 steps
idxs = np.arange(0, len(x_hist), 10)
ax.quiver(x_hist[idxs, 0], x_hist[idxs, 1],
          np.cos(x_hist[idxs, 2]), np.sin(x_hist[idxs, 2]),
          color='r', scale=15, width=0.005, label='Heading')


ax.set_title("EKF Odometry-Only Prediction (10s Encoder Data)", fontsize=14)
ax.set_xlabel("X position [m]", fontsize=12)
ax.set_ylabel("Y position [m]", fontsize=12)
ax.axis("equal")
ax.grid(True)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
