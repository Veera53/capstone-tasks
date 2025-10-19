import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load IMU control inputs
# -------------------------------
data = pd.read_csv("imu_data.csv")

# Flip velocity to positive forward motion
data['v'] = -data['v']

# Estimate timestep
dt = np.mean(np.diff(data['time']))

# -------------------------------
# 2. Initialize EKF variables
# -------------------------------
x_est = np.zeros((3, 1))   # [x, y, theta]
P = np.eye(3) * 0.01       # Covariance (not plotted here)
sigma_v, sigma_w = 0.08, 0.03
Q = np.diag([sigma_v**2, sigma_w**2])
x_hist = []

# -------------------------------
# 3. EKF Prediction Loop (IMU)
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

    # Covariance prediction (kept for completeness)
    P = F @ P @ F.T + G @ Q @ G.T

    # Save state
    x_est = x_pred
    x_hist.append(x_est.flatten())

x_hist = np.array(x_hist)

# -------------------------------
# 4. Plot trajectory (clean view)
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_hist[:, 0], x_hist[:, 1], 'b-', linewidth=2, label='EKF IMU-only Prediction')

# Heading arrows every 10 steps
idxs = np.arange(0, len(x_hist), 10)
ax.quiver(x_hist[idxs, 0], x_hist[idxs, 1],
          np.cos(x_hist[idxs, 2]), np.sin(x_hist[idxs, 2]),
          color='r', scale=15, width=0.005, label='Heading')

# Styling
ax.set_title("EKF IMU-Only Prediction (10s IMU Data)", fontsize=14)
ax.set_xlabel("X position [m]", fontsize=12)
ax.set_ylabel("Y position [m]", fontsize=12)
ax.axis("equal")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
