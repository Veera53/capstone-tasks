import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load odometry and IMU data
# -------------------------------
odom = pd.read_csv("odom_data.csv")
imu = pd.read_csv("imu_data.csv")

# Flip velocity to positive forward motion
odom['v'] = -odom['v']
imu['v'] = -imu['v']

# Estimate timestep
dt = np.mean(np.diff(odom['time']))

# -------------------------------
# 2. EKF Initialization
# -------------------------------
x_est = np.zeros((3, 1))       # [x, y, theta]
P = np.eye(3) * 0.01           # Initial covariance

# Noise matrices
sigma_v, sigma_w_odom = 0.05, 0.02   # Odometry control noise
sigma_w_imu = 0.03                   # IMU measurement noise

Q = np.diag([sigma_v**2, sigma_w_odom**2])   # Process noise
R = np.array([[sigma_w_imu**2]])             # Measurement noise

# Storage for plotting
x_hist = []

# -------------------------------
# 3. EKF Prediction + Update Loop
# -------------------------------
n_steps = min(len(odom), len(imu))

for i in range(n_steps):
    v = odom.loc[i, 'v']
    w = odom.loc[i, 'w']

    theta = x_est[2, 0]

    # ---- Prediction Step ----
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
    P_pred = F @ P @ F.T + G @ Q @ G.T

    # ---- Update Step ----
    z = np.array([[imu.loc[i, 'w']]])   # Measurement from IMU
    H = np.array([[0, 0, 1]])
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_est = x_pred + K @ y
    P = (np.eye(3) - K @ H) @ P_pred

    # Store
    x_hist.append(x_est.flatten())

x_hist = np.array(x_hist)

# -------------------------------
# 4. Plot fused EKF trajectory
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_hist[:, 0], x_hist[:, 1], 'b-', linewidth=2, label='EKF Odometry+IMU Fusion')

# Plot heading arrows every 10 steps
idxs = np.arange(0, len(x_hist), 10)
ax.quiver(x_hist[idxs, 0], x_hist[idxs, 1],
          np.cos(x_hist[idxs, 2]), np.sin(x_hist[idxs, 2]),
          color='r', scale=15, width=0.005, label='Heading')

ax.set_title("EKF Odometry+IMU Fusion (10s Data)", fontsize=14)
ax.set_xlabel("X position [m]", fontsize=12)
ax.set_ylabel("Y position [m]", fontsize=12)
ax.axis("equal")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
