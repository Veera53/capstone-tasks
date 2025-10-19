import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Load data
# -------------------------------
odom = pd.read_csv("odom_data.csv")
imu = pd.read_csv("imu_data.csv")

# Flip velocity to positive forward motion
odom['v'] = -odom['v']
imu['v'] = -imu['v']

# Estimate timestep
dt = np.mean(np.diff(odom['time']))

# -------------------------------
# EKF Function
# -------------------------------
def ekf_prediction(v_data, w_data, Q_val, R_val=None, w_meas=None):
    x_est = np.zeros((3,1))
    P = np.eye(3) * 0.01
    x_hist = []

    for i in range(len(v_data)):
        v = v_data[i]
        w = w_data[i]

        theta = x_est[2,0]

        # Prediction
        x_pred = x_est + np.array([
            [v*dt*np.cos(theta)],
            [v*dt*np.sin(theta)],
            [w*dt]
        ])
        F = np.array([
            [1, 0, -v*dt*np.sin(theta)],
            [0, 1, v*dt*np.cos(theta)],
            [0, 0, 1]
        ])
        G = np.array([
            [dt*np.cos(theta),0],
            [dt*np.sin(theta),0],
            [0, dt]
        ])
        P_pred = F @ P @ F.T + G @ Q_val @ G.T

        # Update (if measurement provided)
        if w_meas is not None and R_val is not None:
            z = np.array([[w_meas[i]]])
            H = np.array([[0,0,1]])
            y = z - H @ x_pred
            S = H @ P_pred @ H.T + R_val
            K = P_pred @ H.T @ np.linalg.inv(S)
            x_est = x_pred + K @ y
            P = (np.eye(3) - K @ H) @ P_pred
        else:
            x_est = x_pred
            P = P_pred

        x_hist.append(x_est.flatten())

    return np.array(x_hist)

# -------------------------------
# Noise matrices
# -------------------------------
sigma_v, sigma_w_odom = 0.05, 0.02
Q = np.diag([sigma_v**2, sigma_w_odom**2])
sigma_w_imu = 0.03
R = np.array([[sigma_w_imu**2]])

# -------------------------------
# Run EKF for three cases
# -------------------------------
# 1. Odometry only
x_odom = ekf_prediction(odom['v'].values, odom['w'].values, Q_val=Q)

# 2. IMU only
x_imu = ekf_prediction(imu['v'].values, imu['w'].values, Q_val=np.diag([0.08**2,0.03**2]))

# 3. Fused EKF (odom as control, IMU w as measurement)
n_steps = min(len(odom), len(imu))
x_fused = ekf_prediction(odom['v'].values[:n_steps], 
                         odom['w'].values[:n_steps],
                         Q_val=Q, R_val=R, w_meas=imu['w'].values[:n_steps])

# -------------------------------
# Plot trajectories
# -------------------------------
fig, ax = plt.subplots(figsize=(8,8))
ax.plot(x_odom[:,0], x_odom[:,1], 'b-', linewidth=2, label='Odometry Only')
ax.plot(x_imu[:,0], x_imu[:,1], 'orange', linewidth=2, label='IMU Only')
ax.plot(x_fused[:,0], x_fused[:,1], 'g-', linewidth=2, label='Fused EKF')

# Heading arrows every 10 steps
for data, color in zip([x_odom, x_imu, x_fused], ['b','orange','g']):
    idxs = np.arange(0, len(data), 10)
    ax.quiver(data[idxs,0], data[idxs,1], np.cos(data[idxs,2]), np.sin(data[idxs,2]),
              color=color, scale=15, width=0.005)

ax.set_title("EKF: Odometry vs IMU vs Fused", fontsize=14)
ax.set_xlabel("X position [m]", fontsize=12)
ax.set_ylabel("Y position [m]", fontsize=12)
ax.axis("equal")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
