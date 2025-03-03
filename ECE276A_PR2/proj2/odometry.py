import numpy as np
import matplotlib.pyplot as plt

def load_and_sync_data(dataset=20):

    with np.load("../data/Encoders%d.npz"%dataset) as data:
        encoder_counts = data["counts"] # 4 x n encoder counts
        encoder_stamps = data["time_stamps"] # encoder time stamps

    with np.load("../data/Imu%d.npz"%dataset) as data:
        imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"] # accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements



    # Synchronize IMU data to encoder timestamps
    synced_imu_angular_velocity = np.zeros((3, len(encoder_stamps)))
    synced_imu_stamps = np.zeros(len(encoder_stamps))

    for i in range(len(encoder_stamps)):
        idx = np.argmin(np.abs(encoder_stamps[i] - imu_stamps))  # Find closest IMU timestamp
        synced_imu_angular_velocity[:, i] = imu_angular_velocity[:, idx]  # Assign corresponding IMU data
        synced_imu_stamps[i] = imu_stamps[idx]

    return encoder_counts, encoder_stamps, synced_imu_angular_velocity, synced_imu_stamps

def compute_odometry(encoder_counts, encoder_stamps, synced_imu_angular_velocity):
    meters_per_tick = 0.0022
    # Compute wheel displacements using synced encoder data
    right_wheel_travel = (encoder_counts[0, :] + encoder_counts[2, :]) / 2 * meters_per_tick
    left_wheel_travel = (encoder_counts[1, :] + encoder_counts[3, :]) / 2 * meters_per_tick

    # Compute synchronized time steps
    time_diffs_synced = np.diff(encoder_stamps, prepend=encoder_stamps[0])

    # Initialize pose (assume robot starts at origin)
    x, y, theta = [0], [0], [0]

    for i in range(1, len(encoder_stamps)):
        dt = time_diffs_synced[i]

        # Compute linear velocity v_t from average wheel travel / dt
        dist_right = right_wheel_travel[i]
        dist_left  = left_wheel_travel[i]
        dist_avg   = 0.5 * (dist_right + dist_left)
        v_t = dist_avg / dt  # m/s

        # Yaw rate from IMU
        omega_t = synced_imu_angular_velocity[2, i]  # z-axis rotation, rad/s

        # Euler update
        x_new = x[-1] + dt * v_t * np.cos(theta[-1])
        y_new = y[-1] + dt * v_t * np.sin(theta[-1])
        theta_new = theta[-1] + dt * omega_t

        x.append(x_new)
        y.append(y_new)
        theta.append(theta_new)

    return np.array(x), np.array(y), np.array(theta)

if __name__ == "__main__":
    # --------- A) Load and sync data ---------
    encoder_counts, encoder_stamps, synced_imu_angular_velocity, synced_imu_stamps = load_and_sync_data(dataset=20)

    # --------- B) Compute odometry (x,y,theta) ---------
    x, y, theta = compute_odometry(encoder_counts, encoder_stamps, synced_imu_angular_velocity)

    # Plot the final trajectory with synchronized data
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker="o", markersize=1, linestyle="-", linewidth=1, color="blue", label="Robot Trajectory (Synced Data)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Robot Trajectory from Encoder and IMU Odometry")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")  # Keep aspect ratio
    plt.savefig("./results/odometry_20.png")
    # Show the plot
    plt.show()