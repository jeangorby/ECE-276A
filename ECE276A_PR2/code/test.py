import numpy as np
import matplotlib.pyplot as plt
# plt.ioff()
from odometry import load_and_sync_data, compute_odometry
from scan_matching import pose2d_to_SE3, run_scan_matching
# from load_data import np  # Ensure Hokuyo LiDAR data is loaded correctly

def lidar_to_point_cloud(ranges, angle_min, angle_increment, max_range=30, min_range=0.1):
    """Convert 1D array of LiDAR ranges into 2D (x, y) points in the LiDAR frame."""
    ranges = np.squeeze(ranges)  # Ensure it's a 1D array
    angles = angle_min + np.arange(len(ranges)) * angle_increment[0]

    # Ensure valid mask has the correct shape
    valid = np.logical_and(ranges > min_range, ranges < max_range)
    
    # Apply mask to both ranges and angles
    x = ranges[valid] * np.cos(angles[valid])
    y = ranges[valid] * np.sin(angles[valid])

    return np.vstack((x, y))  # Shape (2, N)

def transform_to_world(odom_pose, lidar_scan, angle_min, angle_increment):
    """Transform LiDAR points from LiDAR frame to World frame using odometry."""
    lidar_pc = lidar_to_point_cloud(lidar_scan, angle_min, angle_increment)
    
    # Convert LiDAR points to homogeneous coordinates (3D)
    lidar_pc_h = np.vstack((lidar_pc, np.zeros((1, lidar_pc.shape[1])), np.ones((1, lidar_pc.shape[1]))))

    # Define the transformation from LiDAR frame to Robot frame (T_lidar)
    T_lidar = np.array([
        [1, 0, 0, 0.2],  # LiDAR is offset by 0.2m in X (adjust as needed)
        [0, 1, 0, 0.0],  # No Y offset
        [0, 0, 1, 0.15], # LiDAR is 0.15m above ground
        [0, 0, 0, 1]
    ])

    # Compute full transformation: World = Robot * LiDAR * LiDAR points
    world_pc = odom_pose @ T_lidar @ lidar_pc_h

    return world_pc[:2, :]  # Return only (x, y)

if __name__ == "__main__":
    # --------- Load Data ---------
    encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps = load_and_sync_data(dataset=20)
    x, y, theta = compute_odometry(encoder_counts, encoder_stamps, imu_angular_velocity)

    # Convert (x, y, theta) to transformation matrices
    odom_poses = [pose2d_to_SE3(x[i], y[i], theta[i]) for i in range(len(x))]

    # Load LiDAR scan data
    hokuyo_data = np.load(f"../data/Hokuyo20.npz")
    lidar_ranges_all = hokuyo_data["ranges"]  # Shape (1081, T)
    angle_min = hokuyo_data["angle_min"]
    angle_increment = hokuyo_data["angle_increment"]

    lidar_scans = []
    for i in range(lidar_ranges_all.shape[1]):
        scan_i = lidar_ranges_all[:, i]
        # Possibly filter out invalid ranges
        # ...
        lidar_scans.append(scan_i)

    # --------- Compute ICP-Enhanced Trajectory ---------
    refined_poses = run_scan_matching(odom_poses, lidar_scans, angle_min, angle_increment)

    # Ensure the number of scans and poses match
    num_scans = lidar_ranges_all.shape[1]  # Total LiDAR scans
    num_odom_poses = len(odom_poses)  # Raw odometry poses
    num_icp_poses = len(refined_poses)  # ICP poses

    # # Determine the minimum count to avoid index errors
    num_frames = min(num_scans, num_odom_poses, num_icp_poses)

    # --------- Process All Scans for Both Trajectories ---------
    all_world_points_odom = []
    all_world_points_icp = []

    for i in range(num_frames):  # Iterate safely
        lidar_scan = lidar_ranges_all[:, i]  # Get scan at time i
        
        # Using raw odometry
        world_points_odom = transform_to_world(odom_poses[i], lidar_scan, angle_min, angle_increment)
        all_world_points_odom.append(world_points_odom)
        
        # Using ICP-refined trajectory
        world_points_icp = transform_to_world(refined_poses[i], lidar_scan, angle_min, angle_increment)
        all_world_points_icp.append(world_points_icp)

    # Convert lists to arrays for plotting
    all_world_points_odom = np.hstack(all_world_points_odom)
    all_world_points_icp = np.hstack(all_world_points_icp)


    # --------- Plot LiDAR Point Cloud + Trajectory (Odometry vs ICP) ---------
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    # --- Plot Odometry-Based LiDAR Mapping ---
    axs[0].plot(x, y, 'b-', marker="o", markersize=2, label="Robot Trajectory (IMU+Encoder)")
    axs[0].scatter(all_world_points_odom[0, :], all_world_points_odom[1, :], s=1, c="black", label="LiDAR Points")
    axs[0].set_xlabel("X Position (m)")
    axs[0].set_ylabel("Y Position (m)")
    axs[0].set_title("LiDAR Mapping with Motion Model")
    axs[0].axis("equal")
    axs[0].legend()
    axs[0].grid(True)

    # --- Plot ICP-Based LiDAR Mapping ---
    refined_x = [T[0, 3] for T in refined_poses]
    refined_y = [T[1, 3] for T in refined_poses]
    
    axs[1].plot(refined_x, refined_y, 'r-', marker="o", markersize=2, label="Optimized Trajectory (ICP)")
    axs[1].scatter(all_world_points_icp[0, :], all_world_points_icp[1, :], s=1, c="black", label="LiDAR Points")
    axs[1].set_xlabel("X Position (m)")
    axs[1].set_ylabel("Y Position (m)")
    axs[1].set_title("LiDAR Scan Matching with ICP Trajectory")
    axs[1].axis("equal")
    axs[1].legend()
    axs[1].grid(True)

    plt.show()

