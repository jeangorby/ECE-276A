import numpy as np
import matplotlib.pyplot as plt
from odometry import load_and_sync_data, compute_odometry
from icp import icp, find_best_icp_init

def pose2d_to_SE3(x, y, theta):
    """
    Embed a 2D pose (x, y, theta) into a 4x4 transformation with z=0.
    """
    T = np.eye(4)
    c = np.cos(theta)
    s = np.sin(theta)
    T[0,0] = c;   T[0,1] = -s
    T[1,0] = s;   T[1,1] = c
    T[0,3] = x;   T[1,3] = y
    return T

def lidar_to_point_cloud(ranges, angle_min, angle_increment):
    """
    Convert 1D array of ranges into a 3xN point cloud (x,y,z=0).
    angles = angle_min + i * angle_increment
    """
    N = len(ranges)
    angles = angle_min + np.arange(N) * angle_increment
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    z = np.zeros_like(x)
    return np.vstack((x, y, z))

def run_scan_matching(odom_poses, lidar_scans, angle_min, angle_increment):
    """
    odom_poses: list of 4x4 transforms from odometry (length T).
    lidar_scans: list of LiDAR scans (each is array of ranges).
    angle_min, angle_increment: sensor specs.

    We'll refine each consecutive pair's relative transform using ICP.
    Return: a list of 4x4 transforms (global_poses) after ICP refinement.
    """
    # Start the global pose with the first odometry pose
    global_poses = [odom_poses[0].copy()]

    # Convert first scan to point cloud
    prev_pts = lidar_to_point_cloud(lidar_scans[0], angle_min, angle_increment)

    # Keep track of the current global pose
    T_global = odom_poses[0].copy()
    num_scans = len(lidar_scans)
    num_poses = len(odom_poses)

    # We only iterate up to the smaller of the two
    num_frames = min(num_scans, num_poses)
    for i in range(1, num_frames):
        print("scan: ", i)

        # Current scan -> point cloud
        curr_pts = lidar_to_point_cloud(lidar_scans[i], angle_min, angle_increment)

        # Odometry relative transform from i-1 to i
        T_odom_rel = np.linalg.inv(odom_poses[i-1]) @ odom_poses[i]
        
        # Use T_odom_rel as initial guess for ICP
        T_icp_rel = icp(prev_pts, curr_pts, R_init=T_odom_rel[0:3, 0:3], p_init=T_odom_rel[:3,3])

        # Update the global pose
        T_global = T_global @ T_icp_rel

        global_poses.append(T_global.copy())

        # The "previous" scan becomes the current one
        prev_pts = curr_pts

    return global_poses

if __name__ == "__main__":

    # --------- A) Load and sync data ---------
    encoder_counts, encoder_stamps, synced_imu_angular_velocity, synced_imu_stamps = load_and_sync_data(dataset=20)

    # --------- B) Compute odometry (x,y,theta) ---------
    x, y, theta = compute_odometry(encoder_counts, encoder_stamps, synced_imu_angular_velocity)

    # Convert (x,y,theta) to a list of 4x4 transformations
    odom_poses = []
    for i in range(len(x)):
        T_i = pose2d_to_SE3(x[i], y[i], theta[i])
        odom_poses.append(T_i)

    # --------- C) Load LiDAR scans (example placeholders) ---------
    #   Suppose we have an array "lidar_scans" of shape [T, 1081] or similar.
    #   And sensor parameters:
    angle_min = -135.0 * np.pi/180.0
    angle_increment = 0.25 * np.pi/180.0

    # Just as an example, load a .npz with your own Hokuyo data:
    hokuyo_data = np.load(f"../data/Hokuyo20.npz")
    lidar_ranges = hokuyo_data["ranges"]       # shape (1081, T)
    # The data might be arranged differently, so adapt as needed.
    # We assume each column is one scan or vice versa.

    # Re-arrange to have a list of scans
    lidar_scans = []
    for i in range(lidar_ranges.shape[1]):
        scan_i = lidar_ranges[:, i]
        # Possibly filter out invalid ranges
        # ...
        lidar_scans.append(scan_i)

    # --------- D) Run scan matching with odometry as initial guess ---------
    global_poses_icp = run_scan_matching(odom_poses, lidar_scans, angle_min, angle_increment)

    # --------- E) Plot final ICP-refined trajectory ---------
    xs_icp = []
    ys_icp = []
    for T in global_poses_icp:
        xs_icp.append(T[0, 3])
        ys_icp.append(T[1, 3])

    plt.figure(figsize=(8,6))
    # Odometry-only
    plt.plot(x, y, 'r-', label="Odometry-Only Traj")
    # ICP-refined
    plt.plot(xs_icp, ys_icp, 'b-', label="ICP-Refined Traj")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Odometry + ICP Scan Matching")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()