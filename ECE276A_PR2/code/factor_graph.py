import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from odometry import compute_odometry, load_and_sync_data
from scan_matching import run_scan_matching, pose2d_to_SE3

def create_pose_graph(refined_poses, loop_closure_method="fixed", loop_closure_interval=10, proximity_threshold=1.0):
    """
    Constructs a factor graph for SLAM using GTSAM with SE(3) poses and loop closures.
    
    Parameters:
        refined_poses (list of np.ndarray): List of 4x4 transformation matrices representing robot poses in SE(3).
        loop_closure_method (str): "fixed" for fixed-interval loop closure, "proximity" for proximity-based loop closure.
        loop_closure_interval (int): Number of steps before introducing a fixed-interval loop closure constraint.
        proximity_threshold (float): Distance threshold for proximity-based loop closure.
    
    Returns:
        gtsam.NonlinearFactorGraph, gtsam.Values: The constructed factor graph and initial estimates.
    """
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
    loop_closure_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
    
    first_pose = gtsam.Pose3(gtsam.Rot3(refined_poses[0][:3, :3]), gtsam.Point3(*refined_poses[0][:3, 3]))
    initial_estimate.insert(0, first_pose)
    graph.add(gtsam.PriorFactorPose3(0, first_pose, prior_noise))
    
    for i in range(len(refined_poses) - 1):
        T_relative = np.linalg.inv(refined_poses[i]) @ refined_poses[i+1]
        rel_pose = gtsam.Pose3(gtsam.Rot3(T_relative[:3, :3]), gtsam.Point3(*T_relative[:3, 3]))
        
        graph.add(gtsam.BetweenFactorPose3(i, i+1, rel_pose, odometry_noise))
        next_pose = gtsam.Pose3(gtsam.Rot3(refined_poses[i+1][:3, :3]), gtsam.Point3(*refined_poses[i+1][:3, 3]))
        initial_estimate.insert(i+1, next_pose)
        
        if loop_closure_method == "fixed":
            if i >= loop_closure_interval and i % loop_closure_interval == 0:
                loop_T_relative = np.linalg.inv(refined_poses[i - loop_closure_interval]) @ refined_poses[i+1]
                loop_pose = gtsam.Pose3(gtsam.Rot3(loop_T_relative[:3, :3]), gtsam.Point3(*loop_T_relative[:3, 3]))
                graph.add(gtsam.BetweenFactorPose3(i - loop_closure_interval, i+1, loop_pose, loop_closure_noise))
        
        elif loop_closure_method == "proximity":
            for j in range(i):
                distance = np.linalg.norm(refined_poses[i][:3, 3] - refined_poses[j][:3, 3])
                if distance < proximity_threshold:
                    loop_T_relative = np.linalg.inv(refined_poses[j]) @ refined_poses[i]
                    loop_pose = gtsam.Pose3(gtsam.Rot3(loop_T_relative[:3, :3]), gtsam.Point3(*loop_T_relative[:3, 3]))
                    graph.add(gtsam.BetweenFactorPose3(j, i, loop_pose, loop_closure_noise))

    return graph, initial_estimate

def optimize_factor_graph(graph, initial_estimate):
    """Optimizes the factor graph using GTSAM's Levenberg-Marquardt optimizer."""
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()
    return result

def visualize_trajectory_2d(optimized_estimate):
    """Visualizes the optimized trajectory in 2D, similar to scan_matching.py."""
    
    x_vals, y_vals = [], []
    for i in range(optimized_estimate.size()):
        pose = optimized_estimate.atPose3(i)
        translation = pose.translation()
        x_vals.append(translation[0])
        y_vals.append(translation[1])
    

    plt.figure(figsize=(8,6))
    plt.plot(x_vals, y_vals, 'b-', label="GTSAM-Refined Traj")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Odometry + GTSAM Scan Matching")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

def visualize_all_trajectories(odom_x, odom_y, refined_poses, optimized_estimate):
    """Visualizes all three trajectories: odometry-only, ICP-enhanced, and GTSAM-enhanced."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Odometry-only trajectory
    ax.plot(odom_x, odom_y, 'r-', label="Odometry Only")
    
    # ICP-enhanced trajectory
    icp_x, icp_y = zip(*[(pose[0, 3], pose[1, 3]) for pose in refined_poses])
    ax.plot(icp_x, icp_y, 'g-', label="ICP Enhanced")
    
    # GTSAM-enhanced trajectory
    gtsam_x, gtsam_y = [], []
    for i in range(optimized_estimate.size()):
        pose = optimized_estimate.atPose3(i)
        translation = pose.translation()
        gtsam_x.append(translation[0])
        gtsam_y.append(translation[1])
    ax.plot(gtsam_x, gtsam_y, 'b-', label="GTSAM Optimized")
    
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Comparison of Trajectories")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    plt.show()

def visualize_graph(graph, initial_estimate):
    """Visualizes the factor graph with nodes and edges using GTSAM utilities in 3D."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract node positions
    node_positions = []
    for i in range(initial_estimate.size()):
        pose = initial_estimate.atPose3(i)
        node_positions.append(pose.translation())
        gtsam_plot.plot_pose3_on_axes(ax, pose, scale=0.3)
    
    node_positions = np.array(node_positions)
    
    # Plot edges between nodes
    for j in range(graph.size()):
        factor = graph.at(j)
        if isinstance(factor, gtsam.BetweenFactorPose3):
            key1, key2 = factor.keys()
            pos1 = initial_estimate.atPose3(key1).translation()
            pos2 = initial_estimate.atPose3(key2).translation()
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 'k-', linewidth=1)

    
    plt.title("Pose Graph Visualization (SE3) with Nodes and Edges")
    plt.show()


if __name__ == "__main__":
    dataset = 20
    
    # Load and sync odometry data
    encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps = load_and_sync_data(dataset)
    x, y, theta = compute_odometry(encoder_counts, encoder_stamps, imu_angular_velocity)
    
    # Load and refine trajectory using ICP
    hokuyo_data = np.load(f"../data/Hokuyo{dataset}.npz")
    angle_min = hokuyo_data["angle_min"]
    angle_increment = hokuyo_data["angle_increment"]
    lidar_ranges_all = hokuyo_data["ranges"]
    lidar_scans = [lidar_ranges_all[:, i] for i in range(lidar_ranges_all.shape[1])]
    
    odom_poses = [pose2d_to_SE3(x[i], y[i], theta[i]) for i in range(len(x))]
    refined_poses = run_scan_matching(odom_poses, lidar_scans, angle_min, angle_increment)
    
    graph, initial_estimate = create_pose_graph(refined_poses, loop_closure_method="proximity")
    # graph, initial_estimate = create_pose_graph(odom_poses)

    optimized_estimate = optimize_factor_graph(graph, initial_estimate)

    # visualize_trajectory_2d(optimized_estimate)
    # visualize_graph(graph, initial_estimate)
    visualize_all_trajectories(x, y, refined_poses, optimized_estimate)
