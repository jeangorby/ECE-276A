import numpy as np
import matplotlib.pyplot as plt
import cv2
from odometry import load_and_sync_data, compute_odometry
from scan_matching import pose2d_to_SE3, run_scan_matching
from pr2_utils import bresenham2D, plot_map


def inverse_pose(T):
    """
    Computes the inverse of a 4x4 pose matrix using the formula:
    
        [ R^T   -R^T * p ]
        [  0^T       1   ]

    Parameters:
    T (numpy.ndarray): 4x4 transformation matrix

    Returns:
    numpy.ndarray: 4x4 inverse transformation matrix
    """
    R = T[:3, :3]  # Extract rotation (3x3)
    p = T[:3, 3]   # Extract translation (3x1)

    R_T = R.T  # Transpose of rotation matrix
    p_inv = -R_T @ p  # Compute -R^T * p

    # Construct inverse transformation matrix
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_T
    T_inv[:3, 3] = p_inv

    return T_inv


def initialize_map(res=0.05):
    """Initialize an occupancy grid map centered at the robot's start position."""
    MAP = {}
    # Set resolution
    MAP['res'] = np.array([res, res])  # Grid cell resolution in meters
    MAP['min'] = np.array([-20.0, -20.0])  # meters
    MAP['max'] = np.array([20.0, 20.0])    # meters
    # Compute grid size
    MAP['size'] = np.ceil((MAP['max'] - MAP['min']) / MAP['res']).astype(int)

    # Ensure grid size is odd so that the origin is centered
    is_even = MAP['size'] % 2 == 0
    MAP['size'][is_even] += 1

    # Initialize log-odds and occupancy grid
    MAP['log_odds'] = np.zeros(MAP['size'])
    MAP['map'] = np.zeros(MAP['size'], dtype=np.int8)

    return MAP

def world_to_map(MAP, point):
    """Convert world coordinates to map grid indices."""
    return np.floor((point - MAP['min'].reshape(-1, 1)/1.75) / MAP['res'].reshape(-1, 1)).astype(int)
    # return np.floor(point / MAP['res'].reshape(-1, 1)).astype(int)

def lidar_to_point_cloud(ranges, angle_min, angle_increment, max_range=30, min_range=0.1):
    """Convert LiDAR scan data into 2D point cloud."""
    ranges = np.squeeze(ranges)
    angles = angle_min + np.arange(len(ranges)) * angle_increment[0]

    valid = np.logical_and(ranges > min_range, ranges < max_range)

    x = ranges[valid] * np.cos(angles[valid])
    y = ranges[valid] * np.sin(angles[valid])

    return np.vstack((x, y))

def transform_to_world(odom_pose, lidar_scan, angle_min, angle_increment):
    """Transform LiDAR points from sensor frame to world frame using odometry."""
    lidar_pc = lidar_to_point_cloud(lidar_scan, angle_min, angle_increment)

    # Convert to homogeneous coordinates
    lidar_pc_h = np.vstack((lidar_pc, np.zeros((1, lidar_pc.shape[1])), np.ones((1, lidar_pc.shape[1]))))

    # LiDAR to Robot transformation
    T_lidar = np.array([
        [1, 0, 0, -0.29833],  # LiDAR offset in X
        [0, 1, 0, 0.0],  # No Y offset
        [0, 0, 1, 0.51435], # LiDAR height above ground
        [0, 0, 0, 1]
    ])

    world_pc = odom_pose @ T_lidar @ lidar_pc_h
    return world_pc[:2, :]


def update_occupancy_grid(MAP, robot_pose, lidar_scan, angle_min, angle_increment, lo_occ=0.85, lo_free=-0.4, lo_min=-5, lo_max=5):
    """Update the occupancy grid using LiDAR scan data with probabilistic log-odds updates."""
    
    # Convert LiDAR scan to world coordinates
    world_pts = transform_to_world(robot_pose, lidar_scan, angle_min, angle_increment)
    robot_pos = robot_pose[:2, 3]

    # Convert robot and LiDAR points to map coordinates
    robot_cell = world_to_map(MAP, robot_pos).flatten()  
    # exit()
    cells = world_to_map(MAP, world_pts)

    # Validate that the LiDAR points are within map bounds
    valid_cells = np.all((cells >= 0) & (cells < MAP['size'].reshape(-1, 1)), axis=0)
    cells = cells[:, valid_cells]

    # **1. Mark Occupied Cells Using Log-Odds Update**
    MAP['log_odds'][cells[0], cells[1]] += lo_occ

    # **2. Compute Free Cells Using Bresenham’s Algorithm**
    free_space_points = []  

    for i in range(cells.shape[1]):
        end_cell = cells[:, i].flatten()
        free_cells = bresenham2D(robot_cell[0], robot_cell[1], end_cell[0], end_cell[1]).astype(int)
        free_space_points.append(free_cells[:, :-1])  # Exclude the last point (occupied cell)
        # print(robot_cell[0], robot_cell[1], end_cell[0], end_cell[1])
        # exit()
    # **3. Apply Log-Odds Update to Free Cells**
    try:
        free_space_points = np.hstack(free_space_points)

        # Validate free space points within map bounds
        valid_idx = (0 <= free_space_points[0]) & (free_space_points[0] < MAP['size'][0]) & \
                    (0 <= free_space_points[1]) & (free_space_points[1] < MAP['size'][1])

        MAP['log_odds'][free_space_points[0, valid_idx], free_space_points[1, valid_idx]] += lo_free
    except:
        pass  # No free points found

    # **4. Clamp Log-Odds Values to Prevent Overconfidence**
    MAP['log_odds'] = np.clip(MAP['log_odds'], lo_min, lo_max)

    # **5. Convert Log-Odds Back to Probabilities**
    MAP['map'] = np.exp(MAP['log_odds']) / (1 + np.exp(MAP['log_odds']))  # Sigmoid function

    return MAP


if __name__ == "__main__":
    dataset = 21

    # ----------------- Load Odometry Data ----------------- #
    encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps = load_and_sync_data(dataset)
    x, y, theta = compute_odometry(encoder_counts, encoder_stamps, imu_angular_velocity)
    odom_poses = [pose2d_to_SE3(x[i], y[i], theta[i]) for i in range(len(x))]

    # ----------------- Load LiDAR Data ----------------- #
    hokuyo_data = np.load(f"../data/Hokuyo{dataset}.npz")
    angle_min = hokuyo_data["angle_min"]
    angle_increment = hokuyo_data["angle_increment"]
    lidar_ranges_all = hokuyo_data["ranges"]
    lidar_scans = [lidar_ranges_all[:, i] for i in range(lidar_ranges_all.shape[1])]

    # ----------------- Compute ICP-Enhanced Trajectory ----------------- #
    refined_poses = run_scan_matching(odom_poses, lidar_scans, angle_min, angle_increment)
    # refined_poses = np.load("best_icp_scan_matching2.npy")

    # Ensure matching scan count
    num_frames = min(len(lidar_scans), len(odom_poses), len(refined_poses))
    # num_frames = min(len(lidar_scans), len(odom_poses))
    # ----------------- Build Occupancy Grid Maps ----------------- #
    MAP_odom = initialize_map()
    MAP_icp = initialize_map()
    # plot_map(MAP_odom['map'])
    for i in range(num_frames):
        print('occupancy update: ', i)
        lidar_scan = lidar_scans[i]

        # Update maps
        MAP_odom = update_occupancy_grid(MAP_odom, odom_poses[i], lidar_scan, angle_min, angle_increment)
        MAP_icp = update_occupancy_grid(MAP_icp, refined_poses[i], lidar_scan, angle_min, angle_increment)
    #     break
    # plot_map(MAP_odom['map'])
    # ----------------- Plot Occupancy Grid Maps ----------------- #
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(MAP_odom['map'].T, cmap="gray", origin="lower")
    axs[0].set_title("Occupancy Grid Map (Odometry)")

    axs[1].imshow(MAP_icp['map'].T, cmap="gray", origin="lower")
    axs[1].set_title("Occupancy Grid Map (ICP)")
    plt.savefig("./results/occupancy_grid_map_21.png")
    plt.show()