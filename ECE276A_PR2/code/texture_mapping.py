import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from occupancy_grid_mapping import initialize_map, world_to_map, update_occupancy_grid
from odometry import load_and_sync_data, compute_odometry
from scan_matching import pose2d_to_SE3, run_scan_matching
import os

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Compute the rotation matrix from roll, pitch, and yaw (ZYX order)."""
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # Final rotation matrix
    R = R_z @ R_y @ R_x  # Equivalent to Rz(ψ) Ry(θ) Rx(ϕ)
    
    return R

def visualize_point_cloud(points, colors, title="3D Point Cloud with RGB Texture"):
    """ Visualize the projected 3D points using Open3D with RGB image texture. """

    # truth
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.T)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255)  # open3d expects color channels 0-1, opencv uses uint8 0-255

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.5)  # visualize the camera regular frame for reference.

    o3d.visualization.draw_geometries([pcd, origin])  # display the pointcloud and origin


def project_depth_to_3d(disp_img, rgb_img, bgr_img):
    height, width = disp_img.shape

    dd = np.array(-0.00304 * disp_img + 3.31)
    depth = 1.03 / dd

    mesh = np.meshgrid(np.arange(0, height), np.arange(0, width), indexing='ij')  
    i_idxs = mesh[0].flatten()
    j_idxs = mesh[1].flatten()

    rgb_i = np.array((526.37 * i_idxs + 19276 - 7877.07 * dd.flatten()) / 585.051, dtype=np.int32)  # force int for indexing
    rgb_j = np.array((526.37 * j_idxs + 16662) / 585.051, dtype=np.int32)

    # some may be out of bounds, just clip them
    rgb_i = np.clip(rgb_i, 0, height - 1)
    rgb_j = np.clip(rgb_j, 0, width - 1)

    colors = rgb_img[rgb_i, rgb_j]

    # lets visualize the image using our transformation to make sure things look correct (using bgr for opencv)
    bgr_colors = bgr_img[rgb_i, rgb_j]
    cv2.imshow("color", bgr_colors.reshape((height, width, 3)))

    uv1 = np.vstack([j_idxs, i_idxs, np.ones_like(i_idxs)])
    K = np.array([[585.05, 0, 242.94],
                [0, 585.05, 315.84],
                [0, 0, 1]])

    # project images to 3d points
    points = depth.flatten() * (np.linalg.inv(K) @ uv1)

    oRr = np.array([[0, -1, 0],
                    [0, 0, -1],
                    [1, 0, 0]])
    # we want rRo because we have points in optical frame and want to move them to the regular frame.
    points = oRr.T @ points
    # points = oRr @ points

    return points, colors

def update_texture_map(MAP, filtered_points, filtered_colors):
    """
    Project transformed 3D points to the occupancy grid and assign RGB values.
    """
      
    map_coords = world_to_map(MAP, filtered_points[:2, :])
    
    valid_mask = (0 <= map_coords[0]) & (map_coords[0] < MAP['size'][0]) & \
                 (0 <= map_coords[1]) & (map_coords[1] < MAP['size'][1])
    
    map_coords = map_coords[:, valid_mask]
    valid_colors = filtered_colors[valid_mask]
    
    MAP['texture'][map_coords[0], map_coords[1], :] = valid_colors
    return MAP


def texture_mapping(dataset, refined_poses, slam_timestamps):
    """
    Perform texture mapping using Kinect RGB-D data and the refined SLAM trajectory.
    """
    MAP = initialize_map()
    MAP['texture'] = np.zeros((*MAP['size'], 3), dtype=np.uint8)  
    
    from scipy.spatial.transform import Rotation as R
    # Given camera extrinsics: pitch = 0.36 rad, yaw = 0.021 rad
    # (0.18, 0.005, 0.36) m
    R_cam = euler_to_rotation_matrix(0, 0.36, 0.021)

    T_cam_robot = np.array([
        [1, 0, 0, 0.18],  # X offset
        [0, 1, 0, 0.005],        # Y offset
        [0, 0, 1, 0.36],  # Z offset (height above ground)
        [0, 0, 0, 1]
    ])

    # Add rotation to the transformation matrix
    T_cam_robot[:3, :3] = R_cam

    kinect_data = np.load(f"../data/Kinect{dataset}.npz")
    disparity_stamps = kinect_data['disparity_time_stamps']
    rgb_stamps = kinect_data['rgb_time_stamps']
    
    for i, rgb_time in enumerate(rgb_stamps, 1):
        # i = 0  # Focus on the first image
        rgb_time = rgb_stamps[i-1]
        print(f"Processing Frame {i}")
        
        closest_idx = np.argmin(np.abs(slam_timestamps - rgb_time))
        closest_pose = refined_poses[closest_idx]

        disparity_image = cv2.imread(f"../data/dataRGBD/Disparity{dataset}/disparity{dataset}_{i}.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
        bgr_img = cv2.imread(f"../data/dataRGBD/RGB{dataset}/rgb{dataset}_{i}.png")
        rgb_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        points, colors = project_depth_to_3d(disparity_image, rgb_image, bgr_img)

        # visualize_point_cloud(points, colors, 'optical to cam')    
        
        pts_robot = (T_cam_robot[:3,:3] @ points) + T_cam_robot[:3, 3][:, np.newaxis]

        pts_world = (closest_pose[:3,:3] @ pts_robot) + closest_pose[:3,3][:, np.newaxis]
        # visualize_point_cloud(pts_world, colors, 'robot to world')

        floor_height_threshold = 0.2  # Adjust as needed
        # floor_height_threshold = np.max(pts_world[2, :])
        valid_mask = np.abs(pts_world[2, :]) < floor_height_threshold

        filtered_points = pts_world[:, valid_mask]
        filtered_colors = colors[valid_mask]

        # visualize_point_cloud(filtered_points, filtered_colors, 'filtered out floor')
        # exit()
        MAP = update_texture_map(MAP, filtered_points, filtered_colors)
        # visualize_texture_map(MAP)
        # exit()

    return MAP

def visualize_texture_map(MAP):
    """
    Display the texture map overlay.
    """
    plt.imshow(np.rot90(MAP['texture']))
    plt.title("Texture Mapped Floor")
    plt.show()

if __name__ == "__main__":
    dataset = 21
    
    encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps = load_and_sync_data(dataset)
    x, y, theta = compute_odometry(encoder_counts, encoder_stamps, imu_angular_velocity)
    odom_poses = [pose2d_to_SE3(x[i], y[i], theta[i]) for i in range(len(x))]
    
    hokuyo_data = np.load(f"../data/Hokuyo{dataset}.npz")
    angle_min = hokuyo_data["angle_min"]
    angle_increment = hokuyo_data["angle_increment"]
    lidar_ranges_all = hokuyo_data["ranges"]
    lidar_scans = [lidar_ranges_all[:, i] for i in range(lidar_ranges_all.shape[1])]
    
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


    
    MAP_texture_icp = texture_mapping(dataset, refined_poses, encoder_stamps)
    MAP_texture_odom = texture_mapping(dataset, odom_poses, encoder_stamps)
    # visualize_texture_map(MAP_texture)
    # MAP_texture = refine_texture_map(MAP_texture)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Occupancy Grid
    axs[0].imshow(np.rot90(MAP_texture_odom['texture']))
    axs[0].set_title("Odometry Texture Map")
    axs[0].axis("off")

    # Texture Map
    axs[1].imshow(np.rot90(MAP_texture_icp['texture']))  # Flip to match orientation
    axs[1].set_title("ICP Enhanced Texture Map")
    axs[1].axis("off")

    plt.savefig('./results/texture_mapping_21.png') 
    plt.show()


