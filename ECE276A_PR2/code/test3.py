import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from occupancy_grid_mapping import initialize_map, world_to_map
from odometry import load_and_sync_data, compute_odometry
from scan_matching import pose2d_to_SE3, run_scan_matching
import os

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

    return points, colors

def visualize_point_cloud(points, colors, title="3D Point Cloud with RGB Texture"):
    """ Visualize the projected 3D points using Open3D with RGB image texture. """
    # points = points[:3, :].T  # Ensure correct shape (N, 3)
    
    # colors = colors.reshape(-1, 3).astype(np.float32) / 255.0  # Ensure correct shape (N, 3) and scale
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    # o3d.visualization.draw_geometries([pcd, origin], window_name=title)

    # truth
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.T)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255)  # open3d expects color channels 0-1, opencv uses uint8 0-255

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.5)  # visualize the camera regular frame for reference.

    o3d.visualization.draw_geometries([pcd, origin])  # display the pointcloud and origin

def texture_mapping_debug(dataset, refined_poses, slam_timestamps):
    MAP = initialize_map()
    MAP['texture'] = np.zeros((*MAP['size'], 3), dtype=np.uint8)
    
    T_cam_robot = np.array([[1, 0, 0, 0.33276], [0, 1, 0, 0], [0, 0, 1, 0.38011], [0, 0, 0, 1]])
    
    kinect_data = np.load(f"../data/Kinect{dataset}.npz")
    disparity_stamps = kinect_data['disparity_time_stamps']
    rgb_stamps = kinect_data['rgb_time_stamps']
    
    i = 0  # Focus on the first image
    rgb_time = rgb_stamps[i]
    print(f"Processing first frame at timestamp {rgb_time}")
    
    disparity_image = cv2.imread(f"../data/dataRGBD/Disparity{dataset}/disparity{dataset}_1.png", cv2.IMREAD_UNCHANGED).astype(np.float32)
    bgr_img = cv2.imread(f"../data/dataRGBD/RGB{dataset}/rgb{dataset}_1.png")
    rgb_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    points, colors = project_depth_to_3d(disparity_image, rgb_image, bgr_img)

    # visualize_point_cloud(points, colors, 'optical to cam')
    
    closest_idx = np.argmin(np.abs(slam_timestamps - rgb_time))
    closest_pose = refined_poses[closest_idx]
    
    pts_robot = (T_cam_robot[:3,:3] @ points) + T_cam_robot[:3, 3][:, np.newaxis]
    # visualize_point_cloud(pts_robot, colors, "cam to robot")

    pts_world = (closest_pose[:3,:3] @ pts_robot) + closest_pose[:3,3][:, np.newaxis]
    # visualize_point_cloud(pts_world, colors, 'robot to world')

    floor_height_threshold = 0.5  # Adjust as needed
    valid_mask = pts_world[2, :] < floor_height_threshold

    filtered_points = pts_world[:, valid_mask]
    filtered_colors = colors[valid_mask]

    visualize_point_cloud(filtered_points, filtered_colors, 'filtered out floor')
    
    return MAP



if __name__ == "__main__":
    dataset = 20
    
    encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps = load_and_sync_data(dataset)
    x, y, theta = compute_odometry(encoder_counts, encoder_stamps, imu_angular_velocity)
    odom_poses = [pose2d_to_SE3(x[i], y[i], theta[i]) for i in range(len(x))]
    
    hokuyo_data = np.load(f"../data/Hokuyo{dataset}.npz")
    angle_min = hokuyo_data["angle_min"]
    angle_increment = hokuyo_data["angle_increment"]
    lidar_ranges_all = hokuyo_data["ranges"]
    lidar_scans = [lidar_ranges_all[:, i] for i in range(lidar_ranges_all.shape[1])]
    
    # refined_poses = run_scan_matching(odom_poses, lidar_scans, angle_min, angle_increment)
    MAP_texture = texture_mapping_debug(dataset, odom_poses, encoder_stamps)
