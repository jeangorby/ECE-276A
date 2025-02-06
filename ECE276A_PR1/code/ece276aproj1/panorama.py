import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
from load_all import load_all_datasets, extract_imu_raw
from quaternion import compute_predicted_quaternions, optimize_quaternion_trajectory
from calibrate_imu import calibrate_datasets
from main import synchronize_imu_vicon
import transforms3d.euler as euler

def display_image_on_sphere(img, rotation_matrix):
    """
    Display an image in spherical coordinates using a given rotation matrix.

    Parameters:
        img (numpy.ndarray): The image to display (H x W x 3).
        rotation_matrix (numpy.ndarray): The rotation matrix to transform image points.

    Returns:
        None
    """
    img_height, img_width, _ = img.shape

    x_grid, y_grid = np.meshgrid(np.arange(img_width), np.arange(img_height))

    lon = (x_grid - img_width / 2) * (60 / img_width)
    lat = (y_grid - img_height / 2) * (45 / img_height)

    cartesian = np.stack([
        np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
        np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
        np.sin(np.radians(lat))
    ], axis=-1)

    world_cartesian = np.einsum('ij,hwj->hwi', rotation_matrix, cartesian)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = world_cartesian[..., 0].ravel(), world_cartesian[..., 1].ravel(), world_cartesian[..., 2].ravel()

    colors = img.reshape(-1, 3) / 255.0
    ax.scatter(x, y, z, c=colors, marker='o', s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Image Mapped to Spherical Coordinates")
    plt.show()


def generate_panorama(datasets, dataset_id, use_vicon=True):
    dataset = datasets[f"dataset_{dataset_id}"]
    cam_data = dataset["cam"]
    imu_data = dataset["imu"]
    vicon_data = dataset.get("vicon")
    
    if cam_data is None:
        print(f"No camera data available for dataset {dataset_id}.")
        return

    cam_images = cam_data["cam"]
    cam_timestamps = cam_data["ts"][0]
    
    if use_vicon:
        orientation_data = vicon_data["rots"]
        orientation_timestamps = vicon_data["ts"][0]
    else:
        gyro_data = imu_data["gyro_calibrated"]
        accel_data = imu_data["accel_calibrated"]
        imu_timestamps = imu_data["ts"]
        dt = np.diff(imu_timestamps)
        
        orientation_data = optimize_quaternion_trajectory(
            np.array(gyro_data.T), np.array(accel_data.T), np.array(dt),
            num_iterations=100, lr=0.01
        )
        orientation_timestamps = imu_timestamps
        orientation_data = np.array([euler.quat2mat(q) for q in orientation_data])

    pan_height = 1000
    pan_width = 2000
    panorama = np.zeros((pan_height, pan_width, 3), dtype=np.uint8)

    for i in range(cam_images.shape[-1]):
        print("Stitching image {}".format(i+1))
        img = cam_images[:, :, :, i]
        timestamp = cam_timestamps[i]
        closest_idx = np.argmin(np.abs(orientation_timestamps - timestamp))
        rotation_matrix = orientation_data[:,:,closest_idx].T if use_vicon else orientation_data[closest_idx].T

        # display_image_on_sphere(img, rotation_matrix)

        img_height, img_width, _ = img.shape
        x_grid, y_grid = np.meshgrid(np.arange(img_width), np.arange(img_height))
        
        lon = (x_grid - img_width / 2) * (60 / img_width)
        lat = (y_grid - img_height / 2) * (45 / img_height)

        cartesian = np.stack([
            np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
            np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
            np.sin(np.radians(lat))
        ], axis=-1)

        world_cartesian = np.einsum('ij,hwj->hwi', rotation_matrix, cartesian)

        norm = np.linalg.norm(world_cartesian, axis=-1, keepdims=True)
        world_cartesian /= norm  # Normalize points to unit sphere

        lon_new = np.degrees(np.arctan2(world_cartesian[..., 1], world_cartesian[..., 0]))
        lat_new = np.degrees(np.arcsin(world_cartesian[..., 2]))

        px = ((lon_new + 180) / 360 * pan_width).astype(int)
        py = ((lat_new + 90) / 180 * pan_height).astype(int)

        valid_mask = (px >= 0) & (px < pan_width) & (py >= 0) & (py < pan_height)
        panorama[py[valid_mask], px[valid_mask]] = img[y_grid[valid_mask], x_grid[valid_mask], :3]

    plt.imshow(panorama)
    plt.title(f"Panorama for Dataset {dataset_id}")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    save_path = os.path.join(results_dir, f'panorama_{dataset_id}.jpg')
    plt.savefig(save_path, format='jpg', dpi=300)

    plt.show()


# Load dataset
datasets = load_all_datasets("../data/imu", "../data/vicon", "../data/cam")

test_datasets = load_all_datasets("../testset/imu", cam_dir = "../testset/cam", test_pan=True)

# Process dataset
imu_raw_datasets = extract_imu_raw(datasets)
vicon_datasets = [datasets[f"dataset_{i}"]["vicon"] for i in range(1, 10)]
imu_raw_datasets, vicon_datasets = synchronize_imu_vicon(imu_raw_datasets, vicon_datasets)
calibration_results, calibrated_datasets = calibrate_datasets(imu_raw_datasets, vicon_datasets)

# Extract calibration values from training set
accel_bias = calibration_results["accel_bias"]
gyro_bias = calibration_results["gyro_bias"]
accel_scale = calibration_results["accel_scale"]
gyro_scale = calibration_results["gyro_scale"]

calibrated_testset = {}
for i, test_data in enumerate(test_datasets, 10):
    
    print(f"Processing test dataset {i}...")
    cam_dict = test_datasets[test_data]['cam']
    test_data = test_datasets[test_data]['imu']

    # Extract raw IMU data
    
    imu_timestamps = test_data['timestamps']  # First row: timestamps
    accel_raw = test_data['accel']    # Rows 1-3: accelerometer
    gyro_raw = test_data['gyro']     # Rows 4-6: gyroscope

    # Calibrate IMU data using training set calibration values
    accel_calibrated = (accel_raw - accel_bias[:, None]) * accel_scale + np.array([0,0,1]).reshape(3,1)
    gyro_calibrated = (gyro_raw - gyro_bias[:, None]) * gyro_scale

    imu_data_dict = {
        "ts": imu_timestamps,
        "accel_calibrated": accel_calibrated,
        "gyro_calibrated": gyro_calibrated
    }

    # Store calibrated data
    calibrated_testset[f"dataset_{i}"] = {
        "imu": imu_data_dict,
        "cam": cam_dict
    }

# Generate panorama using VICON ground truth

# train
# generate_panorama(datasets, dataset_id=9, use_vicon=True)

# test
generate_panorama(calibrated_testset, dataset_id=11, use_vicon=False)
