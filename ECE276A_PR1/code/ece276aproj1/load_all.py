import os
import numpy as np
from load_data import read_data

def load_all_datasets(imu_dir, vicon_dir=None, cam_dir=None, test_pan=False, test_quat=False):
    """
    Load and organize all datasets (IMU, VICON, and Camera) using the provided read_data function.

    Parameters:
        imu_dir (str): Directory containing IMU data files.
        vicon_dir (str): Directory containing VICON data files.
        cam_dir (str): Directory containing Camera data files (datasets 1, 2, 8, and 9 only).

    Returns:
        dict: Organized dataset containing IMU, VICON, and CAM data.
    """
    datasets = {}
    if not test_pan and not test_quat:


        for i in range(1, 10):  # Loop through datasets 1 to 9
            imu_file = os.path.join(imu_dir, f"imuRaw{i}.p")
            vicon_file = os.path.join(vicon_dir, f"viconRot{i}.p")
            cam_file = os.path.join(cam_dir, f"cam{i}.p") if i in [1, 2, 8, 9] else None  # Only these have CAM data

            # Load data using read_data
            imu_data = read_data(imu_file)  # Numpy array, shape (7, number of samples)
            vicon_data = read_data(vicon_file)
            cam_data = read_data(cam_file) if cam_file and os.path.exists(cam_file) else None

            # Organize IMU data
            imu_data_dict = {
                "timestamps": imu_data[0, :],  # First row contains timestamps
                "accel": imu_data[1:4, :],     # Rows 1-3 contain accelerometer data
                "gyro": imu_data[4:7, :]       # Rows 4-6 contain gyroscope data
            }

            datasets[f"dataset_{i}"] = {
                "imu": imu_data_dict,
                "vicon": vicon_data,
                "cam": cam_data
            }
    
    elif test_pan:
        for i in range(10, 12):  # Loop through datasets 10 to 11

            imu_file = os.path.join(imu_dir, f"imuRaw{i}.p")
            cam_file = os.path.join(cam_dir, f"cam{i}.p") 

            # Load data using read_data
            imu_data = read_data(imu_file)  # Numpy array, shape (7, number of samples)
            cam_data = read_data(cam_file) if cam_file and os.path.exists(cam_file) else None

            # Organize IMU data
            imu_data_dict = {
                "timestamps": imu_data[0, :],  # First row contains timestamps
                "accel": imu_data[1:4, :],     # Rows 1-3 contain accelerometer data
                "gyro": imu_data[4:7, :]       # Rows 4-6 contain gyroscope data
            }

            datasets[f"dataset_{i}"] = {
                "imu": imu_data_dict,
                "cam": cam_data
            }
    elif test_quat:
        for i in range(10, 12):  # Loop through datasets 10 to 11

            imu_file = os.path.join(imu_dir, f"imuRaw{i}.p")

            # Load data using read_data
            imu_data = read_data(imu_file)  # Numpy array, shape (7, number of samples)

            # Organize IMU data
            imu_data_dict = {
                "timestamps": imu_data[0, :],  # First row contains timestamps
                "accel": imu_data[1:4, :],     # Rows 1-3 contain accelerometer data
                "gyro": imu_data[4:7, :]       # Rows 4-6 contain gyroscope data
            }

            datasets[f"dataset_{i}"] = {
                "imu": imu_data_dict,
            }

    return datasets
        

def extract_imu_raw(datasets, test=False):
    """
    Extract IMU raw data for calibration.

    Parameters:
        datasets (dict): Organized datasets.

    Returns:
        list: List of IMU raw data arrays (6xN) for calibration.
    """
    # print(datasets["dataset_1"]["imu"].keys())
    imu_raw_datasets = []
    if not test:
        for i in range(1, 10):
            imu_data = datasets[f"dataset_{i}"]["imu"]
            # Concatenate accelerometer and gyroscope data to create a 6xN array
            imu_raw = np.vstack((imu_data["timestamps"],imu_data["accel"], imu_data["gyro"]))
            imu_raw_datasets.append(imu_raw)
    else:
        for i in range(10, 12):
            imu_data = datasets[f"dataset_{i}"]["imu"]
            # Concatenate accelerometer and gyroscope data to create a 6xN array
            imu_raw = np.vstack((imu_data["timestamps"],imu_data["accel"], imu_data["gyro"]))
            imu_raw_datasets.append(imu_raw)

    return imu_raw_datasets

