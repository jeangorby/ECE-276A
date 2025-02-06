import numpy as np
from scipy.spatial.transform import Rotation as R

def recalibrate_imu_static_vicon(imu_raw_datasets, vicon_datasets, static_duration=5.0):
    """
    Recalibrate IMU data using the assumption that the first `static_duration` seconds
    of each dataset correspond to a static period in the VICON rotation.

    Parameters:
        imu_raw_datasets (list): List of raw IMU data arrays (6xN per dataset).
        vicon_datasets (list): List of VICON data (containing 'rots' and 'ts').
        static_duration (float): Duration (in seconds) of the initial static period.

    Returns:
        dict: Calibration results (biases and scale factors).
    """
    accel_biases = []
    gyro_biases = []

    for dataset_idx, (imu_raw, vicon_data) in enumerate(zip(imu_raw_datasets, vicon_datasets)):
        # Extract IMU timestamps
        imu_timestamps = imu_raw[0, :]  # First row: timestamps

        # Extract VICON timestamps and rotations
        vicon_rot = vicon_data["rots"].transpose(2, 0, 1)  # Shape (N, 3, 3)
        vicon_timestamps = vicon_data["ts"]

        # Find indices corresponding to the static period
        static_end_time = vicon_timestamps[0] + static_duration

        static_mask = (vicon_timestamps <= static_end_time)
        static_indices = np.where(static_mask)[0]

        if len(static_indices) == 0:
            raise ValueError(f"No valid static period detected in dataset {dataset_idx + 1}.")

        # Map VICON static indices to IMU timestamps
        static_start_time = vicon_timestamps[static_indices[0]]
        static_end_time = vicon_timestamps[static_indices[-1]]

        imu_static_mask = (imu_timestamps >= static_start_time) & (imu_timestamps <= static_end_time)
        imu_static_indices = np.where(imu_static_mask)[0]

        if len(imu_static_indices) == 0:
            raise ValueError(f"No IMU data points found in the static period for dataset {dataset_idx + 1}.")

        # Extract accelerometer and gyroscope data
        accel_raw = imu_raw[1:4, :]  # Rows 1-3: Accelerometer
        gyro_raw = imu_raw[4:7, :]  # Rows 4-6: Gyroscope

        # Compute biases during the static period
        accel_bias = np.mean(accel_raw[:, imu_static_indices], axis=1)
        gyro_bias = np.mean(gyro_raw[:, imu_static_indices], axis=1)
        
        accel_biases.append(accel_bias)
        gyro_biases.append(gyro_bias)


    # Aggregate biases across datasets
    accel_bias = np.mean(accel_biases, axis=0)
    gyro_bias = np.mean(gyro_biases, axis=0)

    # Scale factors from the datasheet (example values, update as needed)
    accel_sensitivity = 300e-3  # mV/g
    gyro_sensitivity = 3.33e-3  # mV/deg/s
    v_ref = 3.3  # Reference voltage in V
    adc_resolution = 1023  # 10-bit ADC

    accel_scale = v_ref / (adc_resolution * accel_sensitivity)
    gyro_scale = v_ref / (adc_resolution * gyro_sensitivity) * (np.pi / 180)  # Convert to rad/s

    return {
        "accel_bias": accel_bias,
        "gyro_bias": gyro_bias,
        "accel_scale": accel_scale,
        "gyro_scale": gyro_scale,
    }


def calibrate_datasets(imu_raw_datasets, vicon_datasets=None):
    """
    Perform calibration for multiple datasets and return calibrated data.
    
    If VICON data is provided, use it to detect the static period for bias estimation.

    Parameters:
        imu_raw_datasets (list of numpy.ndarray): List of raw IMU data arrays (7xN arrays per dataset).
        vicon_datasets (list of dict, optional): List of VICON data dictionaries (containing 'rots' and 'ts').

    Returns:
        tuple: Calibration results and list of calibrated datasets.
    """
    if vicon_datasets is not None:
        # Use VICON-based calibration
        # results = calibrate_imu_using_vicon(imu_raw_datasets, vicon_datasets)

        results = recalibrate_imu_static_vicon(imu_raw_datasets, vicon_datasets)

    calibrated_datasets = []
    for imu_raw in imu_raw_datasets:
        # Extract raw accelerometer and gyroscope data
        accel_raw = imu_raw[1:4, :]  # Rows 1-3: Accelerometer
        gyro_raw = imu_raw[4:7, :]  # Rows 4-6: Gyroscope
        ts = imu_raw[0,:]
        
        # Apply calibration biases and scales
        accel_calibrated = (accel_raw - results["accel_bias"][:, None]) * results["accel_scale"] + np.array([0,0,1]).reshape(3,1)
        gyro_calibrated = (gyro_raw - results["gyro_bias"][:, None]) * results["gyro_scale"]

        # Store calibrated data
        calibrated_datasets.append({
            "ts": ts,
            "accel_calibrated": accel_calibrated,
            "gyro_calibrated": gyro_calibrated
        })

    return results, calibrated_datasets
