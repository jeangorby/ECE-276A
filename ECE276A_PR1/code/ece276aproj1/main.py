import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from load_all import load_all_datasets, extract_imu_raw
from calibrate_imu import calibrate_datasets
from quaternion import optimize_quaternion_trajectory, compute_predicted_quaternions, observation_model
from quaternion import check_observation_model_gravity, check_motion_model_cost
from plot import plot_predicted_quaternions_vs_vicon, plot_comparison_trajectories, plot_test_set_results
import transforms3d.quaternions as quat
import transforms3d.euler as euler
import jax.numpy as jnp
import jax

def synchronize_imu_vicon(imu_raw_datasets, vicon_datasets):
    """
    Synchronize IMU and VICON datasets by interpolating VICON timestamps to match IMU timestamps.

    Parameters:
        imu_raw_datasets (list): List of raw IMU data arrays (6xN per dataset).
        vicon_datasets (list): List of VICON data dictionaries (containing 'rots' and 'ts').

    Returns:
        tuple: Synchronized IMU and VICON datasets.
    """

    synchronized_vicon_datasets = []

    for imu_raw, vicon_data in zip(imu_raw_datasets, vicon_datasets):
        # Extract IMU timestamps
        imu_timestamps = imu_raw[0, :]

        # Extract VICON timestamps and rotations
        vicon_rot = vicon_data["rots"].transpose(2, 0, 1)  # Shape (N, 3, 3)
        vicon_timestamps = vicon_data["ts"].transpose().squeeze()
        
        # Interpolate VICON rotation matrices to match IMU timestamps
        interpolated_rots = []
        for i in range(3):  # Interpolate each row of the rotation matrix separately
            for j in range(3):
                interp_func = interp1d(
                    vicon_timestamps, vicon_rot[:, i, j], kind="linear", bounds_error=False, fill_value="extrapolate"
                )
                interpolated_rots.append(interp_func(imu_timestamps))

        interpolated_rots = np.array(interpolated_rots).reshape(3, 3, -1).transpose(2, 0, 1)

        # Save synchronized VICON data
        synchronized_vicon_datasets.append({
            "rots": interpolated_rots,
            "ts": imu_timestamps  # Now synchronized with IMU
        })

    return imu_raw_datasets, synchronized_vicon_datasets


def process_test_set(test_datasets, calibration_results, compute_predicted_quaternions, optimize_quaternion_trajectory, plot_comparison_trajectories):
    """
    Process the test set with raw IMU data, calibrate it using training set calibration values, 
    predict quaternion trajectories, and plot the results.

    Parameters:
        test_datasets (list): List of raw IMU test datasets (6xN arrays).
        calibration_results (dict): Calibration results from the training set.
        compute_predicted_quaternions (function): Function to compute unoptimized quaternion trajectory.
        optimize_quaternion_trajectory (function): Function to optimize quaternion trajectory.
        plot_comparison_trajectories (function): Function to plot results.

    Returns:
        None
    """
    print("Processing test datasets...")

    # Extract calibration values from training set
    accel_bias = calibration_results["accel_bias"]
    gyro_bias = calibration_results["gyro_bias"]
    accel_scale = calibration_results["accel_scale"]
    gyro_scale = calibration_results["gyro_scale"]

    for i, test_data in enumerate(test_datasets, 10):

        print(f"Processing test dataset {i}...")

        # Extract raw IMU data
        
        imu_timestamps = test_data[0, :]  # First row: timestamps
        accel_raw = test_data[1:4, :]    # Rows 1-3: accelerometer
        gyro_raw = test_data[4:7, :]     # Rows 4-6: gyroscope

        # Calibrate IMU data using training set calibration values
        accel_calibrated = (accel_raw - accel_bias[:, None]) * accel_scale + np.array([0,0,1]).reshape(3,1)
        gyro_calibrated = (gyro_raw - gyro_bias[:, None]) * gyro_scale

        # Compute unoptimized quaternion trajectory
        print("Computing unoptimized quaternion trajectory...")
        unoptimized_quaternions = compute_predicted_quaternions(gyro_calibrated, imu_timestamps)

        # Compute time differences
        dt = np.diff(imu_timestamps)

        # Compute optimized quaternion trajectory
        print("Computing optimized quaternion trajectory...")
        optimized_quaternions = optimize_quaternion_trajectory(
            jnp.array(gyro_calibrated.transpose()),
            jnp.array(accel_calibrated.transpose()),
            jnp.array(dt),
            num_iterations=200,
            lr=0.01
        )

        # Convert quaternions to Euler angles
        unoptimized_euler = np.array([euler.quat2euler(q) for q in unoptimized_quaternions])
        optimized_euler = np.array([euler.quat2euler(q) for q in optimized_quaternions])

        # Plot results
        print("Plotting results...")
        plot_test_set_results(imu_timestamps, unoptimized_euler, optimized_euler, dataset_index=i + 1)




def main():
    # Define directories
    imu_dir = "../data/imu"
    vicon_dir = "../data/vicon"
    cam_dir = "../data/cam"

    print("Loading datasets...")
    # Load all datasets
    datasets = load_all_datasets(imu_dir, vicon_dir, cam_dir)

    # Extract IMU raw data for calibration
    print("Extracting IMU raw data...")
    imu_raw_datasets = extract_imu_raw(datasets)

    

    print(f"Loaded {len(imu_raw_datasets)} IMU datasets.")
    for i, imu_raw in enumerate(imu_raw_datasets, 1):
        print(f"Dataset {i}: IMU data shape = {imu_raw.shape}")
        cam_status = "available" if datasets[f"dataset_{i}"]["cam"] is not None else "not available"
        print(f"Dataset {i}: CAM data is {cam_status}.")

    

    # # Extract VICON data
    vicon_datasets = [datasets[f"dataset_{i}"]["vicon"] for i in range(1, 10)]

    # plot_vicon_euler(vicon_datasets[0]['rots'], vicon_datasets[0]['ts'])

    print("Calibrating IMU using VICON data...")
    # Perform calibration using VICON
    imu_raw_datasets, vicon_datasets = synchronize_imu_vicon(imu_raw_datasets, vicon_datasets)
    calibration_results, calibrated_datasets = calibrate_datasets(imu_raw_datasets, vicon_datasets)
 
    '''
    
    Pure motion model section

    '''

    # # Predict quaternion trajectory using motion model
    # for j, calibrated_data in enumerate(calibrated_datasets):
    #     print(f"Processing dataset {j + 1}...")
    #     imu_timestamps = calibrated_data["ts"]
    #     gyro_calibrated = calibrated_data["gyro_calibrated"]

    #     print("Computing quaternion trajectory...")
    #     predicted_quaternions = compute_predicted_quaternions(gyro_calibrated, imu_timestamps)

    #     # Compare with VICON quaternions
    #     vicon_rotations = vicon_datasets[j]["rots"]

    #     vicon_euler = np.array([euler.mat2euler(rots) for rots in vicon_rotations])
  
    #     predicted_quaternions = np.array([euler.quat2euler(q) for q in predicted_quaternions])

    #     print("Plotting results...")
    #     plot_predicted_quaternions_vs_vicon(imu_timestamps, predicted_quaternions, vicon_euler)
    #     exit()

    '''
    
    Pure optimization

    '''

    # # Iterate over datasets for quaternion trajectory and comparison
    # for i, calibrated_data in enumerate(calibrated_datasets):
    #     print(f"Processing Dataset {i+1}...")

    #     # Extract calibrated gyroscope data and timestamps
    #     gyro_calibrated = calibrated_data["gyro_calibrated"]
    #     imu_timestamps = calibrated_data["ts"]
        
    #     # Compute time differences
    #     dt = np.diff(imu_timestamps)

    #     # Compute quaternion trajectory
    #     print("Computing quaternion trajectory...")
        
    #     quaternions = optimize_quaternion_trajectory(
    #         jnp.array(gyro_calibrated.transpose()),
    #         jnp.array(calibrated_data["accel_calibrated"].transpose()),
    #         jnp.array(dt),
    #         num_iterations=1000,
    #         lr=0.01
    #     )

    #     # Convert quaternion trajectory to Euler angles using transforms3d
    #     estimated_euler = np.array([euler.quat2euler(q) for q in quaternions])
        
    #     # Load and interpolate VICON ground truth
    #     vicon_rotations = vicon_datasets[i]["rots"]
   
    #     vicon_euler = np.array([euler.mat2euler(rots) for rots in vicon_rotations])

    #     print(vicon_euler.shape, estimated_euler.shape)

    #     # Plot comparison
    #     plot_predicted_quaternions_vs_vicon(imu_timestamps, estimated_euler, vicon_euler)
    #     exit()


    '''
    
    Optimization and motion model

    '''

    # # Predict quaternion trajectory using motion model
    # for j, calibrated_data in enumerate(calibrated_datasets):
    #     # if j < 3:
    #     #     continue
    #     print(f"Processing dataset {j + 1}...")
    #     imu_timestamps = calibrated_data["ts"]
    #     gyro_calibrated = calibrated_data["gyro_calibrated"]

    #     # Compute unoptimized quaternion trajectory
    #     print("Computing unoptimized quaternion trajectory...")
    #     unoptimized_quaternions = compute_predicted_quaternions(gyro_calibrated, imu_timestamps)

    #     # Convert VICON rotations to Euler angles
    #     vicon_rotations = vicon_datasets[j]["rots"]
    #     vicon_euler = np.array([euler.mat2euler(rots) for rots in vicon_rotations])

    #     # Convert unoptimized quaternions to Euler angles
    #     unoptimized_euler = np.array([euler.quat2euler(q) for q in unoptimized_quaternions])

    #     # Compute time differences
    #     dt = np.diff(imu_timestamps)

    #     # Compute optimized quaternion trajectory
    #     print("Computing optimized quaternion trajectory...")
    #     optimized_quaternions = optimize_quaternion_trajectory(
    #         jnp.array(gyro_calibrated.transpose()),
    #         jnp.array(calibrated_data["accel_calibrated"].transpose()),
    #         jnp.array(dt),
    #         num_iterations=100,
    #         lr=0.01
    #     )

    #     # Convert optimized quaternions to Euler angles
    #     optimized_euler = np.array([euler.quat2euler(q) for q in optimized_quaternions])

    #     # Plot all trajectories
    #     print("Plotting results...")
    #     plot_comparison_trajectories(
    #         imu_timestamps,
    #         unoptimized_euler,
    #         optimized_euler,
    #         vicon_euler,
    #         dataset_index=j + 1
    #     )

        # exit()

    '''
    
    Test set
    
    '''

    # test_imu_dir = "../testset/imu"
    # test_datasets = load_all_datasets(test_imu_dir, test=True)
    # process_test_set(
    #     extract_imu_raw(test_datasets, test=True),  # List of raw IMU test datasets
    #     calibration_results,  # Calibration results from the training set
    #     compute_predicted_quaternions,  # Function for unoptimized prediction
    #     optimize_quaternion_trajectory,  # Function for optimization
    #     plot_test_set_results  # Plotting function
    # )

    '''
    
    Debug
    
    '''
    # # Initialize quaternion trajectory
    # for j, calibrated_data in enumerate(calibrated_datasets):
    #     gyro_calibrated = calibrated_data["gyro_calibrated"]
    #     print("Initializing quaternion trajectory...")
    #     q_traj = np.zeros((gyro_calibrated.shape[1], 4))
    #     q_traj[0] = [1.0, 0.0, 0.0, 0.0]  # Start with identity quaternion
    #     accel_calibrated = calibrated_data["accel_calibrated"]
    #     imu_timestamps = calibrated_data["ts"]
    #     dt = np.diff(imu_timestamps)

    #     # Check observation model gravity
    #     print("Checking observation model gravity...")
    #     check_observation_model_gravity(
    #         q_traj=jnp.array(q_traj),
    #         accel_data=jnp.array(accel_calibrated.transpose())
    #     )

        # Check initial motion model cost
        # print("Checking initial motion model cost...")
        # check_motion_model_cost(
        #     q_traj=jnp.array(q_traj),
        #     omega=jnp.array(gyro_calibrated.transpose()),
        #     dt=jnp.array(dt)
        # )

    '''
    
    accel debugging
    
    '''
    # for j, calibrated_data in enumerate(calibrated_datasets):
    #     if j < 3:
    #         continue
    #     print(f"Processing dataset {j + 1}...")
    #     imu_timestamps = calibrated_data["ts"]
    #     accel_calibrated = calibrated_data["accel_calibrated"]
    #     gyro_calibrated = calibrated_data["gyro_calibrated"]

    #     print("Computing quaternion trajectory...")
    #     predicted_quaternions = compute_predicted_quaternions(gyro_calibrated, imu_timestamps)

    #     print("Testing observation model...")
    #     # Compute estimated accelerations using the observation model
    #     g_world = np.array([0, 0, 1])  # Gravity vector in the world frame
    #     estimated_accel = np.array([
    #         observation_model(q, g=g_world)[1:] for q in predicted_quaternions
    #     ])  # Ignore the scalar part of the quaternion

    #     print("Plotting results...")
    #     labels = ['X-axis', 'Y-axis', 'Z-axis']
    #     colors = ['r', 'g', 'b']

    #     plt.figure(figsize=(12, 8))
    #     for i in range(3):  # Plot for each axis
    #         plt.subplot(3, 1, i + 1)
    #         plt.plot(imu_timestamps, accel_calibrated[i, :], label=f'Ground Truth {labels[i]}', color=colors[i], linestyle='-')
    #         plt.plot(imu_timestamps, estimated_accel[:, i], label=f'Estimated {labels[i]}', color=colors[i], linestyle='--')
    #         plt.xlabel('Time (s)')
    #         plt.ylabel('Acceleration (m/s²)')
    #         plt.legend()
    #         plt.grid()

    #     plt.suptitle(f'Dataset {j + 1}: Observation Model - Estimated vs Ground Truth Acceleration')
    #     plt.tight_layout()
    #     plt.show()

    #     # Exit after the first dataset for testing purposes
    #     exit()


    '''
    
    Panorama
    
    '''

    test_imu_dir = "../testset/imu"
    test_datasets = load_all_datasets(test_imu_dir, test_quat=True)
    process_test_set(
        extract_imu_raw(test_datasets, test=True),  # List of raw IMU test datasets
        calibration_results,  # Calibration results from the training set
        compute_predicted_quaternions,  # Function for unoptimized prediction
        optimize_quaternion_trajectory,  # Function for optimization
        plot_test_set_results  # Plotting function
    )


if __name__ == "__main__":
    main()
