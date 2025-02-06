import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from load_all import load_all_datasets, extract_imu_raw
from calibrate_imu import calibrate_datasets
from quaternion import optimize_quaternion_trajectory,  quaternion_multiply
import transforms3d.quaternions as quat
import transforms3d.euler as euler
import jax.numpy as jnp



def plot_predicted_quaternions_vs_vicon(imu_timestamps, predicted_euler, vicon_euler):
    """
    Plot roll, pitch, and yaw for predicted and VICON quaternions.

    Parameters:
        imu_timestamps (numpy.ndarray): Timestamps for the predicted quaternions.
        predicted_quaternions (numpy.ndarray): Predicted quaternion trajectory (Nx4).
        vicon_rotations (numpy.ndarray): VICON rotation matrices (Nx3x3).
    """

    # Convert VICON rotation matrices to quaternions

    # predicted_euler = np.array([quat2euler(q, axes='sxyz') for q in predicted_quaternions])
    # vicon_euler = np.array([quat2euler(q, axes='sxyz') for q in vicon_rotations])

    labels = ['Roll (degrees)', 'Pitch (degrees)', 'Yaw (degrees)']
    colors = ['b', 'g', 'r']

    plt.figure(figsize=(12, 8))


    for i, label in enumerate(labels):
        plt.subplot(3, 1, i + 1)
        plt.plot(imu_timestamps, np.degrees(predicted_euler[:,i]), linestyle='--', color=colors[i], label=f'Predicted {label}')
        plt.plot(imu_timestamps, np.degrees(vicon_euler[:, i]), color='k', label=f'VICON {label}')
        plt.xlabel('Time (s)')
        plt.ylabel(label)
        plt.legend()
        plt.grid()

    plt.suptitle('Predicted Quaternion vs. VICON Ground Truth')
    plt.tight_layout()
    plt.show()


def plot_comparison_trajectories(imu_timestamps, unoptimized_euler, optimized_euler, vicon_euler, dataset_index):
    """
    Plot roll, pitch, and yaw for unoptimized, optimized, and VICON trajectories.

    Parameters:
        imu_timestamps (numpy.ndarray): Timestamps for the predicted quaternions.
        unoptimized_euler (numpy.ndarray): Unoptimized quaternion trajectory converted to Euler angles (Nx3).
        optimized_euler (numpy.ndarray): Optimized quaternion trajectory converted to Euler angles (Nx3).
        vicon_euler (numpy.ndarray): VICON ground truth Euler angles (Nx3).
        dataset_index (int): Index of the dataset for labeling.
    """
    labels = ['Roll (degrees)', 'Pitch (degrees)', 'Yaw (degrees)']
    colors = ['b', 'g', 'r']

    plt.figure(figsize=(12, 8))

    for i, label in enumerate(labels):
        plt.subplot(3, 1, i + 1)
        # Plot unoptimized trajectory
        plt.plot(imu_timestamps, np.degrees(unoptimized_euler[:, i]), linestyle='--', color=colors[i], label=f'Unoptimized {label}')
        # Plot optimized trajectory
        plt.plot(imu_timestamps, np.degrees(optimized_euler[:, i]), linestyle='-', color=colors[i], alpha=0.7, label=f'Optimized {label}')
        # Plot VICON ground truth
        plt.plot(imu_timestamps, np.degrees(vicon_euler[:, i]), linestyle='-', color='k', label=f'VICON {label}')
        
        plt.xlabel('Time (s)')
        plt.ylabel(label)
        plt.legend()
        plt.grid()

    plt.suptitle(f'Comparison of Pitch, Roll, and Yaw: Dataset {dataset_index}')
    plt.tight_layout()

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    save_path = os.path.join(results_dir, f'comparison_dataset_{dataset_index}.jpg')
    plt.savefig(save_path, format='jpg', dpi=300)

    plt.show()

def plot_test_set_results(imu_timestamps, unoptimized_euler, optimized_euler, dataset_index):
    """
    Plot roll, pitch, and yaw for unoptimized and optimized quaternion trajectories for the test set.

    Parameters:
        imu_timestamps (numpy.ndarray): Timestamps for the IMU data.
        unoptimized_euler (numpy.ndarray): Unoptimized quaternion trajectory in Euler angles (Nx3).
        optimized_euler (numpy.ndarray): Optimized quaternion trajectory in Euler angles (Nx3).
        dataset_index (int): Index of the dataset for labeling.

    Returns:
        None
    """
    labels = ['Roll (degrees)', 'Pitch (degrees)', 'Yaw (degrees)']
    colors = ['b', 'g', 'r']

    plt.figure(figsize=(12, 8))

    for i, label in enumerate(labels):
        plt.subplot(3, 1, i + 1)
        # Plot unoptimized trajectory
        plt.plot(imu_timestamps, np.degrees(unoptimized_euler[:, i]), linestyle='--', color=colors[i], label=f'Unoptimized {label}')
        # Plot optimized trajectory
        plt.plot(imu_timestamps, np.degrees(optimized_euler[:, i]), linestyle='-', color=colors[i], alpha=0.7, label=f'Optimized {label}')
        plt.xlabel('Time (s)')
        plt.ylabel(label)
        plt.legend()
        plt.grid()

    plt.suptitle(f'Test Set: Comparison of Pitch, Roll, and Yaw (Dataset {dataset_index})')
    plt.tight_layout()

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    save_path = os.path.join(results_dir, f'comparison_dataset_{dataset_index}.jpg')
    plt.savefig(save_path, format='jpg', dpi=300)

    plt.show()
