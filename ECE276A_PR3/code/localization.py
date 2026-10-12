import numpy as np
from pr3_utils import *


def imu_ekf_prediction(v_t, w_t, timestamps, W, initial_pose=None, initial_cov=None):
    """
    Implements the EKF prediction step for IMU localization using SE(3) kinematics.
    
    :param v_t: Linear velocity measurements (Nx3)
    :param w_t: Angular velocity measurements (Nx3)
    :param timestamps: Time stamps (Nx1)
    :param W: Process noise covariance (6x6)
    :param initial_pose: Initial IMU pose (4x4 SE(3))
    :param initial_cov: Initial covariance (6x6)
    :return: Estimated IMU trajectory (Nx4x4), Covariance estimates (Nx6x6)
    """
    N = v_t.shape[0]  # Number of time steps
    
    # Initialize IMU state
    if initial_pose is None:
        mu_t = np.eye(4)  # Start with identity matrix
    else:
        mu_t = initial_pose
    
    if initial_cov is None:
        Sigma_t = np.eye(6) * 1e-3  # Small initial uncertainty
    else:
        Sigma_t = initial_cov
    
    imu_trajectory = np.zeros((N, 4, 4))
    imu_covariances = np.zeros((N, 6, 6))
    
    imu_trajectory[0] = mu_t
    imu_covariances[0] = Sigma_t
    
    u_t = np.hstack((v_t, w_t))
    u_hat_t = axangle2twist(u_t)

    u_curly_hat_t = pose2adpose(u_hat_t)


    for t in range(1, N):
        tau_t = timestamps[t] - timestamps[t - 1]  # Time step

        exp_tau_u = twist2pose(tau_t * u_hat_t[t])

        mu_t = mu_t @ exp_tau_u

        exp_tau_curly_u = twist2pose(tau_t * u_curly_hat_t[t])
        Sigma_t = exp_tau_curly_u @ Sigma_t @ exp_tau_curly_u.T + tau_t * W

        # Store results
        imu_trajectory[t] = mu_t
        imu_covariances[t] = Sigma_t

    
    return imu_trajectory, imu_covariances

if __name__ == '__main__':
    # Load the dataset
    filename = "../data/dataset00/dataset00.npy"
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(filename)

    # Define process noise covariance (tune as needed)
    W = np.diag([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])  # Small uncertainty
    
    # Run EKF Prediction Step
    imu_trajectory, imu_covariances = imu_ekf_prediction(v_t, w_t, timestamps, W)

    # Visualize the trajectory
    visualize_trajectory_2d(imu_trajectory, show_ori=False, path="./results/localization_00.png", path_name="dataset00")

