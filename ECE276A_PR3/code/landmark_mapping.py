import numpy as np
import cv2
from scipy.sparse import csr_matrix, lil_matrix, eye, kron
from scipy.sparse.linalg import spsolve
from pr3_utils import *
from localization import imu_ekf_prediction


def compute_stereo_calibration(K_l, K_r, extL_T_imu, extR_T_imu):
    f_u_l, f_v_l, c_u_l, c_v_l = K_l[0,0], K_l[1,1], K_l[0,2], K_l[1,2]
    f_u_r, f_v_r, c_u_r, c_v_r = K_r[0,0], K_r[1,1], K_r[0,2], K_r[1,2]

    # Correct baseline calculation explicitly in optical frame:
    R_regular_to_optical = np.array([[0, -1, 0],
                                     [0,  0, -1],
                                     [1,  0, 0]])
    baseline_vector_reg = extR_T_imu[:3, 3] - extL_T_imu[:3, 3]
    baseline_vector_opt = R_regular_to_optical @ baseline_vector_reg
    b = np.abs(baseline_vector_opt[0])

    # Now correctly define Ks
    K_s = np.array([
        [f_u_l, 0, c_u_l, 0],
        [0, f_v_l, c_v_l, 0],
        [f_u_r, 0, c_u_r, -f_u_r * b],
        [0, f_v_r, c_v_r, 0]
    ])
    
    return K_s


def initialize_landmarks(features, K_l, K_r, extL_T_imu, extR_T_imu, imu_poses, max_distance=40):
    """
    Initialize landmark positions using the first valid observation of each landmark,
    performing batch triangulation and transforming to world coordinates.
    """
    M = features.shape[1]  # Number of landmarks
    T = features.shape[2]  # Number of time steps
    mu = np.zeros((3, M))

     # Regular to optical rotation
    R_regular_to_optical = np.array([[0, -1, 0],
                                    [0,  0, -1],
                                    [1,  0,  0]])

    T_regular_to_optical = np.eye(4)
    T_regular_to_optical[:3,:3] = R_regular_to_optical

    # Projection matrices for left and right cameras
    baseline = np.linalg.norm(extR_T_imu[:3, 3] - extL_T_imu[:3, 3])
    P_l = K_l @ np.hstack((np.eye(3), np.zeros((3,1))))
    P_r = K_r @ np.hstack((np.eye(3), np.array([[-baseline,0,0]]).T))

    
    # Find the first valid observation for each landmark across all timesteps
    valid_mask = np.any(features != -1, axis=0)  # Shape: (M, T), True where feature is observed
    first_obs_idx = np.argmax(valid_mask, axis=1)  # Get first valid observation index per feature

    # Extract the corresponding pixel coordinates for triangulation
    left_pts = features[:2, np.arange(M), first_obs_idx]  # (2, M)
    right_pts = features[2:, np.arange(M), first_obs_idx]  # (2, M)
    
    # Triangulate in batch
    pts_4D = cv2.triangulatePoints(P_l, P_r, left_pts, right_pts)
    landmarks_3D = (pts_4D[:3] / pts_4D[3])  # Convert from homogeneous to Cartesian
    
    # Get the IMU pose at the first observation of each landmark
    T_first_obs = imu_poses[first_obs_idx]  # Shape: (M, 4, 4)
    
    # # Convert landmarks to world coordinates
    landmarks_3D_hom = np.vstack((landmarks_3D, np.ones((1, M))))  # Convert to homogeneous coordinates
    mu = np.einsum('mij,jm->im', T_first_obs @ inversePose(extL_T_imu) @ inversePose(T_regular_to_optical), landmarks_3D_hom)[:3]  # Batch transformation, in world frame

    # Compute distances from landmarks to IMU trajectory in batch
    distances = np.linalg.norm(imu_poses[:, :3, 3][np.newaxis, :, :] - mu.T[:, np.newaxis, :], axis=2)
    min_distances = np.min(distances, axis=1)
    
    # Filter landmarks based on distance
    valid_landmarks = min_distances <= max_distance
    mu = mu[:, valid_landmarks]

    # Create a mapping from original indices to new valid indices
    landmark_map = np.full(M, -1)  # Initialize mapping array (-1 for removed landmarks)
    landmark_map[valid_landmarks] = np.arange(np.sum(valid_landmarks))  # Map old indices to new
    
    ### --- PROJECTION CHECK STARTS HERE --- ###

    # Convert filtered landmarks (mu) into homogeneous coordinates
    mu_hom = np.vstack([mu, np.ones((1, mu.shape[1]))])  # Shape: (4, M_valid)

    # Transform to left camera frame for projection
    q = np.zeros((mu.shape[1], 4))
    for i in range(mu.shape[1]):
        q[i] = T_regular_to_optical @ extL_T_imu @ inversePose(T_first_obs[valid_landmarks][i]) @ mu_hom[:, i]


    # Project using the provided projection function
    projected_points = projection(q)  # Shape: (M_valid, 4)

    # Extract only x, y for pixel coordinates
    reprojected_pixel_coords = ((K_l @ projected_points[:, :3].T).T)[:, :2]  # Shape: (M_valid, 2)
    
    # Get the actual observed pixel locations (only for valid indices)
    left_pts_valid = features[:2, np.where(valid_landmarks)[0], first_obs_idx[valid_landmarks]].T  # Shape: (M_valid, 2)

    # Compute reprojection error
    errors = np.linalg.norm(reprojected_pixel_coords - left_pts_valid, axis=1)  # Euclidean distance error
    mean_error = np.mean(errors)

    print(f"[Projection Check] Mean reprojection error: {mean_error:.6f}")

    ### --- PROJECTION CHECK ENDS HERE --- ###

    return mu, landmark_map




def ekf_landmark_mapping(imu_poses, features, K_s, extL_T_imu, mu, landmark_map):
    """
    Implements the EKF update step for landmark mapping with optimized sparse matrix operations.
    Only updates the landmark mean for the landmarks that are observed at each time step.

    :param imu_poses: (N, 4, 4) array of IMU poses.
    :param features: (4, num_features, N) array of feature observations.
    :param K_s: Stereo calibration matrix (4x4).
    :param extL_T_imu: Transformation from IMU to left camera (4x4).
    :param mu: (3, M) initial landmark means (filtered landmarks).
    :param landmark_map: Mapping from original feature indices to filtered landmark indices.
    :return: Updated mu and covariance Sigma (mu: (3, M), Sigma: (3*M, 3*M)).
    """
    M = mu.shape[1]  # Number of filtered landmarks
    N = features.shape[2]  # Number of time steps

    # Initialize sparse covariance matrix (3*M x 3*M)
    Sigma = csr_matrix(eye(3 * M) * 0.1)  # Small initial uncertainty

    # Measurement noise covariance (for a single landmark measurement, 4x4)
    V = 100*eye(4)
    mu_orig = mu.copy()

    # Regular to optical rotation
    R_regular_to_optical = np.array([[0, -1, 0],
                                    [0,  0, -1],
                                    [1,  0,  0]])

    T_regular_to_optical = np.eye(4)
    T_regular_to_optical[:3,:3] = R_regular_to_optical

    for t in range(N):
        print("timestamp:", t)
        T_t = imu_poses[t]            # Current IMU pose (4x4)
        inv_Tt = inversePose(T_t)    # Inverse pose

        # Identify original indices of landmarks observed at time t (i.e., features not equal to -1)
        orig_indices = np.where(np.any(features[:, :, t] != -1, axis=0))[0]

        # Map these indices to the filtered landmark indices and collect the corresponding measurements.
        valid_obs = []
        z_actual_list = []
        for idx in orig_indices:
            new_idx = landmark_map[idx]
            if new_idx != -1:
                valid_obs.append(new_idx)
                z_actual_list.append(features[:, idx, t])
        if len(valid_obs) == 0:
            continue

        # Use the filtered indices for this update
        obs_indices = np.array(valid_obs)  # indices in the filtered state (subset of 0,...,M-1)
        num_obs = len(obs_indices)

        # Build homogeneous coordinates for the observed landmarks
        m_obs_h = np.vstack([mu[:, obs_indices], np.ones(num_obs)])  # shape: (4, num_obs)

        # Transform landmarks into the left camera frame:
        # q = extL_T_imu * T_t^{-1} * m_obs_h
        # q = extL_T_imu @ inv_Tt @ m_obs_h  # shape: (4, num_obs)
        q = np.einsum('ij,jk->ki', T_regular_to_optical @ extL_T_imu @ inv_Tt, m_obs_h)  # Result: (28, 4)

        # Compute projection Jacobians for each observed landmark using the helper function.
        # J_pi = np.array([projectionJacobian(q[i]) for i in range(num_obs)])
        J_pi = projectionJacobian(q)

        # Construct the measurement Jacobian H_t as a sparse matrix.
        # H_t has dimensions (4*num_obs) x (3*M), and we only fill blocks corresponding to the observed landmarks.
        P = np.eye(3, 4)
        H_t = lil_matrix((4 * num_obs, 3 * M))
        for i, j in enumerate(obs_indices):
            H_j = K_s @ J_pi[i] @ T_regular_to_optical @ extL_T_imu @ inv_Tt @ P.T
            H_t[4 * i: 4 * (i + 1), 3 * j: 3 * (j + 1)] = H_j
        H_t = H_t.tocsc()  # Convert to CSC format for efficient solves

        # Predicted observations for the observed landmarks: z_pred = K_s * projection(q)
        # z_pred = np.hstack([ (K_s @ projection(q[i])).flatten() for i in range(num_obs) ])
        z_pred = np.einsum('ij,kj->ki', K_s, projection(q)).flatten()

        # Actual observed measurements (concatenated into a single vector)
        z_actual = np.hstack(z_actual_list)

        # Compute innovation (measurement residual)
        innovation = z_actual - z_pred

        # print("Measurement residuals:", np.mean(z_actual - z_pred, axis=0))

        # Compute Kalman gain: K_t = Sigma * H_t^T * (H_t * Sigma * H_t^T + I⊗V)^{-1}
        S = H_t @ Sigma @ H_t.T + kron(eye(num_obs), V).tocsc()
        S_inv = spsolve(S, eye(4 * num_obs))
        # print(type(S))
        # S_inv = np.linalg.inv(S.toarray())
        K_t = Sigma @ H_t.T @ S_inv
        K_t = csr_matrix(K_t)

        # Compute the state correction (full state dimension: 3*M)
        # delta_full = (K_t @ innovation).reshape(3, M)
        # print(delta_full)

        # Compute the state correction ONLY for the observed landmarks
        delta_obs = (K_t @ innovation).reshape(-1, 3).T  # Reshape directly to (3, num_obs)

        # Update only the observed landmarks in mu.
        # mu[:, obs_indices] += delta_obs
        mu += delta_obs

        # Update only the observed landmarks in mu.
        # mu[:, obs_indices] = mu[:, obs_indices] + delta_full[:, obs_indices]

        # Update covariance.
        Sigma = (eye(3 * M, format='csr') - K_t @ H_t) @ Sigma
        # exit()
    print(np.linalg.norm(mu_orig - mu))
    return mu, Sigma


def visualize_trajectory_and_landmarks(imu_poses, mu, show_ori=False, path_name=None):
    """
    Visualizes the IMU trajectory and estimated landmark positions, with optional orientation display.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    n_pose = imu_poses.shape[0]
    
    # Plot IMU trajectory
    ax.plot(imu_poses[:, 0, 3], imu_poses[:, 1, 3], 'r-', label='IMU Trajectory')
    ax.scatter(imu_poses[0, 0, 3], imu_poses[0, 1, 3], marker='s', label="Start")
    ax.scatter(imu_poses[-1, 0, 3], imu_poses[-1, 1, 3], marker='o', label="End")
    
    # Plot estimated landmark positions
    ax.scatter(mu[0, :], mu[1, :], s=2, c='b', label='Landmarks')
    
    if show_ori:
        select_ori_index = list(range(0, n_pose, max(int(n_pose / 50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _, _, yaw = mat2euler(imu_poses[i, :3, :3])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx, dy = [dx, dy] / np.sqrt(dx**2 + dy**2)
        ax.quiver(imu_poses[select_ori_index, 0, 3], imu_poses[select_ori_index, 1, 3], dx, dy,
                  color="b", units="xy", width=1, headlength=0.002, headaxislength=0.001)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('IMU Trajectory and Estimated Landmarks')
    ax.legend()
    ax.grid()
    if path_name is not None:
        plt.savefig(path_name)
    plt.show()

    return fig, ax

def visualize_trajectory_and_landmarks_2d(imu_poses, mu_original, mu_ekf, path_name=None):
    """
    Visualizes the IMU trajectory and estimated landmark positions (before and after EKF update) in 2D.

    :param imu_poses: IMU trajectory (N, 4, 4)
    :param mu_original: Landmarks before EKF update (3, M)
    :param mu_ekf: Landmarks after EKF update (3, M)
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract IMU trajectory positions from transformation matrices
    imu_positions = imu_poses[:, :3, 3]  # Shape: (N, 3)

    # Ensure original and EKF landmarks are separate copies
    mu_original = np.copy(mu_original)  # Prevent overwriting issues
    mu_ekf = np.copy(mu_ekf)

    # Plot IMU trajectory (red line)
    ax.plot(imu_positions[:, 0], imu_positions[:, 1], 'r-', label="IMU Trajectory")

    # Plot start and end points of the trajectory
    ax.scatter(imu_positions[0, 0], imu_positions[0, 1], marker='s', color='red', label="Start")
    ax.scatter(imu_positions[-1, 0], imu_positions[-1, 1], marker='o', color='red', label="End")

    # Plot original triangulated landmarks (blue)
    ax.scatter(mu_original[0, :], mu_original[1, :], s=10, c='blue', label="Original Landmarks", alpha=0.6)

    # Plot EKF-updated landmarks (green)
    ax.scatter(mu_ekf[0, :], mu_ekf[1, :], s=10, c='green', label="EKF Updated Landmarks", alpha=0.6)

    # Labels, legend, and grid
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('IMU Trajectory and Landmarks (Before & After EKF)')
    ax.legend()
    ax.grid()
    if path_name is not None:
        plt.savefig(path_name)
    plt.show()


# Load dataset00 and run EKF landmark mapping
if __name__ == '__main__':
    filename = "../data/dataset02/dataset02.npy"
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(filename)
    
    if filename == "../data/dataset02/dataset02.npy":
        features = np.load("feature_measurements.npy")

    # Compute stereo calibration matrix
    K_s = compute_stereo_calibration(K_l, K_r, extL_T_imu, extR_T_imu)

    # Define process noise covariance (tune as needed)
    W = np.diag([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])  # Small uncertainty
    
    # Run IMU localization using EKF prediction
    imu_poses, imu_covariances = imu_ekf_prediction(v_t, w_t, timestamps, W)
    
    
    # # Initialize landmarks using triangulation with filtering
    mu_orig, landmark_map = initialize_landmarks(features, K_l, K_r, extL_T_imu, extR_T_imu, imu_poses)
    # exit()
    mu_original = mu_orig.copy()
    # Visualize trajectory and landmarks with optional orientation
    visualize_trajectory_and_landmarks(imu_poses, mu_orig, show_ori=False, path_name="./results/initial_landmarks_02")

    # Run EKF landmark mapping
    mu, Sigma = ekf_landmark_mapping(imu_poses, features, K_s, extL_T_imu, mu_orig, landmark_map)
    
    # Visualize trajectory and landmarks with optional orientation
    visualize_trajectory_and_landmarks(imu_poses, mu, show_ori=False, path_name="./results/updated_landmarks_02")

    visualize_trajectory_and_landmarks_2d(imu_poses, mu_original, mu, path_name="./results/landmarks_overlayed_02")

    # # Display results
    # print("Estimated landmark positions:", mu)
