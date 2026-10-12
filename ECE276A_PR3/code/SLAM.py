import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, eye, kron, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import block_diag
from pr3_utils import *
from landmark_mapping import initialize_landmarks, compute_stereo_calibration, visualize_trajectory_and_landmarks, visualize_trajectory_and_landmarks_2d
from localization import imu_ekf_prediction

def circle_dot(s):
    """
    Implements the circle-dot operation for Jacobian.
    s: homogeneous coordinates (4,)
    """
    s_skew = axangle2skew(s[:3])
    circ_dot = np.zeros((4, 6))
    circ_dot[:3, :3] = np.eye(3)
    circ_dot[:3, 3:] = -s_skew
    return circ_dot


def visual_inertial_slam(v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu, W, V):
    N = v_t.shape[0]

    # Step 1: Full IMU EKF Prediction
    imu_poses_pred, imu_covariances = imu_ekf_prediction(v_t, w_t, timestamps, W)

    # Step 2: Landmark Initialization using the entire IMU trajectory
    mu_landmarks_original, landmark_map = initialize_landmarks(
        features, K_l, K_r, extL_T_imu, extR_T_imu, imu_poses_pred, max_distance=1)

    mu_landmarks = mu_landmarks_original.copy()
    M = mu_landmarks.shape[1]

    # Stereo Calibration Matrix
    Ks = compute_stereo_calibration(K_l, K_r, extL_T_imu, extR_T_imu)

    # Initialize covariance
    Sigma_t = block_diag(imu_covariances[0], np.eye(3 * M) * 1)

    V_obs = np.eye(4) * V

    # Precompute constant transformations
    T_regular_to_optical = np.eye(4)
    T_regular_to_optical[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    extL_T_imu_optical = T_regular_to_optical @ extL_T_imu
    P = np.eye(3, 4)

    Sigma_sparse = csc_matrix(Sigma_t)
    imu_poses_updated = imu_poses_pred.copy()
    counter = 0
    for t in range(N): #N
        print("timestamps: ", t)
        if counter != 100:
            counter +=1
            continue
        counter=0
        mu_t = imu_poses_updated[t]
        Sigma_sparse[:6, :6] = csc_matrix(imu_covariances[t])

        inv_mu_t = inversePose(mu_t)
        ext_inv_mu_t = extL_T_imu_optical @ inv_mu_t

        obs_indices = np.where(np.any(features[:, :, t] != -1, axis=0))[0]
        obs_indices_filtered = [landmark_map[idx] for idx in obs_indices if landmark_map[idx] != -1]

        if len(obs_indices_filtered) > 0:
            
            num_obs = len(obs_indices_filtered)
            H_t = lil_matrix((4 * num_obs, 6 + 3 * M))
            innovation = np.zeros(4 * num_obs)

            for i, lm_idx in enumerate(obs_indices_filtered):
                landmark_pos = mu_landmarks[:, lm_idx]
                landmark_hom = np.hstack((landmark_pos, 1))

                q = ext_inv_mu_t @ landmark_hom

                z_pred = Ks @ projection(q)
                z_act = features[:, obs_indices[i], t]
                innovation[4*i:4*(i+1)] = z_act - z_pred

                if np.any(np.abs(innovation) > 1e6):
                    continue

                H_i_pose = -Ks @ projectionJacobian(q) @ ext_inv_mu_t @ circle_dot(inv_mu_t @ landmark_hom)
                H_t[4*i:4*(i+1), :6] = H_i_pose

                landmark_jac_idx = 6 + 3*lm_idx
                H_i_landmark = Ks @ projectionJacobian(q) @ ext_inv_mu_t @ P.T
                H_t[4*i:4*(i+1), landmark_jac_idx:landmark_jac_idx+3] = H_i_landmark

            H_t = csc_matrix(H_t)

            S_t = H_t @ Sigma_sparse @ H_t.T + kron(eye(num_obs), csc_matrix(V_obs))
            try:
                K_t = Sigma_sparse @ H_t.T @ spsolve(S_t + 1e-6 * eye(S_t.shape[0], format='csc'), eye(4 * num_obs, format='csc')).T
            except RuntimeError:
                print(f"Singularity encountered at timestamp {t}, skipping update.")
                continue

            delta_x = np.clip(K_t @ innovation, -1e3, 1e3)
            imu_poses_updated[t] = imu_poses_updated[t] @ twist2pose(axangle2twist(delta_x[:6]))

            landmark_updates = delta_x[6:].reshape((M, 3)).T
            for idx, lm_idx in enumerate(obs_indices_filtered):
                mu_landmarks[:, lm_idx] += landmark_updates[:, lm_idx]

            Sigma_sparse = (eye(6 + 3*M, format='csc') - K_t @ H_t) @ Sigma_sparse

    return imu_poses_pred, imu_poses_updated, mu_landmarks_original, mu_landmarks, Sigma_sparse.toarray()
def visualize_trajectory_comparison(poses_pred, poses_updated):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract XY positions
    pred_xy = poses_pred[:, :2, 3]
    updated_xy = poses_updated[:, :2, 3]

    ax.plot(pred_xy[:, 0], pred_xy[:, 1], 'r--', label="Predicted IMU Trajectory")
    ax.plot(updated_xy[:, 0], updated_xy[:, 1], 'g-', label="Updated IMU Trajectory")

    ax.scatter(pred_xy[0, 0], pred_xy[0, 1], marker='s', color='red', label="Start")
    ax.scatter(updated_xy[-1, 0], updated_xy[-1, 1], marker='o', color='green', label="End")

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('IMU Trajectory Before and After EKF Update')
    ax.legend()
    ax.grid()
    plt.show()


if __name__ == '__main__':
    filename = "../data/dataset02/dataset02.npy"
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(filename)

    if filename == "../data/dataset02/dataset02.npy":
        features = np.load("feature_measurements.npy")

    W = np.diag([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
    V = 10000000  # Increased measurement noise to enhance numerical stability

    imu_pred, imu_updated, lm_orig, lm_upd, Sigma = visual_inertial_slam(
        v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu, W, V)

    visualize_trajectory_and_landmarks(imu_updated, lm_upd, show_ori=False)
    visualize_trajectory_and_landmarks_2d(imu_updated, lm_orig, lm_upd)
    visualize_trajectory_comparison(imu_pred, imu_updated)