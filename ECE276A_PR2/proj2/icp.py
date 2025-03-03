import numpy as np
from scipy.spatial import cKDTree  # Faster KDTree
import os
from icp_warm_up.utils import read_canonical_model, load_pc, visualize_icp_result

def create_pose(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def kabsch(source, target):
    source_mean = np.mean(source, axis=1)
    target_mean = np.mean(target, axis=1)

    source_centered = source - source_mean[:, np.newaxis]
    target_centered = target - target_mean[:, np.newaxis]

    Q = source_centered @ target_centered.T
    # Q = np.dot(source_centered.T, target_centered)
    U, Sigma, Vt = np.linalg.svd(Q)
    # print("past Q")
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # D = np.eye(3)
    # D[-1,-1] = np.linalg.det(U @ Vt)

    # R = U @ D @ Vt
    # p = target_mean - R @ source_mean
    p = target_mean - np.dot(R,source_mean.T)

    return R, p

def downsample_pc(pc, num_points=5000):
    """Downsample point cloud for faster processing."""
    if pc.shape[1] > num_points:
        indices = np.random.choice(pc.shape[1], num_points, replace=False)
        return pc[:, indices]
    return pc


def icp(target, source, R_init=np.eye(3), p_init=np.zeros(3), max_iters=100, tolerance=1e-12):
    # R_curr = np.eye(3)
    R_curr = R_init
    # p_curr = np.zeros(3)
    p_curr = p_init
    prev_loss = np.inf
    T_curr = create_pose(R_curr, p_curr)

    # Downsample point clouds before ICP
    source = downsample_pc(source, num_points=1000)
    target = downsample_pc(target, num_points=1000)

    transformed_source = (R_curr @ source) + p_curr[:, np.newaxis]

    for i in range(max_iters):

        # Ensure we find correspondences in the correct direction
        tree = cKDTree(target.T)
        _, indices = tree.query(transformed_source.T)
        matched_target = target[:, indices]

        # Ensure Kabsch gets the correct frame
        R_next, p_next = kabsch(transformed_source, matched_target)  
        T_next = create_pose(R_next, p_next)
        T_curr = T_next @ T_curr
        
        transformed_source = (T_curr[0:3, 0:3] @ source) + T_curr[:3, 3][:, np.newaxis]

        tree = cKDTree(target.T)
        _, indices = tree.query(transformed_source.T)
        matched_target = target[:, indices]

        loss = np.mean(np.linalg.norm(matched_target - transformed_source, axis=0))

        # Stop early if loss does not significantly change
        if abs(prev_loss - loss) < tolerance:
            # print(f"ICP converged at iteration {i}")
            break

        
        prev_loss = loss

    return T_curr


def yaw_rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def find_best_icp_init(target, source, R=np.eye(3), p=np.zeros(3)):
    best_pose = None
    best_loss = np.inf
    yaw_angles = np.linspace(-np.pi, np.pi, num=10)

    for theta in yaw_angles:

        R_init = yaw_rotation_matrix(theta)

        init_pose = icp(source, target, R_init)
        
        transformed_source = (init_pose[0:3, 0:3] @ target) + init_pose[:3, 3][:, np.newaxis]


        tree = cKDTree(source.T)
        _, indices = tree.query(transformed_source.T)
        matched_target = source[:, indices]  # Ensure correct point associations
        
        # Compute loss only on matched points

        loss = np.mean(np.linalg.norm(matched_target - transformed_source, axis=0))
        print("Loss: ", loss)
        if loss < best_loss:
            best_loss = loss
            best_pose = init_pose
            # visualize_icp_result(source.T, target.T, best_pose)
        # exit()
    
    return best_pose

def test_icp():
    objects = ['drill', 'liq_container']
    num_pcs = 4
    
    for obj in objects:
        # obj = 'liq_container'
        source = read_canonical_model(obj)
        
        for i in range(num_pcs):
            
            target = load_pc(obj, i)
            best_pose = find_best_icp_init(target.T, source.T)
            print(f"Best Estimated Pose for {obj} scan {i}:")

            # visualize_icp_result(source, target, best_pose)
            visualize_icp_result(target, source, best_pose)
            
            # exit()

if __name__ == "__main__":
    test_icp()
