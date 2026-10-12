import numpy as np
from pr3_utils import *


if __name__ == '__main__':

    # Load the measurements
    # filename = "../data/dataset02/dataset02_imgs.npy"
    # # v_t,w_t,timestamps,features,K_l,K_r,extL_T_imu,extR_T_imu = load_data(filename)
    # test = load_data(filename)
    # Load the file with allow_pickle enabled and extract the dictionary
	# imgs_dict = np.load("../data/dataset02/dataset02_imgs.npy", allow_pickle=True).item()

	# # Extract left and right images from the dictionary
	# left_imgs = imgs_dict['cam_imgs_L']
	# right_imgs = imgs_dict['cam_imgs_R']
	# print(type(left_imgs[0]))
    
	features = np.load("feature_measurements.npy")
	print(features[:,0,0])
    
	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)


