import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_initial_features(img, max_corners=200, quality_level=0.01, min_distance=7):
    """
    Detect features in the given grayscale image using the Shi-Tomasi corner detector.
    
    :param img: Grayscale image.
    :param max_corners: Maximum number of features to detect.
    :param quality_level: Minimum accepted quality.
    :param min_distance: Minimum distance between detected features.
    :return: Array of detected feature points of shape (N, 2).
    """
    features = cv2.goodFeaturesToTrack(img, maxCorners=max_corners,
                                       qualityLevel=quality_level,
                                       minDistance=min_distance)
    if features is not None:
        features = np.squeeze(features)
        if features.ndim == 1:  # Ensure a 2D array even if only one point is detected.
            features = features.reshape(1, -1)
        return features
    else:
        return np.empty((0, 2), dtype=np.float32)

def track_features_optical_flow(prev_img, curr_img, prev_points):
    """
    Track feature points from the previous image to the current image using Lucas–Kanade optical flow.
    
    :param prev_img: Previous grayscale image.
    :param curr_img: Current grayscale image.
    :param prev_points: Array of points (N, 2) in the previous image.
    :return: Tuple (curr_points, status) where:
             - curr_points is an array of new point locations.
             - status is a boolean array indicating which points were successfully tracked.
    """
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_img, curr_img,
                                                       prev_points.astype(np.float32),
                                                       None, **lk_params)
    if curr_points is None:
        return np.empty_like(prev_points), np.zeros(prev_points.shape[0], dtype=bool)
    
    curr_points = np.squeeze(curr_points)
    status = np.squeeze(status).astype(bool)
    return curr_points, status

def track_features_measurements(left_imgs, right_imgs, min_features_threshold=100, max_corners=200):
    """
    Process the left and right image sequences to track features over time and produce a measurement array z
    of shape (4, total_features, N), where N is the number of frames. For each feature, at each time t, z[:,i,t]
    is:
        [l_x, l_y, r_x, r_y]
    If the feature is not detected at time t, the measurement is [-1, -1, -1, -1].
    
    Dynamic re-detection is performed when the number of actively tracked features falls below a threshold.
    
    :param left_imgs: List of left camera grayscale images.
    :param right_imgs: List of right camera grayscale images.
    :param min_features_threshold: Minimum number of active features to maintain.
    :param max_corners: Maximum number of features to detect when re-detecting.
    :return: A NumPy array of shape (4, total_features, N).
    """
    N = len(left_imgs)
    # Dictionary mapping feature id -> measurement list (length N, each measurement is a 4-element list)
    measurements = {}
    # Active features: mapping feature id -> current left point (np.array of shape (2,))
    active_features = {}
    next_id = 0
    
    # Frame 0: detect features in the left image.
    features0 = detect_initial_features(left_imgs[0], max_corners=max_corners)
    # Perform stereo matching on frame 0.
    right0, status0 = track_features_optical_flow(left_imgs[0], right_imgs[0], features0)
    
    for i, pt in enumerate(features0):
        fid = next_id
        next_id += 1
        # Initialize measurement for all frames with [-1, -1, -1, -1]
        meas = [[-1, -1, -1, -1] for _ in range(N)]
        lx, ly = pt[0], pt[1]
        if status0[i]:
            rx, ry = right0[i]
        else:
            rx, ry = -1, -1
        meas[0] = [lx, ly, rx, ry]
        measurements[fid] = meas
        active_features[fid] = pt  # store left coordinate
    
    # Process frames 1 to N-1.
    for t in range(1, N):
        print("timestamp t: ", t)
        # List of active feature IDs.
        if active_features:
            active_ids = list(active_features.keys())
            pts_prev = np.array([active_features[fid] for fid in active_ids])
            pts_new, status = track_features_optical_flow(left_imgs[t-1], left_imgs[t], pts_prev)
            new_active = {}
            for j, fid in enumerate(active_ids):
                if status[j]:
                    new_pt = pts_new[j]
                    lx, ly = new_pt[0], new_pt[1]
                    # Perform stereo matching for current frame.
                    right_pt, status_right = track_features_optical_flow(left_imgs[t], right_imgs[t],
                                                      new_pt.reshape(1, 2))
                    right_pt = np.atleast_2d(right_pt)  # Ensure it has shape (1,2)
                    status_right = np.atleast_1d(status_right)  # Ensure it's an array
                    if status_right[0]:
                        rx, ry = right_pt[0, :]
                    else:
                        rx, ry = -1, -1
                    measurements[fid][t] = [lx, ly, rx, ry]
                    new_active[fid] = new_pt
                else:
                    # If tracking fails, measurement remains [-1,-1,-1,-1] for frame t.
                    measurements[fid][t] = [-1, -1, -1, -1]
                    # Feature is removed from active tracking.
            active_features = new_active
        
        # If the number of active features falls below the threshold, detect new features.
        if len(active_features) < min_features_threshold:
            new_feats = detect_initial_features(left_imgs[t], max_corners=max_corners)
            if new_feats.shape[0] > 0:
                # Filter out new features that are too close to any currently active feature.
                if active_features:
                    active_pts = np.array(list(active_features.values()))
                    filtered_new = []
                    for pt in new_feats:
                        if np.all(np.linalg.norm(active_pts - pt, axis=1) > 10):  # at least 10 pixels apart
                            filtered_new.append(pt)
                    if filtered_new:
                        new_feats = np.array(filtered_new)
                    else:
                        new_feats = np.empty((0, 2))
                # For each new detection, assign a new feature id.
                for pt in new_feats:
                    fid = next_id
                    next_id += 1
                    # Initialize measurement for frames 0 to t-1 as [-1,-1,-1,-1]
                    meas = [[-1, -1, -1, -1] for _ in range(N)]
                    lx, ly = pt[0], pt[1]
                    right_pt, status_right = track_features_optical_flow(left_imgs[t], right_imgs[t],
                                                      new_pt.reshape(1, 2))
                    right_pt = np.atleast_2d(right_pt)  # Ensure it has shape (1,2)
                    status_right = np.atleast_1d(status_right)  # Ensure it's an array
                    if status_right[0]:
                        rx, ry = right_pt[0, :]
                    else:
                        rx, ry = -1, -1
                    meas[t] = [lx, ly, rx, ry]
                    measurements[fid] = meas
                    active_features[fid] = pt
    
    total_features = next_id
    # Create an array of shape (N, 4, total_features) from the dictionary.
    meas_array = np.empty((N, 4, total_features), dtype=np.float32)
    for fid in range(total_features):
        meas_array[:, :, fid] = np.array(measurements[fid])
    # Rearrange to shape (4, total_features, N)
    meas_array = np.transpose(meas_array, (1, 2, 0))
    return meas_array

def create_video_from_measurements(left_imgs, right_imgs, meas_array, 
                                   left_video_path, right_video_path, fps=10):
    """
    Generate two MP4 videos (left and right) using the saved measurement array.
    
    The measurement array is expected to have shape (4, total_features, N), where:
      - The first two rows contain the left image coordinates,
      - The last two rows contain the right image coordinates,
      - N is the number of timestamps.
    
    If a feature is not detected at time t, its measurement remains [-1,-1,-1,-1].
    
    The video overlays the features as circles and draws lines connecting positions
    of the same feature from one frame to the next if both are valid.
    
    :param left_imgs: List of left grayscale images.
    :param right_imgs: List of right grayscale images.
    :param meas_array: Numpy array of shape (4, total_features, N) containing feature measurements.
    :param left_video_path: Output filename for the left camera video (e.g., "left_features.mp4").
    :param right_video_path: Output filename for the right camera video.
    :param fps: Frames per second for the output videos.
    """
    N = len(left_imgs)
    total_features = meas_array.shape[1]
    height, width = left_imgs[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    left_writer = cv2.VideoWriter(left_video_path, fourcc, fps, (width, height))
    right_writer = cv2.VideoWriter(right_video_path, fourcc, fps, (width, height))
    
    # To store the previous valid coordinate for each feature
    prev_left_coords = [None] * total_features
    prev_right_coords = [None] * total_features
    
    for t in range(N):
        # Convert current images to color for drawing.
        left_frame = cv2.cvtColor(left_imgs[t], cv2.COLOR_GRAY2BGR)
        right_frame = cv2.cvtColor(right_imgs[t], cv2.COLOR_GRAY2BGR)
        
        # Loop over each feature.
        for i in range(total_features):
            lx = meas_array[0, i, t]
            ly = meas_array[1, i, t]
            rx = meas_array[2, i, t]
            ry = meas_array[3, i, t]
            
            # Process left image measurements.
            if lx != -1 and ly != -1:
                pt_left = (int(lx), int(ly))
                cv2.circle(left_frame, pt_left, 3, (0, 255, 0), -1)
                if prev_left_coords[i] is not None:
                    cv2.line(left_frame, prev_left_coords[i], pt_left, (255, 0, 0), 1)
                prev_left_coords[i] = pt_left
            else:
                prev_left_coords[i] = None  # Reset if feature not detected.
            
            # Process right image measurements.
            if rx != -1 and ry != -1:
                pt_right = (int(rx), int(ry))
                cv2.circle(right_frame, pt_right, 3, (0, 255, 0), -1)
                if prev_right_coords[i] is not None:
                    cv2.line(right_frame, prev_right_coords[i], pt_right, (255, 0, 0), 1)
                prev_right_coords[i] = pt_right
            else:
                prev_right_coords[i] = None
        
        # Write frames to the video files.
        left_writer.write(left_frame)
        right_writer.write(right_frame)
    
    left_writer.release()
    right_writer.release()
    print("Videos saved as:")
    print(" Left:", left_video_path)
    print(" Right:", right_video_path)


def generate_overlaid_features_figure(left_imgs, right_imgs, meas_array, t=0, dt=15):
    """
    Produces a 2-row figure:
      - TOP ROW: Overlays the left and right images at time t in a single composite,
                  then draws:
                    * Left-camera features in BLUE
                    * Right-camera features in RED
                    * Green lines connecting the same feature (left <-> right)
      - BOTTOM ROW: Overlays the left image at time t and time t+dt in a single composite,
                    then draws:
                      * The time-t features in BLUE
                      * The time-(t+dt) features in RED
                      * Green lines connecting the same feature across time

    :param left_imgs:  list (or array) of grayscale left images.
    :param right_imgs: list (or array) of grayscale right images.
    :param meas_array: (4, total_features, N) feature measurements:
                        [ l_x, l_y, r_x, r_y ] in rows.
                       If a feature is missing, it is [-1,-1,-1,-1].
    :param t:          time index for the top row overlay.
    :param dt:         offset for the bottom row overlay (default=15).
    """

    # Total frames
    N = len(left_imgs)
    if t < 0 or t >= N:
        raise ValueError(f"Invalid time index t={t} (must be in [0, {N-1}]).")
    if t+dt >= N:
        print(f"Warning: t+dt={t+dt} exceeds total frames; adjusting dt.")
        dt = max(0, N-1 - t)

    # Convert the relevant frames to BGR for color drawing
    left_bgr_t = cv2.cvtColor(left_imgs[t], cv2.COLOR_GRAY2BGR)
    right_bgr_t = cv2.cvtColor(right_imgs[t], cv2.COLOR_GRAY2BGR)
    left_bgr_tdt = cv2.cvtColor(left_imgs[t+dt], cv2.COLOR_GRAY2BGR)

    # 1) TOP ROW: Overlaid left & right images at time t
    #    We'll do a simple alpha-blend so that both are visible in one composite.
    top_overlay = cv2.addWeighted(left_bgr_t, 0.5, right_bgr_t, 0.5, 0)

    # Draw the features for the top row
    total_features = meas_array.shape[1]
    for i in range(total_features):
        lx, ly = meas_array[0, i, t], meas_array[1, i, t]
        rx, ry = meas_array[2, i, t], meas_array[3, i, t]
        if lx != -1 and ly != -1 and rx != -1 and ry != -1:
            # Draw the left feature in BLUE
            cv2.circle(top_overlay, (int(lx), int(ly)), 4, (255, 0, 0), -1)
            # Draw the right feature in RED
            cv2.circle(top_overlay, (int(rx), int(ry)), 4, (0, 0, 255), -1)
            # Draw a green line between them
            cv2.line(top_overlay, (int(lx), int(ly)), (int(rx), int(ry)), (0, 255, 0), 1)

    # 2) BOTTOM ROW: Overlaid left images at time t and time t+dt
    bottom_overlay = cv2.addWeighted(left_bgr_t, 0.5, left_bgr_tdt, 0.5, 0)

    # Draw the features for the bottom row
    for i in range(total_features):
        lx_t, ly_t = meas_array[0, i, t],   meas_array[1, i, t]
        lx_dt, ly_dt = meas_array[0, i, t+dt], meas_array[1, i, t+dt]
        if lx_t != -1 and ly_t != -1 and lx_dt != -1 and ly_dt != -1:
            # Feature at time t in BLUE
            cv2.circle(bottom_overlay, (int(lx_t), int(ly_t)), 4, (255, 0, 0), -1)
            # Same feature at time t+dt in RED
            cv2.circle(bottom_overlay, (int(lx_dt), int(ly_dt)), 4, (0, 0, 255), -1)
            # Connect them with a green line
            cv2.line(bottom_overlay, (int(lx_t), int(ly_t)),
                                     (int(lx_dt), int(ly_dt)), (0, 255, 0), 1)

    # Convert BGR -> RGB for matplotlib
    top_rgb = cv2.cvtColor(top_overlay, cv2.COLOR_BGR2RGB)
    bottom_rgb = cv2.cvtColor(bottom_overlay, cv2.COLOR_BGR2RGB)

    # Plot in a 2x1 figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].imshow(top_rgb)
    axs[0].set_title(f"Overlay of Left vs Right at time t={t}")
    axs[0].axis('off')

    axs[1].imshow(bottom_rgb)
    axs[1].set_title(f"Overlay of Left(t={t}) vs Left(t+{dt}={t+dt})")
    axs[1].axis('off')
    plt.savefig('./results/feature_detection.png')
    plt.tight_layout()
    plt.show()


def main():
    # # Load dataset02 images from the .npy file.
    # imgs_dict = np.load("../data/dataset02/dataset02_imgs.npy", allow_pickle=True).item()
    # left_imgs = list(imgs_dict['cam_imgs_L'])
    # right_imgs = list(imgs_dict['cam_imgs_R'])
    
    # print("Loaded", len(left_imgs), "left images and", len(right_imgs), "right images.")
    # print("First left image shape:", left_imgs[0].shape)
    
    # # Run the tracking to produce a measurement array of shape (4, total_features, N)
    # measurement_array = track_features_measurements(left_imgs, right_imgs,
    #                                                  min_features_threshold=100,
    #                                                  max_corners=200)
    # print("Measurement array shape:", measurement_array.shape)
    
    # # Save the measurement array for later use.
    # np.save("feature_measurements.npy", measurement_array)

    # Code for generating video
    # Load your images dictionary.
    imgs_dict = np.load("../data/dataset02/dataset02_imgs.npy", allow_pickle=True).item()
    left_imgs = list(imgs_dict['cam_imgs_L'])
    right_imgs = list(imgs_dict['cam_imgs_R'])
    
    print("Loaded {} left images and {} right images.".format(len(left_imgs), len(right_imgs)))
    print("First left image shape:", left_imgs[0].shape)
    
    # Load the saved measurement array.
    meas_array = np.load("feature_measurements.npy")
    print("Measurement array shape:", meas_array.shape)
    
    generate_overlaid_features_figure(left_imgs, right_imgs, meas_array, t=400, dt=15)

    # # Generate videos from the measurements.
    # create_video_from_measurements(left_imgs, right_imgs, meas_array,
    #                                left_video_path="left_features.mp4",
    #                                right_video_path="right_features.mp4",
    #                                fps=10)
    
if __name__ == '__main__':
    main()
