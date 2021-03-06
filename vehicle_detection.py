import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from functools import partial
import glob
import time
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage.measurements import label

orient = 9
pix_per_cell = 8
cell_per_block = 2


# From "35. Hog Sub-sampling Window Search"
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


# From "35. Hog Sub-sampling Window Search"
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


# From "35. Hog Sub-sampling Window Search"
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


# From "35. Hog Sub-sampling Window Search"
def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# From "34. Search and Classify"
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features =\
                    get_hog_features(feature_image[:, :, hog_channel], orient,
                                     pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list, MUST match feature set chosen in find_cars()
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# From "34. Search and Classify"
# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# From "35. Hog Sub-sampling Window Search"
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # each "box" takes the form ((x1, y1), (x2, y2))
    box_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction, MUST match feature set chosen in extract_features()
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                box = ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart))
                box_list.append(box)
    return box_list


# From "29. HOG Classify" with modifications
def predict_cars():
    colorspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"

    cars = glob.glob('vehicles/*/*.png')
    notcars = glob.glob('non-vehicles/**/*.png')
    t = time.time()
    car_features = extract_features(cars, color_space=colorspace, orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel)
    notcar_features = extract_features(notcars, color_space=colorspace, orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel)

    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # for further tuning possibilities, see "27. Parameter tuning"
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()

    # Optimize classifier hyperparameters, re-init classifier (takes about 2 minutes w 10 steps...)
    # np.logspace(-6, -1, 10) => C=0.0001668 => score=0.9923986
    # np.logspace(-6,  0, 20) => C=0.0001623 & tol=0.0001 => score=0.9919
    # np.logspace(-5,  1, 40) => C=0.0002424 & tol=0.0001 => score=0.9919
    # np.logspace(-5, -2, 40) => C=0.0001425 & tol=0.0001 => score=0.9923
    # np.logspace(-4, -3, 40) => C=0.0001914 or C=0.0001125 & tol=0.0001 => score=0.9925 or 0.993
    c_s = np.logspace(-4, -3, 40)
    tol_list = [0.0001, 0.00001]
    n_jobs = 4  # I have a quad-core :-)
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=c_s, tol=tol_list), n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    print('Classifier best score: ', clf.best_score_)
    print('Classifier best estimator: ', clf.best_estimator_)
    svc = clf.best_estimator_
    svc.fit(X_train, y_train)
    print('Test Accuracy of SVC, optimized = ', round(svc.score(X_test, y_test), 4))

    # Continue with optimized classifier
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to optimize and predict', n_predict, 'labels with SVC')
    return svc, X_scaler


# Based on "37. Multiple Detections & False Positives
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


# Based on "37. Multiple Detections & False Positives
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


# Based on "37. Multiple Detections & False Positives
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


# Process each video frame
def process_image_internal(original_img, svc, X_scaler, heat):
    ystart = 370
    ystop = 656
    spatial_size = (32, 32)
    hist_bins = 32
    all_scales_boxes = []
    for scale in [1.0, 1.5, 2, 3.5, 6]:
        box_list =\
            find_cars(original_img, ystart, ystop, scale, svc, X_scaler, orient,
                      pix_per_cell, cell_per_block, spatial_size, hist_bins)
        all_scales_boxes.extend(box_list)

    # Heatmap processing from "37. Multiple Detections & False Positives"
    add_heat(heat, all_scales_boxes)
    heat = apply_threshold(heat, 2.5)
    heatmap = np.clip(heat, 0, 255)
    # Simple moving-average
    heat *= 0.8

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(original_img), labels)

    return draw_img, heatmap


# Process each video frame
def process_image(original_img, svc, X_scaler, heat):
    draw_img, heatmap = process_image_internal(original_img, svc, X_scaler, heat)
    return draw_img


def create_heatmap(shape):
    heat = np.zeros(shape).astype(np.float)
    return heat


# For each video, process each frame and output the resulting video
def process_video(input, output, process_image_fun):
    clip = VideoFileClip(input)
    # Uncomment to generate small quick range, range in seconds (start, end)
    # clip = clip.subclip(25, 29)
    heatmap = create_heatmap((clip.h, clip.w))
    process_image_fun = partial(process_image_fun, heat=heatmap)

    vid_clip = clip.fl_image(process_image_fun)
    vid_clip.write_videofile(output, audio=False)
    return


# Main, process the videos
def vehicle_detection_main():
    svc, X_scaler = predict_cars()

    part_process_image = partial(process_image, svc=svc, X_scaler=X_scaler)
    process_video('test_video.mp4', 'output_images/test_video_out.mp4', part_process_image)

    part_process_image = partial(process_image, svc=svc, X_scaler=X_scaler)
    process_video('project_video.mp4', 'output_images/project_video_out.mp4', part_process_image)


if __name__ == "__main__":
    vehicle_detection_main()
