import cv2
import numpy as np
import vehicle_detection as vd
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Inputs
cars = glob.glob('vehicles/*/*.png')
notcars = glob.glob('non-vehicles/**/*.png')
test1_f_name = 'test_images/test1.jpg'
test3_f_name = 'test_images/test3.jpg'
test5_f_name = 'test_images/test5.jpg'
test6_f_name = 'test_images/test6.jpg'

# Outputs, should all go into output_images directory
sample_vehicle_f_name = 'output_images/sample_vehicle.jpg'
sample_non_vehicle_f_name = 'output_images/sample_non_vehicle.jpg'
hog_image_f_name = 'output_images/hog.jpg'
test1_out_f_name = 'output_images/test1_windows.jpg'
test3_out_f_name = 'output_images/test3_windows.jpg'
test5_out_f_name = 'output_images/test5_windows.jpg'
test6_out_f_name = 'output_images/test6_windows.jpg'
heatmap1_f_name = 'output_images/heatmap1.jpg'
heatmap3_f_name = 'output_images/heatmap3.jpg'
heatmap5_f_name = 'output_images/heatmap5.jpg'
heatmap6_f_name = 'output_images/heatmap6.jpg'

# Same constants as in vehicle_detection.py
orient = vd.orient
pix_per_cell = vd.pix_per_cell
cell_per_block = vd.cell_per_block
ystart = 370
ystop = 656
spatial_size = (32, 32)
hist_bins = 32


def sample_training_data():
    # Sample "random" (non)-vehicle images
    vehicle_image = mpimg.imread(cars[42])
    non_vehicle_image = mpimg.imread(notcars[42])
    mpimg.imsave(sample_vehicle_f_name, vehicle_image)
    mpimg.imsave(sample_non_vehicle_f_name, non_vehicle_image)
    return vehicle_image, non_vehicle_image


def hog_features(car, non_car):
    # Copying parts of vd.extract_features() to create an image with hog features etc
    car_cs_img = cv2.cvtColor(car, cv2.COLOR_RGB2YCrCb)
    ccs1 = car_cs_img[:, :, 0]
    ccs2 = car_cs_img[:, :, 1]
    ccs3 = car_cs_img[:, :, 2]

    non_car_cs_img = cv2.cvtColor(non_car, cv2.COLOR_RGB2YCrCb)
    nccs1 = non_car_cs_img[:, :, 0]
    nccs2 = non_car_cs_img[:, :, 1]
    nccs3 = non_car_cs_img[:, :, 2]

    cf1, hog_image1 = vd.get_hog_features(ccs1, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
    cf2, hog_image2 = vd.get_hog_features(ccs2, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
    cf3, hog_image3 = vd.get_hog_features(ccs3, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
    ncf1, nchog_image1 = vd.get_hog_features(nccs1, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
    ncf2, nchog_image2 = vd.get_hog_features(nccs2, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
    ncf3, nchog_image3 = vd.get_hog_features(nccs3, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)

    # Plotting code originally from "20. scikit-image HOG"
    fig = plt.figure(figsize=(8, 7))

    # Raw RGB images
    plt.subplot(441)
    plt.axis('off')
    plt.imshow(car)
    plt.title('Car')

    plt.subplot(443)
    plt.axis('off')
    plt.imshow(non_car)
    plt.title('not-Car')

    # Channel 1
    plt.subplot(445)
    plt.axis('off')
    plt.imshow(ccs1, cmap='gray')
    plt.title('Car CH-1')

    plt.subplot(446)
    plt.axis('off')
    plt.imshow(hog_image1, cmap='gray')
    plt.title('Car CH-1 Hog')

    plt.subplot(447)
    plt.axis('off')
    plt.imshow(non_car[:, :, 0], cmap='gray')
    plt.title('not-Car CH-1')

    plt.subplot(448)
    plt.axis('off')
    plt.imshow(nchog_image1, cmap='gray')
    plt.title('not-Car CH-1 Hog')

    # Channel 2
    plt.subplot(449)
    plt.axis('off')
    plt.imshow(ccs2, cmap='gray')
    plt.title('Car CH-2')

    plt.subplot(4, 4, 10)
    plt.axis('off')
    plt.imshow(hog_image2, cmap='gray')
    plt.title('Car CH-2 Hog')

    plt.subplot(4, 4, 11)
    plt.axis('off')
    plt.imshow(non_car[:, :, 1], cmap='gray')
    plt.title('not-Car CH-2')

    plt.subplot(4, 4, 12)
    plt.axis('off')
    plt.imshow(nchog_image2, cmap='gray')
    plt.title('not-Car CH-2 Hog')

    # Channel 3
    plt.subplot(4, 4, 13)
    plt.axis('off')
    plt.imshow(ccs3, cmap='gray')
    plt.title('Car CH-3')

    plt.subplot(4, 4, 14)
    plt.axis('off')
    plt.imshow(hog_image3, cmap='gray')
    plt.title('Car CH-3 Hog')

    plt.subplot(4, 4, 15)
    plt.axis('off')
    plt.imshow(non_car[:, :, 2], cmap='gray')
    plt.title('not-Car CH-3')

    plt.subplot(4, 4, 16)
    plt.axis('off')
    plt.imshow(nchog_image3, cmap='gray')
    plt.title('not-Car CH-1 Hog')

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

    fig.savefig(hog_image_f_name)


def sliding_window(input_f_name, output_f_name, heatmap_f_name, svc, X_scaler):
    image = mpimg.imread(input_f_name)
    heat = vd.create_heatmap(image.shape)
    # run several times to "warm up" the heat map, as if it was part of a video with similar frames before it
    for _ in range(10):
        draw_img, heatmap = vd.process_image_internal(image, svc, X_scaler, heat)
    mpimg.imsave(output_f_name, draw_img, format='jpg')
    mpimg.imsave(heatmap_f_name, heatmap, cmap=cm.Reds_r, format='jpg')
    print('Sliding window image generated: ' + output_f_name)


def sliding_window_all(svc, X_scaler):
    sliding_window(test1_f_name, test1_out_f_name, heatmap1_f_name, svc, X_scaler)
    sliding_window(test3_f_name, test3_out_f_name, heatmap3_f_name, svc, X_scaler)
    sliding_window(test5_f_name, test5_out_f_name, heatmap5_f_name, svc, X_scaler)
    sliding_window(test6_f_name, test6_out_f_name, heatmap6_f_name, svc, X_scaler)


def main():
    vehicle_image, non_vehicle_image = sample_training_data()

    print('HOG features...')
    hog_features(vehicle_image, non_vehicle_image)
    print('HOG features done!')

    print('Training...')
    svc, X_scaler = vd.predict_cars()
    print('Training done!')

    print('Sliding windows and heatmaps...')
    sliding_window_all(svc, X_scaler)
    print('Sliding windows done!')

main()
