import cv2
import numpy as np
import vehicle_detection as vd
import glob
import matplotlib.image as mpimg


# Inputs
cars = glob.glob('vehicles/*/*.png')
notcars = glob.glob('non-vehicles/**/*.png')

# Outputs, should all go into output_images directory
sample_vehicle_f_name = 'output_images/sample_vehicle.jpg'
sample_non_vehicle_f_name = 'output_images/sample_non_vehicle.jpg'


def main():
    # Sample "random" (non)-vehicle images
    vehicle_image = mpimg.imread(cars[42])
    non_vehicle_image = mpimg.imread(notcars[42])
    mpimg.imsave(sample_vehicle_f_name, vehicle_image)
    mpimg.imsave(sample_non_vehicle_f_name, non_vehicle_image)



main()
