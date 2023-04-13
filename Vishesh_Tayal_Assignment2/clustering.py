import csv
import os
import shutil
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from skimage import color
from skimage.measure import label, regionprops, regionprops_table


def slicing_clustering(image, slices_path, boundaries_path, image_index):
    # Converting thresh image to grayscale and finding region props and labels using skimage module
    img = cv2.imread(image)
    g_im = color.rgb2gray(img)
    b_im = g_im < 1
    labels = label(b_im)
    regions = regionprops(labels)

    # Finding 'R' points by checking the mode value of the below DF
    df = pd.DataFrame(regionprops_table(labels, img,
                                        properties=["area", "convex_area", "bbox_area",
                                                    "centroid", "orientation"]))
    # print(df.mode())

    # Using df.mode(), we saw that 'R' has a value of 2, using 2 to find the coordinates of 'R'
    bbox_array = []
    masks_array = []
    indexes = []

    for iterator, x in enumerate(regions):
        area = x.area
        convex_area = x.convex_area
        if area == 2:
            masks_array.append(regions[iterator].convex_image)
            bbox_array.append(regions[iterator].bbox)
            indexes.append(iterator)

    # count = len(masks_array)
    # print(count)

    centre_points = []
    for points in bbox_array:
        centre_points.append([(points[0] + points[2]) / 2, (points[1] + points[3]) / 2])
    # print(centre_points)

    # Uncomment below code to see the locations of 'R'
    # plt.imshow(img)
    # r_locations = np.array(centre_points)
    # print(r_locations)
    # plt.scatter(r_locations[:, 1], r_locations[:, 0],
    # marker = "P", color = "red", s=1)

    # Using the below code snippet to find the distance between 'R' points and slice the small square areas
    x_coordinates = {}
    for x, y in centre_points:
        if x not in x_coordinates:
            # print(x)
            x_coordinates[x] = [y]
        else:
            x_coordinates[x].append(y)
            # print(x_coordinates)
    # print(x_coordinates)

    x_dict = {}
    for key, values in x_coordinates.items():
        x_total = 0
        for i in range(1, len(values)):
            x_total += values[i] - values[i - 1]
            # print(x_total)
        if len(values) <= 1:
            continue
        x_dict[key] = x_total / (len(values) - 1)
        # print(len(values))
    # print(x_dict)

    y_coordinates = {}
    for x, y in centre_points:
        if y not in y_coordinates:
            y_coordinates[y] = [x]
        else:
            y_coordinates[y].append(x)

    y_dict = {}
    for key, values in y_coordinates.items():
        total_y = 0
        for i in range(1, len(values)):
            total_y += values[i] - values[i - 1]
        if len(values) <= 1:
            continue
        y_dict[key] = total_y / (len(values) - 1)
    # print(y_dict)

    x_sum = 0
    y_sum = 0
    for values in x_dict.values():
        x_sum = x_sum + values
    for values in y_dict.values():
        y_sum = y_sum + values

    # Found the length and breadth of the bounding boxes
    x_box = x_sum / len(x_dict.values())
    y_box = y_sum / len(y_dict.values())

    # Performing slicing on the thresh images and then applying contouring algorithms on each slice
    dir_name = os.path.join(slices_path, str(image_index))

    slice_counter = 1
    cluster_count = []

    for y in range(138, 728, int(x_box)):
        for x in range(3, 947, int(y_box)):
            tiles = img[y + 2:y + 116, x + 2:x + 116]  # Cropping small slices recursively
            gray_version = cv2.cvtColor(tiles, cv2.COLOR_BGR2GRAY)  # Ignoring all black images
            count = cv2.countNonZero(gray_version)
            if count != 0:
                cv2.imwrite(dir_name + "/" + "Slice" + "_" + str(slice_counter) + ".png", tiles)
                cropped_image_copy = tiles.copy()

                # Performing filtering on the basis of pixel value and applying DBSCAN algo
                hsv_image = cv2.cvtColor(cropped_image_copy, cv2.COLOR_BGR2HSV)
                saturation = hsv_image[:, :, 1] > 90
                mask = saturation
                red = cropped_image_copy[:, :, 0] * mask
                green = cropped_image_copy[:, :, 1] * mask
                blue = cropped_image_copy[:, :, 2] * mask
                masked_image = np.dstack((red, green, blue))
                masked_gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
                masked_copy = masked_image.copy()
                img_yellow_array = np.any(masked_image != [0, 0, 0], axis=-1)
                masked_copy[img_yellow_array] = [0, 255, 255]

                cluster_dir_name = os.path.join(boundaries_path, str(image_index))
                cv2.imwrite(cluster_dir_name + '/' + 'Cluster_Image' + '_' + str(slice_counter) + ".png", masked_copy)

                pixel_points = []

                pixel_count = np.array([])
                for x_pixel in range(masked_gray.shape[0]):
                    for y_pixel in range(masked_gray.shape[1]):
                        if masked_gray[x_pixel][y_pixel] > 0:
                            pixel_points.append([y_pixel, x_pixel])

                if len(pixel_points) > 0:
                    db_image = DBSCAN(eps=2, min_samples=5, metric='euclidean', algorithm='auto')
                    db_image.fit(pixel_points)
                    labels = db_image.labels_
                    unique_count, pixel_count = np.unique(labels, return_counts=True)

                    cluster_count.append([str(slice_counter), len(pixel_count[pixel_count > 135])])

                ds = pd.DataFrame(cluster_count, columns=['SliceNumber', 'ClusterCount'])
                ds.to_csv(os.path.join(cluster_dir_name, 'cluster_count.csv'), index=False)

                slice_counter = slice_counter + 1
            else:
                continue

# Implemented a function to create required directories and overwrite them
# for multiple iterations of code


def create_dir(index, path):
    if os.path.exists(os.path.join(path, str(index))):
        shutil.rmtree(os.path.join(path, str(index)))
        os.mkdir(os.path.join(path, str(index)))
    else:
        os.mkdir(os.path.join(path, str(index)))
