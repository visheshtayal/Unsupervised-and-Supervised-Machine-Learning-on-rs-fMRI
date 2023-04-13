from clustering import *
import glob
import os

# Setting paths for the requirements
slices_path = "./Slices"
test_patient_path = "./testPatient"
cluster_path = "./Clusters"

# Checking if "testPatient" directory exists or not
if os.path.exists(test_patient_path):
    print("test_patient directory exists.")
else:
    print("test_patient directory not found!")

# Creating "Slices" and "Boundaries" folders, if they don't exist
if not os.path.exists(slices_path):
    os.mkdir(slices_path)
    print("Slices folder created")
else:
    print(slices_path + " already exists")

if not os.path.exists(cluster_path):
    os.mkdir(cluster_path)
    print("Clusters folder created")
else:
    print(cluster_path + " already exists")

thresh_file_name_pattern = "*thresh.png"
thresh_images_list = glob.glob(os.path.join(test_patient_path, thresh_file_name_pattern))

# for iter in thresh_images_list:
#     print(iter)

# Iterating through the above list
for iterator, img in enumerate(thresh_images_list):
    # print(i, f)
    create_dir(iterator, slices_path)
    create_dir(iterator, cluster_path)
    slicing_clustering(img, slices_path, cluster_path, iterator)





