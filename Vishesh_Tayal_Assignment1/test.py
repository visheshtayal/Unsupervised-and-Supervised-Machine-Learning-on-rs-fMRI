from brainExtraction import *
import glob
import os

# Setting paths for the requirements
slices_path = "./Slices"
test_patient_path = "./testPatient"
boundaries_path = "./Boundaries"

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

if not os.path.exists(boundaries_path):
    os.mkdir(boundaries_path)
    print("Boundaries folder created")
else:
    print(boundaries_path + " already exists")

thresh_file_name_pattern = "*thresh.png"
thresh_images_list = glob.glob(os.path.join(test_patient_path, thresh_file_name_pattern))

# for iter in thresh_images_list:
#     print(iter)

# Iterating through the above list
for iterator, img in enumerate(thresh_images_list):
    # print(i, f)
    create_dir(iterator, slices_path)
    create_dir(iterator, boundaries_path)
    slicing_contouring(img, slices_path, boundaries_path, iterator)





