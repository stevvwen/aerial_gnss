import os
from tqdm import tqdm

# Utility of this file: Remove the image and label files that 
# contain invalid data augmentations ( negative values or values greater than 1)


source_image_dir= ...
source_label_dir= ...

source_image_dir= os.path.abspath(source_image_dir)
source_label_dir= os.path.abspath(source_label_dir)


# Get all image files in the source directory and the corresponding label files
image_files= os.listdir(source_image_dir)
label_files= os.listdir(source_label_dir)

# sort the files
image_files.sort()
label_files.sort()


for image_file, label_file in tqdm(zip(image_files, label_files)):
    
    # Check if the label files contains negative values or values greater than 1
    with open(os.path.join(source_label_dir, label_file), "r") as f:
        for line in f:
            class_id, x_center, y_center, width, height= map(float, line.strip().split())
            if x_center<0 or x_center>1 or y_center<0 or y_center>1 or width<0 or width>1 or height<0 or height>1:
                # remove the image and label files
                os.remove(os.path.join(source_image_dir, image_file))
                os.remove(os.path.join(source_label_dir, label_file))

                break
                
