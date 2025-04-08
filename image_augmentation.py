import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import os
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from tqdm import tqdm


ia.random.seed(205)

# Define the augmentation pipeline
augment = iaa.Sequential([

    # horizontal flips with 50% probability
    iaa.Fliplr(0.4),
    # vertical flips with 50% probability
    iaa.Flipud(0.5),
    # crop and pad images
    iaa.Sometimes(0.6, iaa.CropAndPad(percent=(-0.15, 0.15))),
    # apply gaussian blur to the images
    iaa.Sometimes(0.6, iaa.GaussianBlur(sigma=(0.0, 1))),
    # apply motion blur to the images
    iaa.Sometimes(0.6, iaa.imgcorruptlike.MotionBlur(severity=1)),
    # apply fog to the images
    iaa.Sometimes(0.6, iaa.imgcorruptlike.GlassBlur(severity=2)),
    # apply brightness changes to the images
    iaa.Sometimes(0.6, iaa.imgcorruptlike.Brightness(severity=2)),
    # apply color temperature changes to the images
    iaa.Sometimes(0.6, iaa.ChangeColorTemperature((4000, 20000))),
    # apply affine transformations to the images
    iaa.Sometimes(0.6, iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        rotate=(-15, 15),
        shear=(-8, 8)
    )),

], random_order=True) # apply augmenters in random order

# Create output directory if it does not exist
output_dir = "augmented_images1"
train_dir= output_dir+"/train"
image_dir= train_dir+"/images"
label_dir= train_dir+"/labels"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    os.mkdir(train_dir)
    os.mkdir(image_dir)
    os.mkdir(label_dir)

source_image_dir= "datasets/multiaqua_data/train/images"
source_label_dir= "datasets/multiaqua_data/train/labels"

# Get all image files in the source directory and the corresponding label files
image_files= os.listdir(source_image_dir)
label_files= os.listdir(source_label_dir)

# sort the files
image_files.sort()
label_files.sort()

image_files= image_files[650:]
label_files= label_files[650:]

for image_file, label_file in tqdm(zip(image_files, label_files)):
    image_path= os.path.join(source_image_dir, image_file)
    label_path= os.path.join(source_label_dir, label_file)

    # Read the image and label files
    image= cv2.imread(image_path)

    # Get the image width and height
    image_height, image_width= image.shape[:2]
    
    # Multiple labels can be present in the label file
    with open(label_path, "r") as f:
        labels= []
        class_ids= []
        for line in f:
            
            # Get the class_id, x center, y center, widht and height of the bounding box
            class_id, x_center, y_center, width, height= map(float, line.strip().split())
            class_ids.append(class_id)

            # Convert the x_center, y_center, width and height to the top left and bottom right coordinates
            x_center= x_center*image_width
            y_center= y_center*image_height
            width= width*image_width
            height= height*image_height

            x_min= int(x_center-width/2)
            y_min= int(y_center-height/2)
            x_max= int(x_center+width/2)
            y_max= int(y_center+height/2)

            # Append the label to the labels list as a bouding box
            labels.append(BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max))

        boxes= BoundingBoxesOnImage(labels, shape
                                    =(image_height, image_width))
        
                
    image_file= ".".join(image_file.split(".")[:-1])
    label_file= ".".join(label_file.split(".")[:-1])

    
    # Apply 10 augmentations to each image
    for i in range(10):
        # Apply augmentation to the image
        augmented_image, augmented_boxes= augment(image=image, bounding_boxes=boxes)


        #augmented_image = augmented_boxes.draw_on_image(augmented_image, size=2, color=[0, 0, 255])

        # Write the augmented image to the output directory
        cv2.imwrite(f"{image_dir}/{image_file}_{i}.jpg", augmented_image)

        # Write the label file to the output directory
        with open(f"{label_dir}/{label_file}_{i}.txt", "w") as f:
            if len(augmented_boxes)==0:
                f.write("")
            else:
                for idx, box in enumerate(augmented_boxes):
                    x_min, y_min, x_max, y_max= box.x1, box.y1, box.x2, box.y2

                    # Convert the top left and bottom right coordinates to x_center, y_center, width and height
                    x_center= (x_min+x_max)/2
                    y_center= (y_min+y_max)/2
                    width= x_max-x_min
                    height= y_max-y_min

                    # Write the label to the file
                    f.write(f"{class_ids[idx]} {x_center/image_width} {y_center/image_height} {width/image_width} {height/image_height}\n")
