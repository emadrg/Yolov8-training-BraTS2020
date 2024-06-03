import nibabel as nib
import numpy as np
import cv2
import os
import shutil
import random

# Directories for the original train and validation data
data_directory_train = "E:/facultate/an3/sem2/Proiect/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# New base directories for processed images and labels
base_processed_dir = 'E:/facultate/an3/sem2/Proiect_Yolov8/Processed_BraTS2020'
output_images_train = os.path.join(base_processed_dir, 'images/train')
output_labels_train = os.path.join(base_processed_dir, 'labels/train')


# Function to explore the BraTS data and find the paths of the images for each patient
def explore_brats_data(directory):
    patients_data = {} # Empty dictionary to store the paths of the images for each patient
    for root, dirs, files in os.walk(directory): # Walk through the directory
        patient_id = None # Initialize the patient ID
        for file in files: 
            if file.endswith(".nii"): 
                if patient_id is None: 
                    patient_id = root.split(os.path.sep)[-1] # Get the patient ID from the directory name
                if patient_id not in patients_data:
                    patients_data[patient_id] = {'FLAIR': None, 'T1': None, 'T1ce': None, 'T2': None, 'Seg': None} # Initialize the dictionary for the patient
                if 'flair' in file.lower():
                    patients_data[patient_id]['FLAIR'] = os.path.join(root, file) # Store the path of the FLAIR image
                elif 't1ce' in file.lower():
                    patients_data[patient_id]['T1ce'] = os.path.join(root, file) # Store the path of the T1ce image
                elif 't1' in file.lower() and 't1ce' not in file.lower():
                    patients_data[patient_id]['T1'] = os.path.join(root, file) # Store the path of the T1 image
                elif 't2' in file.lower():
                    patients_data[patient_id]['T2'] = os.path.join(root, file) # Store the path of the T2 image
                elif 'seg' in file.lower():
                    patients_data[patient_id]['Seg'] = os.path.join(root, file) # Store the path of the Segmentation mask
        if patient_id and patient_id in patients_data:
            print("Files found for patient", patient_id, patients_data[patient_id]) # Print the paths of the images for the patient
        else:
            print("No files found or patient ID not set for directory:", root)
    return patients_data


# Save a slice of a NIfTI file as a JPEG image
def save_as_image(nifti_file, output_folder, slice_index=90):
    img = nib.load(nifti_file) # Load the NIfTI file
    data = img.get_fdata() # Get the image data
    slice = data[:, :, slice_index] # Get the slice
    normalized_slice = cv2.normalize(slice, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) 
    normalized_slice = np.uint8(normalized_slice) # Normalize the slice and convert to 8-bit unsigned integer
    base_filename = os.path.basename(nifti_file).replace('.nii', f'_{slice_index}') # Get the base filename
    image_path = os.path.join(output_folder, base_filename + '.jpg') # Create the path for the output image
    cv2.imwrite(image_path, normalized_slice)
    print(f"Saved image {image_path}")
    return image_path, base_filename


# Convert a mask to a bounding box
def mask_to_bbox(mask, image_width, image_height):
    x = np.any(mask, axis=0) # Check if there are any non-zero values along the x-axis
    y = np.any(mask, axis=1) # Check if there are any non-zero values along the y-axis
    if not np.any(x) or not np.any(y):
        return None
    x_min, x_max = np.where(x)[0][[0, -1]] # Get the minimum and maximum x values
    y_min, y_max = np.where(y)[0][[0, -1]] # Get the minimum and maximum y values
    x_center = ((x_min + x_max) / 2) / image_width # Calculate the x center
    y_center = ((y_min + y_max) / 2) / image_height # Calculate the y center
    bbox_width = (x_max - x_min) / image_width # Calculate the bounding box width
    bbox_height = (y_max - y_min) / image_height # Calculate the bounding box height
    return (x_center, y_center, bbox_width, bbox_height) # Return the bounding box coordinates


def process_patient(patient_data, images_folder, labels_folder=None):
    if 'FLAIR' not in patient_data:
        print(f"FLAIR data missing for patient, skipping processing. Data found: {list(patient_data.keys())}")
        return
    flair_image = patient_data['FLAIR'] # Get the path of the FLAIR image 
    print(f"Processing FLAIR image {flair_image}")
    image_path, base_filename = save_as_image(flair_image, images_folder) # Save the FLAIR image as a JPEG
    if labels_folder and 'Seg' in patient_data and patient_data['Seg']: # Check if the labels folder is provided and the Segmentation mask is available
        seg_image = patient_data['Seg'] # Get the path of the Segmentation mask	
        seg = nib.load(seg_image).get_fdata() # Load the Segmentation mask
        bbox = mask_to_bbox(seg[:, :, 90], seg.shape[0], seg.shape[1]) # Convert the mask to a bounding box
        label_path = os.path.join(labels_folder, base_filename + '.txt') # Create the path for the label file
        with open(label_path, 'w') as f: 
            if bbox:
                x_center, y_center, bbox_width, bbox_height = bbox
                f.write(f'0 {x_center} {y_center} {bbox_width} {bbox_height}\n')
           

# Process all patients and create labels
def process_all_patients(data_directory, images_folder, labels_folder=None): 
    os.makedirs(images_folder, exist_ok=True) # Create the images folder if it does not exist
    if labels_folder:
        os.makedirs(labels_folder, exist_ok=True) # Create the labels folder if it does not exist
    patient_files = explore_brats_data(data_directory) # Explore the BraTS data and get the paths of the images for each patient
    for patient_id, patient_data in patient_files.items(): # Iterate through the patients
        process_patient(patient_data, images_folder, labels_folder) # Process the patient


# Organize images and labels by moving them to their respective directories
def organize_files(src_dir, dst_dir): 
    if not os.path.exists(dst_dir): # Create the destination directory if it does not exist
        os.makedirs(dst_dir)
    for root, dirs, files in os.walk(src_dir): # Walk through the source directory
        for file in files:
            if file.endswith('.jpg') or (file.endswith('.txt') and 'labels' in dst_dir): # Check if the file is an image or a label
                shutil.move(os.path.join(root, file), os.path.join(dst_dir, file)) # Move the file to the destination directory
                print(f"Moved {file} to {dst_dir}")


def split_data(base_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    image_dir = os.path.join(base_dir, 'images/train')
    label_dir = os.path.join(base_dir, 'labels/train')

    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    random.shuffle(images)

    train_split = int(len(images) * train_ratio) 
    val_split = int(len(images) * (train_ratio + val_ratio)) 

    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]

    for image_set, folder_name in zip([val_images, test_images], ['val', 'test']):
        for image in image_set:
            shutil.move(os.path.join(image_dir, image), os.path.join(base_dir, f'images/{folder_name}', image))
            label_name = image.replace('.jpg', '.txt')
            shutil.move(os.path.join(label_dir, label_name), os.path.join(base_dir, f'labels/{folder_name}', label_name))
            print(f"Moved {image} and {label_name} to {folder_name}")


# Process all patients and create labels
process_all_patients(data_directory_train, output_images_train, output_labels_train)

# Organize images and labels by moving them to their respective directories
organize_files(output_images_train, os.path.join(base_processed_dir, 'images/train'))
organize_files(output_labels_train, os.path.join(base_processed_dir, 'labels/train'))

# Split the data into training, validation, and test sets
split_data(base_processed_dir)


# Train the model

from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # build a new model from scratch

results = model.train(data="coco8.yaml", epochs=5)  # train the model

#Implement grad-cam for yolov8