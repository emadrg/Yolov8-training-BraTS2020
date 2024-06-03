# Training YOLOv8 on BraTS2020 Data

In this repo, I'll present the process of training a YOLOv8 model using the BraTS2020 dataset, focusing on comparing the nano, medium and extra large version.

## Training Set

I used the BraTS2020 dataset for this project, which you can download [here](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation). This dataset includes MRI scans of brains from different patients. I only used the training set (BraTS2020_TrainingData), as the validation set didn't have any tumor masks.

## Understanding the Training Set

The data is formatted into 369 directories, one for each patient. Each patient directory contains multiple MRI scans of their brain, as well as a mask (the seg file) of the tumor. I decided to only use the "flair" images because they have the best contrast, making the tumor most visible. Therefore, for training our YOLOv8 model, we will only use the "flair" and the "seg" files. You can use another type of image instead of "flair," but make sure to use the same type for all patients to maintain consistent training.

## Converting the Raw Data into the YOLOv8 Format

The YOLOv8 algorithm works with images and bounding boxes. A bounding box is a rectangle that outlines the object we want to detect from an image. In our case, the bounding box would contain our tumor. The bounding boxes can be created in multiple ways, but I took advantage of the masks provided for the tumors and created the bounding boxes based on them. Usually, we won't have masks, only the images for training, so in such cases, bounding boxes are created manually (you can use [CVAT](https://www.cvat.ai/)). Also, have a look at [Open Images](https://storage.googleapis.com/openimages/web/index.html) for lots of YOLO-friendly datasets.

### Steps for Conversion:

1. Convert all 369 .nii flair images to .jpg as YOLOv8 cannot understand .nii files.
2. Create bounding box files based on the masks. I created the bounding boxes by checking whether the pixels in the mask file were white.

## Formatting the Data

This step is crucial since the YOLOv8 algorithm expects a specific structure for the training data. You need a directory with all your data (I named mine "Processed_BraTS2020") containing two directories: "images" and "labels." Each of these directories should have "train," "val," and optionally "test" subdirectories. You cannot change the images/labels structure, or else the algorithm won't run.

## Running the Algorithm

After formatting your data, you can start the training process. You will need a .yaml file (mine is named coco8.yaml). This file should have a specific structure where the path to the data and detection classes are specified. Modify this to fit your dataset but don't change the structure. Then, run these lines:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # Build a new model from scratch

results = model.train(data="coco8.yaml", epochs=50)  # Train the model
```

In this example, I'm training the nano version for 50 epochs. Read more about the YOLOv8 types of algorithms [here](https://docs.ultralytics.com/models/yolov8/#performance-metrics).

## Mistakes I Made in training the model

1. **Not Formatting the Data Correctly**:
   - Ensure that your data is in the desired YOLO format. This means having two directories named "images" and "labels" (you cannot choose any other names). These two directories should have "train" and "val" folders (you can name these subfolders however you want, but it is advisable to use appropriate names).

2. **Ensure Images and Labels Have the Same Name**:
   - This mistake took me a while to figure out. The images and labels should have exactly the same name but different file extensions so the algorithm knows which label corresponds to which image. If you're getting an error like "No labels detected," double-check this.

3. **Not Having Correct Label Files**:
   - The labels should have a specific format. If you're creating your labels manually, exporting the labels will be done correctly. However, I created my labels based on the tumor masks and had to format them myself using a Python function. One issue I encountered was for brain images without any tumors, regarded by YOLO as "backgrounds." If you want to include images without the objects you're detecting, leave their corresponding label files completely empty. Initially, I used all 0 values for these labels (0 0.0 0.0 0.0 0.0), and the algorithm considered these as actual parameters for the tumors. To solve this, either leave out the images without the object or leave the label files empty. Including or excluding backgrounds may lead to different model performances, so feel free to experiment.

4. **Not Using Enough Epochs**:
   - When I first ran my script, I only ran it for one epoch, resulting in no predictions for the validation data. Using too few epochs gives you shallow training, so behavior like this is expected. If you're encountering this problem, try running more epochs. Initially, I used 50, but since I wasn't pleased with the predictions on the validation set, I tried 100, then 300. Beware that using too many epochs may lead to overfitting, so the only real way to evaluate the model is on test data, completely unseen by the algorithm.
  
## Comparing different models

I trained the nano, medium and extra large models to see the difference between them and how the results would be influenced by both the model type, as well as the number of epochs used. For me, [Google Colab](https://docs.ultralytics.com/models/yolov8/#performance-metrics) was really helpful for running YOLO, since my laptop isn't very fast. I'm doing the comparison between the different models by looking at the [Performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/) resulted after the training process. In order to visualize everything better, I'll be storing the relevant data in a table. 


| Model       | Nr epochs | Accuracy | Avg Precision | Avg Recall |
|-------------|-----------|----------|---------------|------------|
| Nano        | 50        | 0.80     | 0.51          | 0.53       |
| Nano        | 100       | 0.78     | 0.72          | 0.57       |
| Nano        | 300       | 0.96     | 0.76          | 0.72       |
| Medium      | 50        | 0.80     | 0.44          | 0.44       |
| Medium      | 100       | 0.91     | 0.64          | 0.58       |
| Medium      | 300       | 0.91     | 0.71          | 0.64       |
| Extra Large | 50        | 0.69     | 0.44          | 0.41       |
| Extra Large | 100       | 0.82     | 0.52          | 0.50       |
| Extra Large | 300       | 0.89     | 0.71          | 0.65       |

Since my dataset is only 369 images and only 70% of them were used for training, I'll consider that I have a small dataset. Now let's take a look at the values in the table. 
