from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load your custom YOLOv8 models
models = {
    'nano_50': YOLO('E:/facultate/an3/sem2/Proiect_Yolov8/runs/detect/train_nano_50/weights/best.pt'),
    'nano_300': YOLO('E:/facultate/an3/sem2/Proiect_Yolov8/runs/detect/train_nano_300/weights/best.pt'),
    'nano_300_imgsz320': YOLO('E:/facultate/an3/sem2/Proiect_Yolov8/runs/detect/train_nano_300_imgsz320/weights/best.pt'),
    'medium_50': YOLO('E:/facultate/an3/sem2/Proiect_Yolov8/runs/detect/train_medium_50/weights/best.pt'),
    'medium_300': YOLO('E:/facultate/an3/sem2/Proiect_Yolov8/runs/detect/train_medium_300/weights/best.pt'),
    'medium_300_imgsz320': YOLO('E:/facultate/an3/sem2/Proiect_Yolov8/runs/detect/train_medium_300_imgsz320/weights/best.pt'),
    'xlarge_50': YOLO('E:/facultate/an3/sem2/Proiect_Yolov8/runs/detect/train_xlarge_50/weights/best.pt'),
    'xlarge_300': YOLO('E:/facultate/an3/sem2/Proiect_Yolov8/runs/detect/train_xlarge_300/weights/best.pt'),
    'xlarge_300_imgsz320': YOLO('E:/facultate/an3/sem2/Proiect_Yolov8/runs/detect/train_xlarge_300_imgsz320/weights/best.pt')
}

# Load the image
image_path = 'E:/facultate/an3/sem2/Proiect_Yolov8/Processed_BraTS2020/images/test/BraTS20_Training_115_flair_90.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB

# Create a figure to plot the results
fig, axs = plt.subplots(3, 3, figsize=(5, 10))

# Iterate through models and perform inference
for ax, (model_name, model) in zip(axs.flat, models.items()):
    # Perform inference
    results = model(image_rgb)
    
    # Assuming results are a list, get the first item
    result = results[0]
    
    # Get the annotated image
    annotated_image = result.plot()
    
    # Display the image using matplotlib
    ax.imshow(annotated_image)
    ax.set_title(model_name)
    ax.axis('off')

plt.tight_layout()
plt.show()
