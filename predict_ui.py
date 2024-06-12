import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def browse():
    image_path = askopenfilename(initialdir="/", title="Select file",
                                 filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    if image_path:
        process_image(image_path)

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB

    # Load the YOLO model
    model = YOLO('E:/facultate/an3/sem2/Proiect_Yolov8/runs/detect/train_medium_300/weights/best.pt')

    # Use the model to predict the image
    results = model(image_rgb)
    result = results[0]

    # Clear previous plot if exists
    for widget in canvas_frame.winfo_children():
        widget.destroy()

    # Add prediction label
    prediction_label = tk.Label(canvas_frame, text="Prediction:", font=("Verdana", 14), fg="gray")
    prediction_label.pack()

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(result.plot())
    ax.axis('off')

    # Embed the plot in the Tkinter interface
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

root = tk.Tk()
root.title("Brain Tumor Detection")
root.geometry("750x700")
root.config(bg="black")

file_explorer = tk.Label(root, text="Choose an image to predict:", 
                         font=("Verdana", 14),
                         width=100, height=4, fg="gray")
button = tk.Button(root, text="Browse files from your computer", command=browse)

file_explorer.pack()
button.pack(pady=20)

canvas_frame = tk.Frame(root)
canvas_frame.pack(fill=tk.BOTH, expand=True)

root.mainloop()
