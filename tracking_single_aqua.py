from ultralytics import YOLO
import cv2
import os
import logging
import numpy as np
import requests
from estimator_func import positional_estimate
from PIL import Image
import tempfile
import ollama
from pydantic import BaseModel
import matplotlib.pyplot as plt

np.set_printoptions(precision=16, suppress=True)

# Load data
data = np.genfromtxt('tracking_data.csv', skip_header=1, delimiter=',', dtype=np.float64)

# Suppress ultralytics logs
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Load YOLO model
model = YOLO("yolo11n_trained.pt")

video_path = "./videos/tracking.MP4"

true_gps_lat = 13.1939387032944
true_gps_lon = -59.6415065636197
	
# Create output directories
output_dir = "output_videos"
image_output_dir = "output_images"  # Directory for saved images
os.makedirs(output_dir, exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f"{output_dir}/output_video1.mp4", fourcc, fps, (width, height))

frame_count = 0
error = []

# Control variable
display = 1  # Set to 1 to save images

class aqua_position(BaseModel):
    X_pos: int
    Y_pos: int
    reason: str  # Explanation of the decision

class aqua_validation(BaseModel):
    Answer: str
    reason: str  # Explanation of the decision

# Function to send image to LLaVA/Ollama
def send_to_model(image_path):
    aqua_path = 'output_images/aqua.jpg'

    prompt = '''In the first image is there a robot inside of the red bounding box. An example of this image is seen in the second image?
    ```json
    {{
        "answer": yes or no,
        "reason": why yes or no?
    }}
    ```
    '''

    response = ollama.chat(
	model="llava",
	messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [image_path,aqua_path]
            }
        ],
    format=aqua_validation.model_json_schema(),
    )
    response_output = aqua_validation.model_validate_json(response.message.content)
    aqua_existance = response_output.Answer.lower()

    if aqua_existance == "yes":
        return True
    else:
        return False
    

tmp_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        for i, box in enumerate(result.boxes):
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1
            x_center = x1 + abs(x2 - x1)
            y_center = height - (y1 + abs(y2 - y1))

            # Get class label
            class_id = int(box.cls[0])
            label = model.names[class_id]  # Get YOLO label
            
            # Compute positional estimate error
            #altitude,drone_angle,camera_angle,heading,lat,lon,pixel_x,pixel_y, true_lat,true_lon
            value_error = positional_estimate(
                data[frame_count][2], data[frame_count][3], data[frame_count][5], data[frame_count][4], 
                data[frame_count][0], data[frame_count][1], x_center, y_center, 
                 true_gps_lat, true_gps_lon
            )
            error.append(value_error)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # --- format the numbers to 8 significant figures ---
            lat = f"{value_error[1]:.10g}"
            lon = f"{value_error[2]:.10g}"

            # --- compose the two lines ---
            line1 = f"Aqua GNSS Coordinates:"   # first line
            line2 = f"{lat}, {lon}"               # second line (numbers only)

            # --- drawing parameters ---
            origin_x, origin_y = x1, y2 + 40   # anchor for the first line
            font        = cv2.FONT_HERSHEY_SIMPLEX
            font_scale  = 1.0                  # larger text
            color       = (255, 255, 0)        # light blue / cyan (BGR)
            thickness   = 2
            line_gap_px = 30                   # vertical spacing between lines

            # --- render the two lines ---
            cv2.putText(frame, line1, (origin_x, origin_y),
                        font, font_scale, color, thickness, cv2.LINE_AA)

            cv2.putText(frame, line2, (origin_x, origin_y + line_gap_px),
                        font, font_scale, color, thickness, cv2.LINE_AA)


    # Write the modified frame to the output video
    out.write(frame)

    tmp_counter += 1
    if tmp_counter == 3:
        frame_count += 1
        tmp_counter = 0


print("Frames processed:", frame_count)
print("Min error:", np.min(np.array(error)))
print("Min error:", np.max(np.array(error)))

cap.release()
out.release()
cv2.destroyAllWindows()
