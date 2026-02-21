import os
import cv2
import numpy as np
from flask import Flask, request, render_template_string
from ultralytics import YOLO
from PIL import Image
import io
import base64

# --- HTML TEMPLATE (Embedded in the script) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv8 Single-File App</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; background: #f0f2f5; padding: 40px; color: #333; }
        .card { max-width: 700px; margin: auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
        h1 { color: #1a73e8; margin-bottom: 20px; }
        .upload-section { border: 2px dashed #ccc; padding: 20px; border-radius: 10px; margin-bottom: 20px; transition: 0.3s; }
        .upload-section:hover { border-color: #1a73e8; }
        button { background: #1a73e8; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background: #1557b0; }
        .result-img { max-width: 100%; border-radius: 10px; margin-top: 20px; border: 4px solid #fff; box-shadow: 0 4px 10px rgba(0,0,0,0.2); }
        .badge { display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: bold; margin-top: 15px; }
        .PEDESTRIANS { background: #fee2e2; color: #dc2626; border: 1px solid #dc2626; }
        .VEHICLES { background: #dcfce7; color: #16a34a; border: 1px solid #16a34a; }
    </style>
</head>
<body>
    <div class="card">
        <h1>🚦 Traffic AI Monitor</h1>
        <p>Upload a traffic snapshot to determine signal priority.</p>
        
        <form method="post" enctype="multipart/form-data">
            <div class="upload-section">
                <input type="file" name="file" accept="image/*" required>
            </div>
            <button type="submit">Analyze Traffic</button>
        </form>

        {% if error %}
            <p style="color: #dc2626; margin-top: 20px;">{{ error }}</p>
        {% endif %}

        {% if img_data %}
            <div class="badge {{ priority }}">
                PRIORITY: {{ priority }} ({{ count }} Pedestrians)
            </div>
            <br>
            <img class="result-img" src="data:image/jpeg;base64,{{ img_data }}" alt="Detection Result">
        {% endif %}
    </div>
</body>
</html>
"""

app = Flask(__name__)

# Load YOLOv8 model (downloads 'yolov8n.pt' on first run)
model = YOLO('yolov8n.pt')

def process_image(img_file):
    # Read image using PIL and convert to OpenCV format
    img = Image.open(img_file).convert("RGB")
    img_array = np.array(img)
    # Convert RGB to BGR for OpenCV/YOLO consistency
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run Detection
    results = model(img_bgr)
    
    # Count persons (Class 0 in COCO dataset)
    person_count = 0
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                person_count += 1
    
    # Generate annotated image (result is in BGR)
    annotated_img = results[0].plot()
    
    # Encode to Base64
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return person_count, img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template_string(HTML_TEMPLATE, error="No file uploaded")
        
        try:
            count, img_data = process_image(file)
            # Logic: If more than 5 people, prioritize Pedestrians
            priority = "PEDESTRIANS" if count > 5 else "VEHICLES"
            
            return render_template_string(HTML_TEMPLATE, 
                                        count=count, 
                                        img_data=img_data, 
                                        priority=priority)
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error=str(e))
    
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    # Using threaded=False because YOLO/PyTorch can sometimes have issues with 
    # Flask's default reloader in multi-threaded environments.
    app.run(debug=True, port=5000)
