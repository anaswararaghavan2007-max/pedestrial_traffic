import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(page_title="Pedestrian System Analyzer", page_icon="🚦")

# 2. Load the YOLOv8 model (cached so it only loads once)
@st.cache_resource
def load_model():
    # 'yolov8n.pt' is the nano model: extremely fast and lightweight
    return YOLO('yolov8n.pt')

model = load_model()

# 3. Web App UI
st.title("🚦 Pedestrian System Analyzer Demo")
st.write("Upload a traffic image to count pedestrians and determine the right of way.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    
    # Show a loading spinner while analyzing
    with st.spinner('Analyzing image...'):
        # Convert PIL Image to numpy array for YOLO
        img_array = np.array(image)
        
        # 4. Run object detection
        results = model(img_array)
        
        # 5. Count the people (Class '0' in the COCO dataset corresponds to 'person')
        person_count = 0
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Check if the detected object is a person
                if int(box.cls[0]) == 0: 
                    person_count += 1
        
        # Generate the image with bounding boxes drawn on it
        annotated_img = results[0].plot()
        
    # 6. Display Results
    st.subheader("Analysis Results")
    st.image(annotated_img, caption="Analyzed Image with Bounding Boxes", use_container_width=True)
    
    st.markdown(f"### 🧍 Total Pedestrians Detected: **{person_count}**")
    
    # 7. Priority Logic
    if person_count > 5:
        st.error("🚨 **ACTION: Prioritize Pedestrians**")
    else:
        st.success("✅ **ACTION: Prioritize Vehicles**")