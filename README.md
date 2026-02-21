
# Pedestrian System Analyzer 🚦

## Basic Details

### Team Name: GridMaster

### Team Members

* Member 1: Asaswara - College of Engineering Trikaripur
* Member 2: Devika - College of Engineering Trikaripur


### Project Description

This project is a Flask-based web application that utilizes the YOLOv8 object detection model to count pedestrians in uploaded images. It provides real-time traffic priority suggestions based on human density to assist in pedestrian safety and traffic management.

### The Problem statement

Static traffic signals do not account for real-time pedestrian volume, often leading to inefficient wait times for large groups of people or unnecessary stops for vehicles when no pedestrians are present.

### The Solution

The application uses the YOLOv8 'Nano' model to identify and count individuals (specifically class 0) in a scene. If more than 5 pedestrians are detected, the system flags a priority for pedestrians; otherwise, it suggests prioritizing vehicles.

---

## Technical Details

### Technologies/Components Used

**For Software:**

* **Languages used**: Python
* **Frameworks used**: Flask
* **Libraries used**: `ultralytics` (YOLOv8), `OpenCV` (`cv2`), `Pillow` (`PIL`), `NumPy`, `base64`
* **Tools used**: VS Code, Git

---

## Features

List the key features of your project:

* **Real-time Detection**: Uses the YOLOv8n model for fast and efficient pedestrian identification.
* **Automated Decision Logic**: Automatically recommends traffic priority based on a threshold of 5 persons.
* **Visual Feedback**: Displays annotated images with bounding boxes drawn around detected individuals.
* **Web Interface**: Includes a clean, Bootstrap-based UI for easy image uploads and instant results.

---

## Implementation

### For Software:

#### Installation

```bash
pip install flask ultralytics opencv-python pillow numpy

```

#### Run

```bash
python app.py

```

---

## Project Documentation

### For Software:

#### Screenshots (Add at least 3)

![Screenshot1](/img/img1.png)
*Initial upload screen for the Pedestrian System Analyzer*

![Screenshot2](img\img2.png)
*Result showing "Prioritize Pedestrians" alert for high density counts*

![Screenshot3](img\img3.png)
*Result showing "Prioritize Vehicles" alert for low density counts*

#### Diagrams

**System Architecture:**

*The user uploads an image through the Flask frontend, which is processed by the YOLOv8 model in the backend to count persons and return an annotated image.*

**Application Workflow:**

*Upload Image -> YOLOv8 Inference -> Count People -> Logical Threshold Check (Count > 5) -> Display Result & Action.*

---

## Additional Documentation

### For Web Projects with Backend:

#### API Documentation

**Base URL:** `http://localhost:5000`

##### Endpoints

**GET /**

* **Description**: Loads the main pedestrian analyzer interface.

**POST /**

* **Description**: Processes the uploaded image and returns the pedestrian count and annotated image.
* **Parameters**:
* `file` (image): The image to be analyzed.


* **Response**: HTML page containing the `person_count` and the Base64 encoded `img_data`.


---

## AI Tools Used (Optional)

**Tool Used**: YOLOv8 (Ultralytics), Flask

**Purpose**:

* Object detection and pedestrian counting.
* Web server management and image rendering.

**Percentage of AI-generated code**: Approximately 20%

---

## Team Contributions

* **Anaswara**: Backend development and YOLOv8 model integration.
* **Devika**: Frontend HTML/Bootstrap design and documentation.

---

Made with ❤️ at TinkerHub