# Final_Summative_Group_Project

# Traffic Monitoring System Readme

This readme provides comprehensive information and instructions for a Flask-based Traffic Monitoring System. The system uses YOLO (You Only Look Once) object detection to monitor traffic on a video stream, track vehicles, and record relevant data in a CSV file. Below is a detailed explanation of the code and how to set up and use the Traffic Monitoring System.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)

## Introduction
The Traffic Monitoring System is a web application that allows you to upload a video file (in mp4 format), analyze it using YOLO for object detection (specifically, vehicle detection), track vehicles' movement, and generate a real-time video stream with vehicle tracking information overlaid on the video. It also records vehicle speed and other details in a CSV file.

Key Features:
- Real-time vehicle tracking and speed detection.
- Record traffic statistics, including vehicle speed, direction, and over-speeding status.
- Generate a live video stream with tracking information overlay.
- User-friendly web interface powered by Flask.

## Prerequisites
Before using the Traffic Monitoring System, ensure you have the following prerequisites installed:
- Python 3.x
- Flask
- OpenCV (cv2)
- Pandas
- Ultralytics (YOLO)
- cvzone (for text overlay)

You can install the necessary Python libraries using pip:
```bash
pip install flask opencv-python pandas ultralytics cvzone
```

Additionally, you'll need a pre-trained YOLO model (in this code, 'yolov8s.pt') for vehicle detection. You can download a YOLO model or use your own.

## Installation
Follow these steps to set up the Traffic Monitoring System:

1. Clone or download the code from this repository.

2. Install the required Python libraries as mentioned in the prerequisites section.

3. Place your pre-trained YOLO model file (e.g., 'yolov8s.pt') in the same directory as the code.

4. Create a file named "Coco.txt" containing the class information for YOLO. Each line should represent a class name (e.g., "car", "bus", etc.). This file is used to filter out non-vehicle objects.

5. Create an empty file named "Traffic_Report.csv" to store traffic data.

## Usage
To run the Traffic Monitoring System:

1. Open a terminal and navigate to the directory containing the code.

2. Run the Flask application using the following command:
```bash
python your_filename.py
```
Replace `your_filename.py` with the name of your Python script containing the code.

3. Access the web interface by opening a web browser and entering the following URL:
```
http://localhost:5000/
```

4. You can now upload a video file (in mp4 format) using the web interface.

5. The system will process the video, detect and track vehicles, and display the live video stream with vehicle tracking information.

6. Traffic statistics, including vehicle speed, direction, and over-speeding status, will be recorded in the "Traffic_Report.csv" file.

## Code Explanation
Here's a brief explanation of the major components and functionality of the code:

- The Flask web application is created and configured with a secret key for session management.

- It defines `ALLOWED_EXTENSIONS` to specify allowed video file extensions.

- The `allowed_file` function checks if the uploaded file has a valid extension.

- Upon uploading a video, the application uses YOLO for vehicle detection and tracking.

- Detected vehicles are tracked using a `Tracker` object.

- Vehicle data is recorded in a CSV file, including ID, speed, speed status, direction, and timestamp.

- Lane lines are drawn on the video to aid in traffic monitoring.

- Real-time statistics about incoming and outgoing traffic and over-speeding vehicles are displayed on the video stream.

- The processed video frames are streamed back to the web interface.

