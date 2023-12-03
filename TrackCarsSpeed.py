
from flask import Flask, flash, request, redirect, url_for, render_template, Response
import urllib.request
import os
import pandas as pd
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
from tracker import *
import time
import datetime
import cvzone
import csv

app = Flask(__name__)

# Set a secret key for session management
app.secret_key = "my_super_secret_key_12345"

# Define allowed file extensions for video uploads
ALLOWED_EXTENSIONS = set(['mp4'])

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Route for the home page
@app.route('/')
def home():
    return render_template('display.html')

# Initialize YOLO model
model = YOLO('yolov8s.pt')

# Route for uploading a video file
@app.route('/', methods=['POST', 'GET'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No video selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(filename)
        flash('Video successfully uploaded')

        # Read class information from a file
        my_file = open("Coco.txt", "r")
        data = my_file.read()
        class_list = data.split("\n")

        # Open the video file for processing
        cap = cv2.VideoCapture(filename)

        # Create a tracker
        tracker = Tracker()

        # Define fields for the CSV file
        fields = ['ID', 'Speed', 'Speed Status', 'Direction', 'Date']

        # Define the name of the CSV file
        filename = "Traffic Report.csv"

        # Open the CSV file for writing
        csvfile = open(filename, 'w', newline='')

        # Create a CSV writer object
        csvwriter = csv.writer(csvfile)

        # Write the header row to the CSV file
        csvwriter.writerow(fields)

        # Define parameters for speed detection
        cy1 = 276
        cy2 = 368
        offset = 40
        vh_down = {}
        counter = []
        vh_up = {}
        counter1 = []
        vh_down_sp = {}
        vh_up_sp = {}
        over_speed_down = []
        over_speed_up = []
        time_passed_incoming = {}
        time_passed_outgoing = {}
        checker_list1 = []
        checker_list2 = []

        # Function to generate frames and process traffic
        def generate_frames():
            while True:
                # Open the CSV file for writing
                csvfile = open(filename, 'w', newline='')
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (1020, 500))

                # Use YOLO model for object detection
                results = model.predict(frame)
                a = results[0].boxes.data
                px = pd.DataFrame(a).astype("float")
                bbox_list = []

                # Extract bounding boxes for cars
                for index, row in px.iterrows():
                    x1 = int(row[0])
                    y1 = int(row[1])
                    x2 = int(row[2])
                    y2 = int(row[3])
                    d = int(row[5])
                    c = class_list[d]
                    if 'car' in c:
                        bbox_list.append([x1, y1, x2, y2])
                bbox_id = tracker.update(bbox_list)

                a_speed_kh = 0
                for bbox in bbox_id:
                    x3, y3, x4, y4, id = bbox
                    cx = int((x3 + x4) / 2)
                    cy = int((y3 + y4) / 2)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 100, 255), 2)

                    # Detect incoming traffic
                    if cy1 - offset < cy < cy1 + offset:
                        vh_down[id] = time.time()
                    if id in vh_down:
                        if cy2 - offset < cy < cy2 + offset:
                            elapsed_time = time.time() - vh_down[id]

                            if id not in counter:
                                time_passed_incoming[id] = datetime.datetime.now().strftime("%c")
                                counter.append(id)
                                distance = 20  # meters
                                a_speed_ms = distance / elapsed_time
                                a_speed_kh = a_speed_ms * 3.6
                                vh_down_sp[id] = a_speed_kh

                    if id in vh_down_sp.keys():
                        if vh_down_sp[id] >= 25:
                            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                            if id not in over_speed_down:
                                over_speed_down.append(id)
                            if id in over_speed_down and id not in checker_list1:
                                # Write over speeding data to CSV
                                data_save = [[id, vh_down_sp[id], 'Over Speeding', 'Incoming', time_passed_incoming[id]]]
                                csvwriter.writerows(data_save)
                                checker_list1.append(id)

                            elif id not in checker_list1:
                                # Write normal speed data to CSV
                                data_save = [[id, vh_down_sp[id], 'Normal Speed', 'Incoming', time_passed_incoming[id]]]
                                csvwriter.writerows(data_save)
                                checker_list1.append(id)

                            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                            cv2.putText(frame, str(time_passed_incoming[id]), (x4, y4 - 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                            cv2.putText(frame, str(int(vh_down_sp[id])) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

                    # Detect outgoing traffic
                    if cy2 - offset < cy < cy2 + offset:
                        vh_up[id] = time.time()
                    if id in vh_up:
                        if cy1 - offset < cy < cy1 + offset:
                            elapsed1_time = time.time() - vh_up[id]

                            if id not in counter1:
                                time_passed_outgoing[id] = datetime.datetime.now().strftime("%c")
                                counter1.append(id)
                                distance1 = 20  # meters
                                a_speed_ms1 = distance1 / elapsed1_time
                                a_speed_kh1 = a_speed_ms1 * 3.6
                                vh_up_sp[id] = a_speed_kh1

                    if id in vh_up_sp.keys():
                        if vh_up_sp[id] >= 25:
                            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                            if id not in over_speed_up:
                                checker2 = "True"
                                over_speed_up.append(id)
                            if id in over_speed_up and id not in checker_list2:
                                # Write over speeding data to CSV
                                data_save = [[id, vh_up_sp[id], 'Over Speeding', 'Out Going', time_passed_outgoing[id]]]
                                csvwriter.writerows(data_save)
                                checker_list2.append(id)

                            elif id not in checker_list2:
                                # Write normal speed data to CSV
                                data_save = [[id, vh_up_sp[id], 'Normal Speed', 'Out Going', time_passed_outgoing[id]]]
                                csvwriter.writerows(data_save)
                                checker_list2.append(id)

                            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                            cv2.putText(frame, str(time_passed_outgoing[id]), (x4, y4 - 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                            cv2.putText(frame, str(int(vh_up_sp[id])) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

                # Draw lane lines and display statistics
                cv2.line(frame, (371, cy1), (701, cy1), (255, 255, 255), 1)
                cv2.putText(frame, ('L1'), (368, 274), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)
                cv2.putText(frame, ('L2'), (182, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

                d = len(counter)
                u = len(counter1)
                o1 = len(over_speed_down)
                o2 = len(over_speed_up)

                # Display traffic statistics on the frame
                cvzone.putTextRect(frame, f'Incoming: {str(d)}', (760, 50), 2, 1, colorR=(0, 100, 0), font=cv2.FONT_HERSHEY_PLAIN, colorT=(255, 255, 255))
                cvzone.putTextRect(frame, f'Out-going: {str(u)}', (40, 50), 2, 1, colorR=(0, 100, 0), font=cv2.FONT_HERSHEY_PLAIN, colorT=(255, 255, 255))
                cvzone.putTextRect(frame, f'Over Speed: {str(o1)}', (760, 110), 2, 1, colorR=(0, 100, 0), font=cv2.FONT_HERSHEY_PLAIN, colorT=(255, 255, 255))
                cvzone.putTextRect(frame, f'Over Speed: {str(o2)}', (40, 110), 2, 1, colorR=(0, 100, 0), font=cv2.FONT_HERSHEY_PLAIN, colorT=(255, 255, 255))

                # Encode the frame as JPEG and yield it for streaming
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Return the response with the generated frames as a video stream
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        flash('Allowed video types are - mp4')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
