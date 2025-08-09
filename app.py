import os
from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'} # Add more video formats if needed
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # 100 MB max upload size

# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Checks if the uploaded file has an allowed video extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def count_ocean_waves_web(video_path):
    """
    Counts the number of ocean waves in a video file for web application.
    This version removes the cv2.imshow calls, returning just the count.

    Args:
        video_path (str): The path to the video file.

    Returns:
        int: The estimated number of waves.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return -1

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define zones as a percentage of the frame height
    # These parameters are now part of the function for easier tuning for the web context
    ROI_TOP_PCT = 0.55  # 55% from the top
    ROI_BOTTOM_PCT = 0.75 # 75% from the top
    COUNT_ZONE_TOP_PCT = 0.65 # 65% from the top
    COUNT_ZONE_BOTTOM_PCT = 0.70 # 70% from the top

    ROI_TOP = int(frame_height * ROI_TOP_PCT)
    ROI_BOTTOM = int(frame_height * ROI_BOTTOM_PCT)
    COUNT_ZONE_TOP = int(frame_height * COUNT_ZONE_TOP_PCT)
    COUNT_ZONE_BOTTOM = int(frame_height * COUNT_ZONE_BOTTOM_PCT)

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=True)

    wave_count = 0
    last_wave_time = 0
    cooldown = 1.0  # Cooldown period in seconds to prevent multiple counts

    wave_state = 0 # 0: WAITING, 1: COUNTING
    debounce_frames = 10 # Number of frames to wait before a new wave can be detected
    frames_since_last_detection = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[ROI_TOP:ROI_BOTTOM, :]
        fgmask = fgbg.apply(roi)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wave_detected_in_region = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000: # Min contour area, adjusted for web context
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 50: # Min width for a contour to be considered a wave
                continue
            
            cy = y + h // 2
            full_cy = ROI_TOP + cy
            
            if COUNT_ZONE_TOP <= full_cy <= COUNT_ZONE_BOTTOM:
                wave_detected_in_region = True
        
        # --- State Machine Logic for Counting ---
        if wave_state == 0:  # WAITING state
            if wave_detected_in_region:
                current_time = time.time()
                if (current_time - last_wave_time) > cooldown:
                    wave_state = 1  # Transition to COUNTING
                    wave_count += 1
                    last_wave_time = current_time
                    frames_since_last_detection = 0
                
        elif wave_state == 1:  # COUNTING state
            if not wave_detected_in_region:
                frames_since_last_detection += 1
                if frames_since_last_detection >= debounce_frames:
                    wave_state = 0  # Transition back to WAITING
            else:
                frames_since_last_detection = 0 # Reset counter if wave is still detected

    cap.release()
    return wave_count

@app.route('/')
def upload_form():
    """Renders the video upload form."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handles video upload, processing, and redirection to results."""
    if 'file' not in request.files:
        return redirect(request.url) # No file part in the request

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url) # No selected file

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath) # Save the uploaded video

        # Process the video
        print(f"Processing video: {filepath}")
        wave_count = count_ocean_waves_web(filepath)
        print(f"Finished processing. Wave count: {wave_count}")

        # Clean up: remove the uploaded video file after processing
        os.remove(filepath)
        
        return redirect(url_for('results', count=wave_count))
    else:
        return "Invalid file type. Please upload a video file (mp4, avi, mov)."

@app.route('/results')
def results():
    """Displays the wave counting results."""
    count = request.args.get('count', 'N/A')
    return render_template('results.html', count=count)

if __name__ == '__main__':
    # Run the Flask app
    # Use host='0.0.0.0' to make it accessible externally (e.g., in a Docker container)
    # debug=True allows for auto-reloading and better error messages during development
    app.run(host='0.0.0.0', port=5001, debug=True) # Changed port to 5001
