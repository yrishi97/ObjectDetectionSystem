import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from image_detection import detect_objects_in_image
from video_detection import detect_objects_in_video
from live_detection import detect_objects_live

app = Flask(__name__)

# Folder paths
UPLOAD_FOLDER = 'static/uploads'
DETECTION_FOLDER = 'static/detections'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTION_FOLDER'] = DETECTION_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTION_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        detected_file = None

        # Check the file type for video or image
        if filename.lower().endswith(('mp4', 'avi', 'mov')):
            detected_file = detect_objects_in_video(file_path)
        else:
            detected_file = detect_objects_in_image(file_path)

        # Log the detected file path
        print("Detected File Path:", detected_file)
        return render_template('results.html', detected_file=detected_file)

    return redirect(request.url)

@app.route('/live')
def live_page():
    return render_template('live.html')

@app.route('/start-live-detection', methods=['POST'])
def start_live_detection():
    detect_objects_live()
    return redirect(url_for('live_page'))

if __name__ == '__main__':
    app.run(debug=True)
