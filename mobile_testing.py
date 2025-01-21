# import requests
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import numpy as np
# import pickle as pk
# import mediapipe as mp
# import pandas as pd
# from landmarks import extract_landmarks
# from calc_angles import rangles
# from recommendations import check_pose_angle

# app = Flask(__name__)
# CORS(app)

# # Load model and necessary files
# model = pk.load(open("./models/4_poses.model", "rb"))
# angles_df = pd.read_csv("./csv_files/4_angles_poses_angles.csv")
# mp_pose = mp.solutions.pose

# # Helper functions
# def download_image(image_url):
#     response = requests.get(image_url, stream=True)
#     if response.status_code == 200:
#         img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
#         return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#     else:
#         raise Exception(f"Failed to download image: {response.status_code}")

# def get_pose_name(index):
#     names = {
#         0: "Adho Mukha Svanasana",
#         1: "Phalakasana",
#         2: "Utkata Konasana",
#         3: "Vrikshasana",
#     }
#     return names.get(index, "Unknown Pose")

# @app.route('/upload', methods=['POST'])
# def classify_pose():
#     try:
#         data = request.get_json()
#         image_url = data.get('image_url')
#         if not image_url:
#             return jsonify({"error": "No image URL provided"}), 400

#         # Download the image from the URL
#         image = download_image(image_url)

#         # Resize and preprocess the image
#         resized_image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)

#         # Extract landmarks
#         cols = ["landmark columns here"]  # Replace with actual column names
#         err, df, landmarks = extract_landmarks(resized_image, mp_pose, cols)
#         if err:
#             return jsonify({'error': 'Unable to extract landmarks'}), 500

#         # Predict pose
#         prediction = model.predict(df)
#         probabilities = model.predict_proba(df)

#         if probabilities[0, prediction[0]] > 0.85:
#             pose_name = get_pose_name(prediction[0])
#             angles = rangles(df, landmarks)
#             suggestions = check_pose_angle(prediction[0], angles, angles_df)

#             return jsonify({
#                 'pose_name': pose_name,
#                 'confidence': probabilities[0, prediction[0]],
#                 'suggestions': suggestions
#             })

#         return jsonify({'pose_name': "No Pose Detected", 'confidence': probabilities[0, prediction[0]]}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pickle as pk
import pandas as pd
import logging
from uuid import uuid4
import time
import mediapipe as mp
from landmarks import extract_landmarks
from calc_angles import rangles
from recommendations import check_pose_angle

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "*"}})  # Enable CORS only for /upload endpoint

# Configure upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and necessary files
model = pk.load(open("./models/4_poses.model", "rb"))
angles_df = pd.read_csv("./csv_files/4_angles_poses_angles.csv")
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Initialize helper variables and functions
def init_dicts():
    landmarks_points = {
        "nose": 0,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
        "left_heel": 29, "right_heel": 30,
        "left_foot_index": 31, "right_foot_index": 32,
    }
    col_names = []
    for key in landmarks_points.keys():
        col_names.extend([f"{key}_x", f"{key}_y", f"{key}_z", f"{key}_v"])
    return col_names, landmarks_points

cols, landmarks_points = init_dicts()

def get_pose_name(index):
    names = {
        0: "Adho Mukha Svanasana",
        1: "Phalakasana",
        2: "Utkata Konasana",
        3: "Vrikshasana",
    }
    return names.get(index, "Unknown Pose")

# Function to clean up old files
def cleanup_old_files(directory, max_age_in_seconds=86400):
    current_time = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.getmtime(file_path) < current_time - max_age_in_seconds:
            os.remove(file_path)

cleanup_old_files(UPLOAD_FOLDER)

# Endpoint to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_and_predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Validate file extension
        valid_extensions = ['.jpg', '.jpeg', '.png']
        if not any(file.filename.lower().endswith(ext) for ext in valid_extensions):
            return jsonify({"error": "Invalid image format"}), 400

        # Secure the filename and add unique identifier
        filename = f"{uuid4().hex}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        with open(file_path, 'rb') as f:
            np_img = np.frombuffer(f.read(), np.uint8)
            image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        resized_image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        err, df, landmarks = extract_landmarks(resized_image, mp_pose, cols)

        if err:
            logging.warning(f"Failed to extract landmarks for image {filename}")
            return jsonify({'error': 'Unable to extract landmarks'}), 500

        probabilities = model.predict_proba(df)
        prediction = probabilities.argmax(axis=1)[0]

        if probabilities[0, prediction] > 0.85:
            pose_name = get_pose_name(prediction)
            angles = rangles(df, landmarks_points)
            suggestions = check_pose_angle(prediction, angles, angles_df)

            return jsonify({
                'pose_name': pose_name,
                'confidence': probabilities[0, prediction],
                'suggestions': suggestions
            })

        return jsonify({'pose_name': "No Pose Detected", 'confidence': probabilities[0, prediction]}), 200

    except Exception as e:
        logging.error(f"Exception occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
