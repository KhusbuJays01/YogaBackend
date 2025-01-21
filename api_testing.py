from flask import Flask, request, jsonify
import cv2
import numpy as np
import pickle as pk
import mediapipe as mp
import pandas as pd
from landmarks import extract_landmarks
from calc_angles import rangles
from recommendations import check_pose_angle
from time import time

# Initialize Flask app
app = Flask(__name__)

# Load model and necessary files
model = pk.load(open("./models/4_poses.model", "rb"))
angles_df = pd.read_csv("./csv_files/4_angles_poses_angles.csv")
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

# API endpoint to process the image
@app.route('/predict', methods=['POST'])
def predict_pose():
    try:
        # Check if file is in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        # Read the image file from the request
        file = request.files['image']
        np_img = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Preprocess the image
        resized_image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)

        # Extract landmarks
        err, df, landmarks = extract_landmarks(resized_image, mp_pose, cols)
        if err:
            return jsonify({'error': 'Unable to extract landmarks'}), 500

        # Predict pose
        prediction = model.predict(df)
        probabilities = model.predict_proba(df)

        if probabilities[0, prediction[0]] > 0.85:
            pose_name = get_pose_name(prediction[0])
            angles = rangles(df, landmarks_points)
            suggestions = check_pose_angle(prediction[0], angles, angles_df)

            # Return prediction and suggestions
            return jsonify({
                'pose_name': pose_name,
                'confidence': probabilities[0, prediction[0]],
                'suggestions': suggestions
            })

        return jsonify({'pose_name': "No Pose Detected", 'confidence': probabilities[0, prediction[0]]}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
