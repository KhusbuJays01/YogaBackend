from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Directory to save received images
UPLOAD_FOLDER = 'received_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        # Check if an image is present in the request
        if 'image' not in request.files:
            return jsonify({'message': 'No image uploaded'}), 400
        
        image = request.files['image']
        
        # Save the image with its original filename
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)
        
        # Process the image here if needed
        return jsonify({'message': 'Image received successfully', 'filename': image.filename}), 200
    
    except Exception as e:
        return jsonify({'message': 'An error occurred', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
