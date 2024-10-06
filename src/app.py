from flask import Flask, request, jsonify
import os
from flask import Flask
from flask_cors import CORS
from main import run_predict_and_update
app = Flask(__name__)
CORS(app)  # This will allow all domains. You can customize it to allow specific domains if needed.

@app.route('/')
def home():
    return "Hello, World!"

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

after= False
score=0
@app.route('/upload', methods=['POST'])

def upload_file():
    global score
    global after
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Save the uploaded file to the specified upload folder
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        # Run prediction and update, and capture the return value

        prediction_results, score = run_predict_and_update(file_path, score)
        
        # Return the results along with the file path
        return jsonify({"success": True, "file_path": file_path, "results": prediction_results, "score": score}), 200

if __name__ == '__main__':
    app.run(debug=True)
