import os
import numpy as np
import face_recognition
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

def load_encodings(encodes_folder='encodes'):
    encodings = {}
    for filename in os.listdir(encodes_folder):
        if filename.endswith('.npy'):
            name = os.path.splitext(filename)[0]
            encode_path = os.path.join(encodes_folder, filename)
            encodings[name] = np.load(encode_path)
    return encodings

encodings = load_encodings()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image = face_recognition.load_image_file(file_path)
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) == 0:
            return jsonify({"name": "Desconhecido", "message": "Nenhuma face encontrada."})

        face_encoding = face_encodings[0]

        for name, encoding in encodings.items():
            results = face_recognition.compare_faces([encoding], face_encoding)
            if results[0]:
                return jsonify({"name": name, "message": "Correspondência encontrada."})

        return jsonify({"name": "Desconhecido", "message": "Nenhuma correspondência encontrada."})

if __name__ == '__main__':
    app.run(debug=True)
