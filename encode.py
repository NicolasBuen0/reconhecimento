import os
import numpy as np
import face_recognition

def encode_training_images(training_folder='imagens_treinamento', encodes_folder='encodes'):
    # Verifica se a pasta 'encodes' já existe
    if not os.path.exists(encodes_folder):
        os.makedirs(encodes_folder)
        print(f"Pasta '{encodes_folder}' criada.")

    for filename in os.listdir(training_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            path = os.path.join(training_folder, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                encoding = encodings[0]  # Pega apenas a primeira face encontrada
                name = os.path.splitext(filename)[0]  # Obtém o nome do arquivo sem a extensão
                encode_path = os.path.join(encodes_folder, f"{name}.npy")
                np.save(encode_path, encoding)
                print(f"Codificação salva para {name}.jpg")
            else:
                print(f"Não foi possível encontrar uma face em {filename}")

def main():
    encode_training_images()

if __name__ == "__main__":
    main()
