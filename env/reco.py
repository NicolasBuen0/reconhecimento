import cv2
import face_recognition

# Função para capturar uma imagem da câmera
def capture_image():
    cap = cv2.VideoCapture(0)  # Abrir a câmera padrão (índice 0)
    
    if not cap.isOpened():
        print("Erro ao acessar a câmera.")
        return None
    
    ret, frame = cap.read()  # Capturar um frame da câmera
    if not ret:
        print("Erro ao capturar o frame.")
        return None
    
    cap.release()  # Liberar a captura de vídeo
    return frame

# Função para detectar rostos na imagem
def detect_faces(image):
    # Convertendo a imagem de BGR (OpenCV) para RGB (dlib)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detectar rostos na imagem
    face_locations = face_recognition.face_locations(rgb_image)
    
    return face_locations

# Função para comparar rostos com um banco de dados de imagens
def compare_faces(image, encodings_database):
    # Convertendo a imagem de BGR (OpenCV) para RGB (dlib)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Codificar o rosto na imagem de entrada
    encoding_live = face_recognition.face_encodings(rgb_image)[0]  # Supondo que haja apenas um rosto na imagem
    
    # Comparar a codificação facial ao vivo com as codificações do banco de dados
    threshold = 0.6  # Ajuste isso conforme necessário para o seu caso
    
    for idx, encoding_db in enumerate(encodings_database):
        dist = face_recognition.face_distance([encoding_db], encoding_live)[0]
        print(f"Distância para imagem {idx + 1}: {dist}")
        
        if dist < threshold:
            return f"Correspondência encontrada com a imagem {idx + 1}"
    
    return "Nenhuma correspondência encontrada"

# Função principal
def main():
    # Imagens de treinamento (codificadas previamente)
    # Suponha que você tenha essas codificações de um banco de dados anteriormente
    encodings_database = [
        # Substitua esses arrays pelos codings reais de suas imagens de treinamento
        # Exemplo de codificação facial de uma imagem de treinamento
        [-0.11953112, 0.15078858, 0.06576268, ...],
        # Adicione mais codificações conforme necessário
    ]
    
    # Capturar uma imagem da câmera
    print("Capturando imagem...")
    image = capture_image()
    if image is None:
        return
    
    # Detectar rostos na imagem capturada
    print("Detectando rostos...")
    face_locations = detect_faces(image)
    
    # Desenhar caixas delimitadoras nos rostos detectados
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # Mostrar a imagem com os rostos detectados
    cv2.imshow('Rostos Detectados', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Comparar rostos detectados com o banco de dados
    print("Comparando com o banco de dados...")
    result = compare_faces(image, encodings_database)
    print(result)

if __name__ == "__main__":
    main()
