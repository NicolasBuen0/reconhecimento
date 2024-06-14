import cv2
import os

# Criar uma pasta para armazenar as imagens
os.makedirs('imagens_treinamento', exist_ok=True)

# Inicializar a câmera
cap = cv2.VideoCapture(0)

num_imagens = 10
contador = 0

while contador < num_imagens:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Capturando Imagem', frame)
    
    # Salvar a imagem
    img_path = f'imagens_treinamento/usuario_{contador}.jpg'
    cv2.imwrite(img_path, frame)
    
    contador += 1
    
    # Esperar por 1 segundo
    cv2.waitKey(1000)

cap.release()
cv2.destroyAllWindows()
