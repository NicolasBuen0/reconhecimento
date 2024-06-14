import cv2
import os
import numpy as np
import face_recognition

def capturar_imagem():
    cap = cv2.VideoCapture(0)  # Abre a câmera padrão (0) - pode ser necessário ajustar o número da câmera
    
    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return None
    
    ret, frame = cap.read()  # Captura um quadro da câmera
    
    if ret:
        cv2.imshow('Imagem ao vivo', frame)  # Exibe o quadro capturado
        cv2.waitKey(0)  # Aguarda pressionamento de uma tecla
        cv2.destroyAllWindows()  # Fecha a janela
        cap.release()  # Libera o objeto da câmera
        return frame
    else:
        print("Erro ao capturar quadro da câmera.")
        cap.release()
        return None

def carregar_codificacoes_encodes(diretorio_encodes):
    codificacoes = {}
    
    for filename in os.listdir(diretorio_encodes):
        if filename.endswith('.npy'):
            path = os.path.join(diretorio_encodes, filename)
            codificacao = np.load(path)
            pessoa = filename[:-4]  # Remove a extensão .npy para obter o nome da pessoa
            codificacoes[pessoa] = codificacao
    
    return codificacoes

def comparar_com_encodes(imagem_ao_vivo, codificacoes):
    imagem_encodings = face_recognition.face_encodings(imagem_ao_vivo)
    
    if len(imagem_encodings) == 0:
        print("Nenhuma face encontrada na imagem ao vivo.")
        return
    
    imagem_encoding = imagem_encodings[0]  # Assume que há apenas uma face na imagem ao vivo
    
    for pessoa, codificacao in codificacoes.items():
        comparacao = face_recognition.compare_faces([codificacao], imagem_encoding)
        
        if comparacao[0]:  # Se houver correspondência
            print(f"Correspondência encontrada: {pessoa}")
            return
    
    print("Nenhuma correspondência encontrada.")

def main():
    # Diretório onde as codificações faciais foram salvas
    diretorio_encodes = 'encodes'

    # Carrega as codificações faciais
    codificacoes = carregar_codificacoes_encodes(diretorio_encodes)
    
    # Captura uma imagem ao vivo
    imagem_ao_vivo = capturar_imagem()

    # Verifica se a imagem ao vivo foi capturada com sucesso
    if imagem_ao_vivo is None:
        print("Não foi possível capturar a imagem ao vivo.")
        exit()

    # Realiza a comparação com as codificações carregadas
    comparar_com_encodes(imagem_ao_vivo, codificacoes)

if __name__ == "__main__":
    main()
