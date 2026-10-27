import cv2
from src.detector import Detector

# Instancia o detector
detector = Detector(conf_threshold=0.4)
cap = cv2.VideoCapture("data/raw/EXAMPLE.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # Fim do vídeo
    
    # Extrai as detecções do frame atual
    detections = detector.detect(frame)

    # Desenha cada caixa detectada no frame
    for detection in detections:
        x1, y1, x2, y2, score = detection
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    cv2.namedWindow("Deteccoes YOLO", cv2.WINDOW_NORMAL)
    
    # Exibe o frame na tela
    cv2.imshow("Deteccoes YOLO", frame)
    
    # Sai se pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
         break

cap.release()
cv2.destroyAllWindows()