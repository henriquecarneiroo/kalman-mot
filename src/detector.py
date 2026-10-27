import numpy as np
from ultralytics import YOLO

class Detector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        detection_classes: list = None,
    ):
        # Carrega a rede neural com os pesos informados
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.dectection_classes = detection_classes
    
    def detect(self, frame):
        """
        Recebe como entrada um frame e retorna uma lista com os pares de pontos e score
        """
        
        # Roda o modelo
        results = self.model(
            frame, 
            conf=self.conf_threshold, 
            classes=self.dectection_classes,
            verbose=False,
        )
        
        # Verifica se algo foi encontrado
        boxes = results[0].boxes
        if len(boxes) == 0:
            return np.empty((0, 5))
        
        # Convertendo os tensores para numpy
        xyxy = boxes.xyxy.cpu().numpy() # (N, 4)
        
        # Extraindo as confianças e convertendo para numpy
        conf = boxes.conf.cpu().numpy() # (N)
        
        # Juntando as coordenadas e confiança em uma única matriz
        conf = conf.reshape(-1, 1) # Transforma o vetor N em uma matriz (N, 1)
        detections = np.hstack((xyxy, conf)) # Junta à direita
        
        return detections
                
        
            

        