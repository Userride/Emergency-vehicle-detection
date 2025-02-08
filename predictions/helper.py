import cv2
import numpy as np
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        # Load YAML file
        with open(data_yaml, mode='r') as f:
            data_yml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yml['names']
        self.nc = data_yml['nc']

        # Load YOLO model with OpenCV
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predictions(self, image):
        # Check if the image is loaded correctly
        if image is None:
            raise ValueError("Input image is None. Make sure the image is loaded properly.")

        row, col, d = image.shape
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()  # Prediction from YOLO model

        detection = preds[0]
        boxes = []
        predicted_texts = []

        for i in range(len(detection)):
            row = detection[i]
            confidence = row[4]
            if confidence > 0.4:
                class_score = row[5:].max()
                class_id = row[5:].argmax()
                
                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    left = int((cx - 0.5 * w) * (image.shape[1] / INPUT_WH_YOLO))
                    top = int((cy - 0.5 * h) * (image.shape[0] / INPUT_WH_YOLO))
                    width = int(w * (image.shape[1] / INPUT_WH_YOLO))
                    height = int(h * (image.shape[0] / INPUT_WH_YOLO))
                    
                    boxes.append([left, top, width, height])
                    predicted_texts.append(f'{self.labels[class_id]} : {int(confidence * 100)}%')

        return image, predicted_texts, boxes
