import os
import json
import pandas as pd
import tarfile
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import re
from ultralytics import YOLO

class ImageProcessor:

    # 🔹 클래스 변수 추가
    m = (255, 0, 255)
    c = (0, 255, 255)
    y = (255, 255, 0)

    KEYPOINT_DICT = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
    }

    KEYPOINT_EDGES = {
        (0, 1): m, (0, 2): c, (1, 3): m, (2, 4): c, (0, 5): m, (0, 6): c,
        (5, 7): m, (7, 9): m, (6, 8): c, (8, 10): c, (5, 6): y, (5, 11): m,
        (6, 12): c, (11, 12): y, (11, 13): m, (13, 15): m, (12, 14): c, (14, 16): c
    }

    # MoveNet 모델 로드 (클래스 변수로 유지)
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    movenet = module.signatures['serving_default']

    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)  # YOLOv8 모델 로드

    def json_to_dict(self, path):
        """JSON 파일을 파싱하여 필요한 데이터만 추출"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if 'frames' in data:
            dict_list = []

            for frame in data['frames']:
                for view_name, view_data in frame.items():
                    if 'img_key' in view_data:
                        dict_list.append({
                            'img_key': view_data['img_key'],
                            'type': data['type_info']['type'],
                            'workout': data['type_info']['exercise'],
                            'conditions': data['type_info']['conditions'],
                            'description': data['type_info']['description']
                        })

            return dict_list
        else:
            print(f"⚠ 'frames' 키가 없습니다 - {path}")
            return None

    def detect_pose(self, image):
        """MoveNet을 사용하여 포즈 감지"""
        input_image = tf.image.resize(image, [256, 256])
        input_image = tf.expand_dims(input_image, axis=0)
        input_image = tf.cast(input_image, dtype=tf.int32)

        # Run model inference
        outputs = self.movenet(input_image)  # 🔹 클래스 변수 movenet 사용
        keypoints = outputs['output_0'].numpy()

        return keypoints

    def resized_image(self, image):
        """이미지를 256x256 크기로 조정"""
        return cv2.resize(image, (256, 256))

    def detect_person_yolov8_square_crop(self, image_path, confidence_threshold=0.5):
        """YOLOv8을 사용하여 사람을 감지하고 정사각형으로 크롭하는 함수."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

        H, W, _ = image.shape

        # 객체 탐지 수행
        results = self.model(image)[0]

        best_confidence = 0.0
        best_box = None

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())

            if class_id == 0 and confidence > confidence_threshold:
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box = (x1, y1, x2, y2)

        if best_box:
            x1, y1, x2, y2 = best_box
            w, h = x2 - x1, y2 - y1
            side = max(w, h)

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            new_x1 = int(max(0, min(W - side, center_x - side / 2)))
            new_y1 = int(max(0, min(H - side, center_y - side / 2)))
            new_x2 = new_x1 + int(side)
            new_y2 = new_y1 + int(side)

            cropped_image = image[new_y1:new_y2, new_x1:new_x2]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            return self.resized_image(cropped_image)
        else:
            print("사람이 탐지되지 않았습니다.")
            return None

    def unziped_tar(self, path, save_path='E:/project_data/unziped_file/img_temp/'):
        with tarfile.open(path, "r", encoding="utf-8") as tar:
            tar.extractall(save_path)
        print('압축해제 완료')

    def remove_file(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)  # 폴더 내부까지 완전히 삭제
            print("폴더가 삭제되었습니다.")
        else:
            print("폴더가 존재하지 않습니다.")
