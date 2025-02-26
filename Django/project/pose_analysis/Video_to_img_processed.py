import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

class ImageProcessor:
    # 클래스 변수 추가
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
    module = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-thunder/4")
    movenet = module.signatures['serving_default']

    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)  # YOLOv8 모델 로드

    def detect_pose(self, image):
        """MoveNet을 사용하여 포즈 감지"""
        input_image = tf.image.resize(image, [256, 256])
        input_image = tf.expand_dims(input_image, axis=0)
        input_image = tf.cast(input_image, dtype=tf.int32)
        outputs = self.movenet(input_image)
        keypoints = outputs['output_0'].numpy()
        return keypoints

    def extract_middle_32_frames(self, video_path, output_dir, confidence_threshold=0.5, bg_color=(0, 0, 0)):
        """
        동영상 파일에서 초당 1프레임씩 추출한 후, 전체 추출된 프레임 중 가운데 17프레임에 대해
        YOLOv8 FHD 크롭(detect_person_yolov8_fhd_crop_from_frame)을 적용하여 지정한 output_dir에 저장합니다.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("비디오 파일을 열 수 없습니다.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(fps))
        frames = []
        frame_index = 0

        # 초당 1프레임씩 추출하여 리스트에 저장
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_interval == 0:
                frames.append(frame)
            frame_index += 1

        cap.release()

        total_frames = len(frames)
        if total_frames < 17:
            print(f"추출된 프레임 수({total_frames})가 32보다 적습니다.")
            return

        start_idx = (total_frames - 18) // 2
        end_idx = start_idx + 17

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in range(start_idx, end_idx):
            frame = frames[i]
            # 추출된 프레임에 대해 사람 객체 FHD 크롭 적용
            processed_frame = self.detect_person_yolov8_fhd_crop_from_frame(frame, confidence_threshold, bg_color)
            if processed_frame is not None:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(output_dir, f"{video_name}_frame_{i:04d}.jpg")
                cv2.imwrite(output_path, processed_frame)
                print(f"저장된 프레임: {output_path}")
            else:
                print(f"프레임 {i:04d}에서 사람이 탐지되지 않았습니다.")

    def detect_person_yolov8_fhd_crop_from_frame(self, frame, confidence_threshold=0.5, bg_color=(0, 0, 0)):
        """
        이미지 배열(frame)을 받아 YOLOv8으로 사람 객체를 탐지한 후,
        탐지된 사람의 중심을 기준으로 1920×1080 크기의 영역을 추출합니다.
        만약 추출 영역이 원본 이미지 범위를 벗어난다면, 부족한 부분은 bg_color(기본: 검은색)으로 채웁니다.
        탐지에 실패하면 None을 반환합니다.
        """
        H, W, _ = frame.shape
        results = self.model(frame)[0]

        best_confidence = 0.0
        best_box = None

        for box in results.boxes:
            # 바운딩 박스 좌표를 정수로 변환
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            if class_id == 0 and confidence > confidence_threshold:
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box = (x1, y1, x2, y2)

        if best_box is None:
            print("사람이 탐지되지 않았습니다.")
            return None

        # 탐지된 사람의 중심 좌표 계산
        x1, y1, x2, y2 = best_box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # FHD 해상도 (1920x1080)
        target_w, target_h = 1920, 1080

        # 사람의 중심을 기준으로 FHD 영역의 경계 계산
        left = int(round(center_x - target_w / 2))
        top = int(round(center_y - target_h / 2))
        right = left + target_w
        bottom = top + target_h

        # FHD 크기의 배경 이미지 생성 (배경색: bg_color)
        output_img = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

        # 원본 이미지와 FHD 영역의 겹치는 부분 계산
        src_x1 = max(0, left)
        src_y1 = max(0, top)
        src_x2 = min(W, right)
        src_y2 = min(H, bottom)

        # output_img에서 복사할 위치 계산 (만약 crop 영역이 이미지 범위를 벗어나면 그만큼 offset)
        dst_x1 = max(0, -left)
        dst_y1 = max(0, -top)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        # 겹치는 영역 복사
        output_img[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

        return output_img

    def draw_pose(self, image, keypoints, confidence_threshold=0.3):
        """
        원본 이미지 위에 MoveNet으로 검출된 keypoints를 점과 선으로 시각화합니다.

        Parameters:
            image: 원본 이미지 (RGB)
            keypoints: detect_pose 함수의 출력 (shape: [1, 1, 17, 3])
            confidence_threshold: keypoint 신뢰도 임계값
        """
        # keypoints 배열을 (17,3) 형태로 변환 (y, x, conf)
        keypoints = keypoints[0, 0, :, :]
        H, W, _ = image.shape

        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        # keypoint 점 그리기
        for i, kp in enumerate(keypoints):
            y, x, conf = kp  # 출력 순서는 [y, x, confidence]
            if conf > confidence_threshold:
                plt.scatter(x * W, y * H, s=100, c='red', marker='o')
                plt.text(x * W, y * H, f'{i}', color='white', fontsize=12)

        # KEYPOINT_EDGES에 정의된 선 그리기
        for edge, color in self.KEYPOINT_EDGES.items():
            i1, i2 = edge
            kp1 = keypoints[i1]
            kp2 = keypoints[i2]
            if kp1[2] > confidence_threshold and kp2[2] > confidence_threshold:
                x1, y1 = kp1[1] * W, kp1[0] * H
                x2, y2 = kp2[1] * W, kp2[0] * H
                # 색상은 0~255 범위이므로, 255로 나눠서 0~1 범위로 변환
                plt.plot([x1, x2], [y1, y2], color=np.array(color) / 255.0, linewidth=2)