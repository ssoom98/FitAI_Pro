from django.shortcuts import render
from django.http import HttpResponse
from .forms import VideoUploadForm
from .Video_to_img_processed import ImageProcessor
from .workout_pose_predict import Predictor
from .predict_comment import workout_data
from .predict_comment import evaluate_workout
import json
import os
import shutil
import numpy as np
import cv2

def pose_home(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_video = request.FILES['video']
            # 저장 경로: pose_analysis/static/video/ 폴더
            save_path = os.path.join('pose_analysis/static/video/', uploaded_video.name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb+') as destination:
                for chunk in uploaded_video.chunks():
                    destination.write(chunk)
            return render(request, 'pose/pose_upload_success.html', {'video_name': uploaded_video.name})
    else:
        form = VideoUploadForm()
    return render(request, 'pose/pose_home.html', {'form': form})

def pose_preview(request):
    print(request.method)
    if request.method == 'POST':
        video_name = request.POST['video_name']
        processor = ImageProcessor(model_path = 'pose_analysis/models/yolov8n.pt')
        video_dir = "pose_analysis/static/video/" + video_name
        if os.path.exists(video_dir) and os.path.isfile(video_dir):
            processed_image_path = "pose_analysis/static/processed_image/" + video_name.split(".")[0]
            os.makedirs(processed_image_path, exist_ok=True)
            processor.extract_middle_32_frames(video_dir, processed_image_path)
            key_point = []
            for img_path in os.listdir(processed_image_path):
                path = os.path.join(processed_image_path, img_path)
                img = cv2.imread(path, cv2.COLOR_BGR2RGB)
                key_point.append(processor.detect_pose(img))
            predictor = Predictor()
            key_point = np.array(predictor.pad_features(key_point)).reshape(1, -1)
            pred_workout = predictor.workout_predict(key_point)[0]
            json_file_path = "pose_analysis/static/workout.json"
            with open(json_file_path, 'r', encoding='utf-8') as file:
                workout_dict = json.load(file)
            workout = workout_dict[str(pred_workout)]
            keypoint_json = json.dumps(key_point.tolist(), ensure_ascii=False)
            return render(request, 'pose/pose_preview.html', {'workout':workout, "keypoint":keypoint_json})

def pose_workout_select(request):
    key_point = request.POST['keypoint']
    return render(request, 'pose/pose_workout_select.html',{"keypoint":key_point})

def pose_predict(request):
    keypoint_str = request.POST['keypoint']
    print(keypoint_str)
    # JSON 문자열을 파이썬 리스트로 변환 후 넘파이 배열로 복원
    keypoint = np.array(json.loads(keypoint_str))
    print(type(keypoint))
    keypoint = keypoint.reshape(-1)
    workout = request.POST['workout']
    predictor = Predictor()
    pose_dict = predictor.pose_predict(keypoint=keypoint, workout=workout)
    predicte_comment  = evaluate_workout(workout_name=workout, predicted=pose_dict[0], workout_data=workout_data)
    video_path = "pose_analysis/static/video"
    img_path = "pose_analysis/static/processed_image/"
    # 업로드된 비디오 삭제
    for file_name in os.listdir(video_path):
        file_path = os.path.join(video_path, file_name)
        # 파일인 경우에만 삭제 (디렉토리는 삭제하지 않음)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"{file_path} 삭제 완료")
    # 업로드된 이미지 삭제
    for item in os.listdir(img_path):
        item_path = os.path.join(img_path, item)
        if os.path.isdir(item_path):  # 폴더인 경우 삭제
            shutil.rmtree(item_path)
            print(f"Deleted folder: {item_path}")

    return render(request, 'pose/pose_predict.html', {"pose_predict":predicte_comment})