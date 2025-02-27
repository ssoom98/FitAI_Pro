from django.shortcuts import render
from django.http import HttpResponse
from .forms import VideoUploadForm
from .Video_to_img_processed import ImageProcessor
from .workout_pose_predict import Predictor
import json
import os
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
            # 업로드 성공 후, 업로드 성공 템플릿을 렌더링
            return render(request, 'pose/pose_upload_success.html', {'video_name': uploaded_video.name})
    else:
        form = VideoUploadForm()
    return render(request, 'pose/pose_home.html', {'form': form})

def pose_preview(request):
    if request.method == 'GET':
        video_name = request.GET['video_name']
        processor = ImageProcessor(model_path = 'pose_analysis/models/yolov8n.pt')
        video_dir = "pose_analysis/static/video/" + video_name
        if os.path.exists(video_dir) and os.path.isfile(video_dir):
            processed_image_path = "pose_analysis/static/processed_image/" + video_name
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
            keypoint_json = json.dumps(key_point.tolist())
            return render(request, 'pose/pose_preview.html', {'workout':workout, "keypoint":keypoint_json})
    
    return HttpResponse("예측하지 못함")

def pose_workout_select(request):
    key_point = request.POST['keypoint']
    return render(request, 'pose/pose_workout_select.html',{"keypoint":key_point})

def pose_predict(request):
    keypoint_str = request.POST['keypoint']
    # JSON 문자열을 파이썬 리스트로 변환 후 넘파이 배열로 복원
    keypoint = np.array(json.loads(keypoint_str))
    print(type(keypoint))
    keypoint = keypoint.reshape(-1)
    workout = request.POST['workout']
    predictor = Predictor()
    pose_dict = predictor.pose_predict(keypoint=keypoint, workout=workout)
    return render(request, 'pose/pose_predict.html', {"pose_predict":pose_dict})