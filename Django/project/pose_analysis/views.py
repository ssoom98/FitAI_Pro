from django.shortcuts import render
from django.http import HttpResponse
from .forms import VideoUploadForm
from .Video_to_img_processed import ImageProcessor
import os

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
        processer = ImageProcessor(model_path = 'pose_analysis/models/yolov8n.pt')
        video_dir = "pose_analysis/static/video/" + video_name
        if os.path.exists(video_dir) and os.path.isfile(video_dir):
            processed_image_path = "pose_analysis/static/processed_image/" + video_name
            os.makedirs(processed_image_path, exist_ok=True)
            processer.extract_middle_32_frames(video_dir, processed_image_path)
            return HttpResponse("사진 저장 성공!")

