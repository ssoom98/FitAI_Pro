from django.shortcuts import render
from django.http import HttpResponse
from .forms import VideoUploadForm

def pose_home(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_video = request.FILES['video']
            # 저장 경로: static/uploads/videos 폴더
            save_path = os.path.join(settings.BASE_DIR, 'static', 'videos', uploaded_video.name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb+') as destination:
                for chunk in uploaded_video.chunks():
                    destination.write(chunk)
            return HttpResponse("동영상 업로드 성공!")
    else:
        form = VideoUploadForm()
    return render(request, 'pose/pose_home.html', {'form': form})