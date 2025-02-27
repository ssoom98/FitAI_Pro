from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from .utils import preprocess_image, predict_body_measurements, classify_body_type
import os

def predict_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']

        # 이미지 저장 경로
        file_path = os.path.join('media/uploads', image_file.name)
        default_storage.save(file_path, ContentFile(image_file.read()))

        # EfficientNetB0 + VGG16 기반 특징 벡터 추출 (2304차원)
        features = preprocess_image(file_path)

        # 신체 둘레 예측
        body_measurements = predict_body_measurements(features)

        # 체형 분류
        body_type = classify_body_type(body_measurements)

        # 결과 반환
        context = {
            'body_measurements': body_measurements.tolist(),
            'body_type': body_type,
            'image_url': settings.MEDIA_URL + 'uploads/' + image_file.name
        }
        return render(request, 'result.html', context)

    return render(request, 'upload.html')
