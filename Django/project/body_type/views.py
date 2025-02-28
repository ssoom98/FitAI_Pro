import os
import numpy as np
import xgboost as xgb
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from .forms import ImageUploadForm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0, VGG16
from sklearn.preprocessing import LabelEncoder
import joblib

# 모델 로드
path1 = os.path.join(settings.BASE_DIR, "body_type/models/xgboost_model.pkl")
path2 = os.path.join(settings.BASE_DIR, "body_type/models/xgboost_model_classification.pkl")
path3 = os.path.join(settings.BASE_DIR, "body_type/models/label_encoder.pkl")
xgb_model = joblib.load(path1)
xgb_classification = joblib.load(path2)
label_encoder = joblib.load(path3)
eff_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
vgg_model = VGG16(weights="imagenet", include_top=False, pooling="avg")


def preprocess_image(img_path):
    """ 이미지를 CNN 모델이 사용할 수 있도록 전처리 """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # 정규화
    return img_array


def predict_body_type(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = form.cleaned_data["image"]

            # 업로드된 파일 저장
            upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            file_path = os.path.join(upload_dir, image_file.name)
            with open(file_path, "wb") as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)

            # CNN 특징 벡터 추출 (1792개 Feature)
            eff_features = eff_model.predict(preprocess_image(file_path)).flatten()
            vgg_features = vgg_model.predict(preprocess_image(file_path)).flatten()
            features = np.concatenate([eff_features, vgg_features]).reshape(1, -1)  # (1, 1792)

            # XGBoost 모델을 사용하여 신체 수치 예측
            dmat_features = xgb.DMatrix(features)  # CNN 특징 벡터 → DMatrix 변환
            y_pred = xgb_model.predict(dmat_features)  # 예측된 신체 수치 (8~21개)

            # 🔥 기존 특징 벡터(1792개) + 예측된 신체 수치(8~21개) 결합
            combined_features = np.concatenate([features, y_pred], axis=1)  # 현재 (1, 1800)

            # 🔥 부족한 Feature(0 또는 평균값) 추가하여 1813개로 맞추기
            num_missing_features = 1813 - combined_features.shape[1]  # 부족한 Feature 개수 계산
            additional_features = np.zeros((1, num_missing_features))  # 부족한 Feature 0으로 채우기
            combined_features = np.concatenate([combined_features, additional_features], axis=1)  # (1, 1813)

            # 🔥 체형 분류 모델 실행
            body_shape_index = xgb_classification.predict(combined_features).flatten()[0].item()
            body_shape_label = label_encoder.inverse_transform([int(body_shape_index)])[0]

            # 예측된 값 출력 (디버깅용)
            print(f"예측된 신체 데이터(y_pred): {y_pred}")
            print(f"최종 입력 벡터 크기: {combined_features.shape}")  # (1, 1813)
            print(f"예측된 체형(body_shape_label): {body_shape_label}")

            return render(request, "body_type/result.html", {
                "y_pred": y_pred.tolist()[0],  # 예측된 신체 데이터
                "body_shape": body_shape_label,  # 예측된 체형
                "image_url": settings.MEDIA_URL + "uploads/" + image_file.name
            })

    form = ImageUploadForm()
    return render(request, "body_type/upload.html", {"form": form})
