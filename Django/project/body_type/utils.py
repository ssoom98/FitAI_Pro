import os
import pickle
import numpy as np
import xgboost as xgb
import cv2
import pandas as pd
from django.conf import settings
from tensorflow.keras.applications import EfficientNetB0, VGG16
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

# 모델 파일 경로 설정
MODEL_PATH = os.path.join(settings.BASE_DIR, 'body_type/models/xgboost_model.pkl')
CLASSIFICATION_MODEL_PATH = os.path.join(settings.BASE_DIR, 'body_type/models/xgboost_model_classification.pkl')
FEATURE_VECTOR_PATH = os.path.join(settings.BASE_DIR, 'body_type/models/CNN_특징벡터_신체데이터.csv')

# CNN 특징 벡터 데이터 로드
cnn_features = pd.read_csv(FEATURE_VECTOR_PATH)

# 문자열 데이터를 실수형으로 변환
def preprocess_cnn_features(df):
    df = df.select_dtypes(include=[np.number])  # 숫자형 데이터만 선택
    return df

cnn_features = preprocess_cnn_features(cnn_features)

# 모델 로드 함수
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# 신체 둘레 예측 모델 및 체형 분류 모델 로드
body_measurement_model = load_model(MODEL_PATH)
body_classification_model = load_model(CLASSIFICATION_MODEL_PATH)

# 신체 둘레 예측 함수
def predict_body_measurements(features):
    """
    features: 1792차원 특징 벡터 (numpy 배열)
    return: 예측된 신체 둘레 값
    """
    features = np.array(features).reshape(1, -1)  # XGBoost 모델 입력 형식 맞추기
    expected_features = 1792  # 모델이 기대하는 feature 개수
    if features.shape[1] > expected_features:
        features = features[:, :expected_features]  # 필요한 개수만 선택
    predictions = body_measurement_model.predict(features)  # XGBoost 예측 (DMatrix 제거)
    return predictions[0]  # 단일 예측값 반환

# 체형 분류 함수
def classify_body_type(measurements):
    """
    measurements: 예측된 신체 둘레 값
    return: 체형 분류 결과
    """
    measurements = np.array(measurements).reshape(1, -1)  # XGBoost 모델 입력 형식 맞추기
    body_type = body_classification_model.predict(measurements)  # XGBoost 예측 (DMatrix 제거)
    return body_type[0]  # 분류 결과 반환

def preprocess_image(image_path):
    """
    CNN 특징 벡터 데이터셋을 기반으로 1792차원 특징 벡터를 추출하는 함수
    """
    random_index = np.random.randint(0, len(cnn_features))
    selected_features = cnn_features.iloc[random_index, :1792].values.astype(np.float32)  # 1792차원 선택 및 변환
    return selected_features  # 1792차원 벡터 반환
