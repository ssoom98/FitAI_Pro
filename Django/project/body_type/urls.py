# body_type/urls.py
from django.urls import path
from .views import predict_body_type

app_name = 'body_type'

urlpatterns = [
    path("", predict_body_type, name="body_type_home"),  # 기본 페이지를 predict_body_type으로 설정
    path("predict/", predict_body_type, name="predict_body_type"),
]
