from django.urls import path
from . import views  # pose_analysis/views.py 가져오기

urlpatterns = [
    path('', views.pose_home, name='pose_home'),  # 운동 자세 분석 메인 페이지
]
