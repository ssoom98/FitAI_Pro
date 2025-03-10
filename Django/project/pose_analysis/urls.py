from django.urls import path
from . import views  # pose_analysis/views.py 가져오기

app_name = 'pose'
urlpatterns = [
    path('', views.pose_home, name='home'),  # 운동 자세 분석 메인 페이지
    path('pose_preview/', views.pose_preview, name='preview'),
    path('pose_workout_select/', views.pose_workout_select, name='workout_select'),
    path('pose_predict/', views.pose_predict, name='predict'),
]
