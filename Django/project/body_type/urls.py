from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import predict_view

urlpatterns = [
    path('', predict_view, name='predict'),  # 기본 페이지를 업로드 페이지로 설정
]

# 개발 환경에서 미디어 파일 접근 가능하도록 설정
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
