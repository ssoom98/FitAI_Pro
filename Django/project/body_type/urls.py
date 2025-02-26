from django.urls import path
from . import views

urlpatterns = [
    path('', views.body_home, name='body_home'),
]
