from django.urls import path
from . import views

urlpatterns = [
    path('', views.diet_home, name='diet_home'),
]
