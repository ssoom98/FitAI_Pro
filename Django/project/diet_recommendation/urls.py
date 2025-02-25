from django.urls import path
from . import views

urlpatterns = [
    path('', views.diet_input, name='diet_input'),
]
