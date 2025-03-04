from django.urls import path
from . import views

app_name = 'diet'

urlpatterns = [
    path('', views.diet_input, name='diet_input'),
]
