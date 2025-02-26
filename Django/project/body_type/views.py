from django.shortcuts import render
from django.http import HttpResponse

def body_home(request):
    return HttpResponse("체형 예측 페이지")
