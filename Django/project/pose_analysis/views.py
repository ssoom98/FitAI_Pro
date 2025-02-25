from django.shortcuts import render
from django.http import HttpResponse

def pose_home(request):
    return HttpResponse("운동 자세 페이지")
