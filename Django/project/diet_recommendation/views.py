from django.shortcuts import render
from django.http import HttpResponse

def diet_home(request):
    return HttpResponse("식단 추천 페이지.")
