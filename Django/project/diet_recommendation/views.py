from django.shortcuts import render
import joblib
import numpy as np
import os
from django.conf import settings
import pandas as pd

# 모델 파일 경로 설정
model_plan_path = os.path.join(settings.BASE_DIR, 'diet_recommendation', 'model', 'xgboost_meal_plan.pkl')

# 모델 로드
model_plan = joblib.load(model_plan_path)

# CSV 파일 로드 (NaN 값 0으로 변환)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "diet_recommendation", "static", "data", "food_nutrition_data.csv")

food_data = pd.read_csv(CSV_PATH)
food_data = food_data.fillna(0)  # NaN 값을 0으로 변경

# 사용자 입력 폼
def diet_input(request):
    total_calories = None
    carbs = None
    protein = None
    fat = None

    if request.method == "POST":
        try:
            gender = int(request.POST["gender"])
            body_type = int(request.POST["body_type"])
            goal = int(request.POST["goal"])

            input_data = np.array([[gender, body_type, goal]], dtype=np.float32)

            # 모델 예측 실행
            prediction = model_plan.predict(input_data).flatten()

            # 예측값 개별 추출
            total_calories = int(prediction[0])
            carbs = round(prediction[1], 1)
            protein = round(prediction[2], 1)
            fat = round(prediction[3], 1)

        except Exception as e:
            return render(request, "diet_recommendation/diet_input.html", {
                "error": f"에러 발생: {str(e)}"
            })

    return render(request, "diet_recommendation/diet_input.html", {
        "total_calories": total_calories,
        "carbs": carbs,
        "protein": protein,
        "fat": fat
    })

# ✅ 랜덤 방식으로 식단 추천 함수
def get_random_meal(target_calories):
    filtered_foods = food_data[food_data["칼로리"] <= target_calories]

    if len(filtered_foods) >= 6:
        recommended_foods = filtered_foods.sample(n=6)
    elif len(filtered_foods) > 0:
        recommended_foods = filtered_foods.sample(n=len(filtered_foods))
    else:
        recommended_foods = food_data.sample(n=6)  # 전체 데이터에서 6개 선택

    return recommended_foods.to_dict(orient="records")

def recommend_meal(request):
    if request.method == "POST":
        # 사용자가 선택한 값 받기
        selected_gender = request.POST.get("gender")
        selected_body_type = request.POST.get("body_type")
        selected_goal = request.POST.get("goal")

        # 선택한 값을 텍스트로 변환
        gender_map = {"0": "남성", "1": "여성"}
        body_type_map = {"0": "사과형", "1": "배형", "2": "모래시계형"}
        goal_map = {"0": "다이어트", "1": "유지", "2": "근육 증가"}

        selected_gender_text = gender_map.get(selected_gender, "")
        selected_body_type_text = body_type_map.get(selected_body_type, "")
        selected_goal_text = goal_map.get(selected_goal, "")

        # 👉 모델 예측 실행
        input_data = np.array([[float(selected_gender), float(selected_body_type), float(selected_goal)]], dtype=np.float32)
        prediction = model_plan.predict(input_data).flatten()

        total_calories = int(prediction[0])
        carbs = round(prediction[1], 1)
        protein = round(prediction[2], 1)
        fat = round(prediction[3], 1)

        # ✅ 랜덤 방식으로 식단 추천
        meal_ratios = {"아침": 0.3, "점심": 0.4, "저녁": 0.3}
        breakfast = get_random_meal(total_calories * meal_ratios["아침"])
        lunch = get_random_meal(total_calories * meal_ratios["점심"])
        dinner = get_random_meal(total_calories * meal_ratios["저녁"])

        return render(request, "diet_recommendation/diet_input.html", {
            "selected_gender_text": selected_gender_text,
            "selected_body_type_text": selected_body_type_text,
            "selected_goal_text": selected_goal_text,
            "total_calories": total_calories,
            "carbs": carbs,
            "protein": protein,
            "fat": fat,
            "breakfast": breakfast,
            "lunch": lunch,
            "dinner": dinner,
        })

    return render(request, "diet_recommendation/diet_input.html")


