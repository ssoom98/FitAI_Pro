from django.shortcuts import render
import joblib
import numpy as np
import os
from django.conf import settings
from django.http import JsonResponse

# 모델 파일의 정확한 경로 설정
model_path = os.path.join(settings.BASE_DIR, 'diet_recommendation', 'model', 'xgboost_meal_plan.pkl')

# 모델 로드
model = joblib.load(model_path)
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
            prediction = model.predict(input_data)
            prediction = prediction.flatten()

            # 예측값 개별 추출
            total_calories = int(prediction[0])
            carbs = round(prediction[1], 1)
            protein = round(prediction[2], 1)
            fat = round(prediction[3], 1)

            print(f"예측 결과 (prediction): {prediction}")
            print(f"prediction type: {type(prediction)}")
            print(f"prediction shape: {prediction.shape}")

        except Exception as e:
            print(f"에러 발생: {str(e)}")
            return render(request, "diet_recommendation/diet_input.html", {
                "error": f"에러 발생: {str(e)}"
            })

    return render(request, "diet_recommendation/diet_input.html", {
        "total_calories": total_calories,
        "carbs": carbs,
        "protein": protein,
        "fat": fat
    })
