from django.shortcuts import render
import joblib
import numpy as np
import os
from django.conf import settings

model_plan_path = os.path.join(settings.BASE_DIR, 'diet_recommendation', 'model', 'xgboost_meal_plan.pkl')
model_plan = joblib.load(model_plan_path)

# 체형 + 목표별 추천 음식 리스트 (고정)
fixed_meals = {
    ("사과형", "다이어트"): {
        "아침": ["현미밥", "닭가슴살", "나물"],
        "점심": ["고구마", "삶은 달걀", "샐러드"],
        "저녁": ["연어구이", "아보카도", "채소볶음"]
    },
    ("사과형", "유지"): {
        "아침": ["오트밀", "우유", "견과류"],
        "점심": ["생선구이", "잡곡밥", "채소"],
        "저녁": ["두부 샐러드", "달걀", "견과류"]
    },
    ("사과형", "근육 증가"): {
        "아침": ["스크램블 에그", "호밀빵", "바나나"],
        "점심": ["닭가슴살", "고구마", "브로콜리"],
        "저녁": ["소고기 스테이크", "현미밥", "삶은 계란"]
    },

    ("배형", "다이어트"): {
        "아침": ["두부 샐러드", "아몬드", "요거트"],
        "점심": ["생선구이", "야채볶음", "잡곡밥"],
        "저녁": ["닭가슴살", "고구마", "양상추 샐러드"]
    },
    ("배형", "유지"): {
        "아침": ["바나나", "그릭요거트", "견과류"],
        "점심": ["현미밥", "불고기", "나물"],
        "저녁": ["삶은 계란", "아보카도", "닭가슴살 샐러드"]
    },
    ("배형", "근육 증가"): {
        "아침": ["삶은 계란", "바나나", "우유"],
        "점심": ["돼지고기 등심", "현미밥", "채소"],
        "저녁": ["닭가슴살", "고구마", "아몬드"]
    },

    ("모래시계형", "다이어트"): {
        "아침": ["귀리죽", "삶은 달걀", "견과류"],
        "점심": ["생선구이", "브로콜리", "잡곡밥"],
        "저녁": ["닭가슴살", "아보카도 샐러드", "올리브유"]
    },
    ("모래시계형", "유지"): {
        "아침": ["호밀빵", "스크램블 에그", "아몬드"],
        "점심": ["불고기", "잡곡밥", "채소볶음"],
        "저녁": ["연어 샐러드", "견과류", "두유"]
    },
    ("모래시계형", "근육 증가"): {
        "아침": ["오트밀", "우유", "삶은 계란"],
        "점심": ["닭가슴살", "고구마", "채소"],
        "저녁": ["소고기", "현미밥", "견과류"]
    },

    ("엉덩이형", "다이어트"): {
        "아침": ["오트밀", "아몬드", "저지방 우유"],
        "점심": ["닭가슴살", "고구마", "나물"],
        "저녁": ["생선회", "샐러드", "올리브유"]
    },
    ("엉덩이형", "유지"): {
        "아침": ["호밀빵", "치즈", "바나나"],
        "점심": ["돼지고기 구이", "현미밥", "채소"],
        "저녁": ["두부 샐러드", "삶은 계란", "견과류"]
    },
    ("엉덩이형", "근육 증가"): {
        "아침": ["바나나", "그릭요거트", "아몬드"],
        "점심": ["소고기", "현미밥", "채소"],
        "저녁": ["닭가슴살", "고구마", "아보카도"]
    },

    ("상체형", "다이어트"): {
        "아침": ["두부 샐러드", "아몬드", "저지방 우유"],
        "점심": ["현미밥", "생선구이", "브로콜리"],
        "저녁": ["닭가슴살 샐러드", "고구마", "올리브유"]
    },
    ("상체형", "유지"): {
        "아침": ["호밀빵", "달걀", "견과류"],
        "점심": ["불고기", "잡곡밥", "나물"],
        "저녁": ["생선 샐러드", "두유", "아보카도"]
    },
    ("상체형", "근육 증가"): {
        "아침": ["삶은 계란", "바나나", "우유"],
        "점심": ["닭가슴살", "고구마", "채소"],
        "저녁": ["소고기", "현미밥", "견과류"]
    },

    ("하체형", "다이어트"): {
        "아침": ["오트밀", "아몬드", "저지방 우유"],
        "점심": ["닭가슴살", "고구마", "샐러드"],
        "저녁": ["연어구이", "브로콜리", "잡곡밥"]
    },
    ("하체형", "유지"): {
        "아침": ["스크램블 에그", "호밀빵", "바나나"],
        "점심": ["돼지고기 구이", "현미밥", "채소볶음"],
        "저녁": ["두부 샐러드", "아몬드", "삶은 달걀"]
    },
    ("하체형", "근육 증가"): {
        "아침": ["삶은 계란", "바나나", "우유"],
        "점심": ["닭가슴살", "현미밥", "채소"],
        "저녁": ["소고기", "고구마", "견과류"]
    },

    ("표준체형", "다이어트"): {
        "아침": ["오트밀", "아몬드", "그릭요거트"],
        "점심": ["현미밥", "생선구이", "채소"],
        "저녁": ["닭가슴살 샐러드", "고구마", "올리브유"]
    },
    ("표준체형", "유지"): {
        "아침": ["호밀빵", "스크램블 에그", "바나나"],
        "점심": ["불고기", "잡곡밥", "나물"],
        "저녁": ["연어 샐러드", "견과류", "두유"]
    },
    ("표준체형", "근육 증가"): {
        "아침": ["삶은 계란", "바나나", "우유"],
        "점심": ["닭가슴살", "고구마", "채소"],
        "저녁": ["소고기", "현미밥", "견과류"]
    }
}

def diet_input(request):
    total_calories = None
    carbs = None
    protein = None
    fat = None
    selected_gender_text = None
    selected_body_type_text = None
    selected_goal_text = None
    breakfast = lunch = dinner = None

    if request.method == "POST":
        try:
            gender = int(request.POST["gender"])
            body_type = int(request.POST["body_type"])
            goal = int(request.POST["goal"])

            input_data = np.array([[gender, body_type, goal]], dtype=np.float32)

            prediction = model_plan.predict(input_data).flatten()

            total_calories = int(prediction[0])
            carbs = round(prediction[1], 1)
            protein = round(prediction[2], 1)
            fat = round(prediction[3], 1)

            gender_map = {0: "남성", 1: "여성"}
            body_type_map = {0: "사과형", 1: "배형", 2: "모래시계형", 3: "엉덩이형", 4: "상체형", 5: "하체형", 6: "표준체형"}
            goal_map = {0: "다이어트", 1: "유지", 2: "근육 증가"}

            selected_gender_text = gender_map.get(gender, "")
            selected_body_type_text = body_type_map.get(body_type, "")
            selected_goal_text = goal_map.get(goal, "")

            meal_key = (selected_body_type_text, selected_goal_text)
            if meal_key in fixed_meals:
                breakfast = fixed_meals[meal_key]["아침"]
                lunch = fixed_meals[meal_key]["점심"]
                dinner = fixed_meals[meal_key]["저녁"]
            else:
                breakfast = lunch = dinner = ["식단 정보 없음"]

        except Exception as e:
            return render(request, "diet_recommendation/diet_input.html", {
                "error": f"에러 발생: {str(e)}"
            })

    return render(request, "diet_recommendation/diet_input.html", {
        "total_calories": total_calories,
        "carbs": carbs,
        "protein": protein,
        "fat": fat,
        "selected_gender_text": selected_gender_text,
        "selected_body_type_text": selected_body_type_text,
        "selected_goal_text": selected_goal_text,
        "breakfast": breakfast,
        "lunch": lunch,
        "dinner": dinner
    })
