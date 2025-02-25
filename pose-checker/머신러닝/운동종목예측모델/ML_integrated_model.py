import pandas as pd
import os
import numpy as np
import json
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

base_path = 'D:/KHH/team_project/ML_data/'
save_path = "D:/KHH/team_project/proseced_logic/ML_integrated_model/"
eval_save_path = "D:/KHH/team_project/proseced_logic/ML_integrated_model/model_evaluation/"

def convert_to_list(x):
    # 정규 표현식을 사용하여 숫자만 추출
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", x)
    # 추출된 문자열을 실수(float)로 변환
    return list(map(float, numbers))

ML_df = pd.DataFrame([])
for csv_name in os.listdir(base_path):
    if not('._' in csv_name or 'ML_X_data.csv' in csv_name or 'ML_Y_data.csv' in csv_name or 'ML_proseced_data.csv' in csv_name) : # 임시파일 거르는 용도
        print(csv_name + '병합중')
        ML_df = pd.concat([pd.read_csv(base_path + csv_name), ML_df], axis=0, ignore_index=True)

ML_df.drop(columns=['babel_04','babel_03','babel_02','babel_01'], inplace=True)
nan_index = ML_df[ML_df['keypoints'].apply(lambda x: x is None or (isinstance(x, float) and np.isnan(x)))].index
ML_df.drop(index=nan_index, inplace=True)
ML_df.reset_index(drop=True, inplace=True)

ML_df['folder'] = ML_df['img_key'].apply(lambda x: "/".join(x.split("/")[:4]))
folder_workout_df = ML_df.groupby("folder")["workout"].unique().reset_index()

train_df = pd.concat([pd.read_csv('D:/KHH/team_project/ML_data/ML_X_data.csv'),folder_workout_df['workout']], axis=1, ignore_index=True)

train_df.columns = ['folder', 'key_point', 'workout']

train_df['workout'] = train_df['workout'].map(lambda x : x[0])
train_df['key_point'] = train_df['key_point'].apply(convert_to_list).to_list()

train_df = train_df.sample(frac=1, ignore_index=True, replace=True)

X = np.array(train_df['key_point'].to_list())
y = np.array(train_df['workout'])

X_padded = pad_sequences(X, padding='post', dtype='float32')

le = LabelEncoder()
y_data = le.fit_transform(y)
label_mapping = {str(i): label for i, label in enumerate(le.classes_)}
with open(save_path+"label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, ensure_ascii=False, indent=4)

X_train, X_test, Y_train, Y_test = train_test_split(X_padded, y_data, test_size=0.3, shuffle=True)

xgb_model = XGBClassifier(eval_metric="logloss")
xgb_model.fit(X_train, Y_train)

lgbm_model = LGBMClassifier()
lgbm_model.fit(X_train, Y_train)

xgb_pred = xgb_model.predict(X_test)
lgbm_pred = lgbm_model.predict(X_test)

xgb_accuracy = accuracy_score(Y_test, xgb_pred)
xgb_f1 = f1_score(Y_test, xgb_pred, average='weighted')
xgb_precision = precision_score(Y_test, xgb_pred, average='weighted')
xgb_recall = recall_score(Y_test, xgb_pred, average='weighted')

lgbm_accuracy = accuracy_score(Y_test, lgbm_pred)
lgbm_f1 = f1_score(Y_test, lgbm_pred, average='weighted')
lgbm_precision = precision_score(Y_test, lgbm_pred, average='weighted')
lgbm_recall = recall_score(Y_test, lgbm_pred, average='weighted')

results = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM'],
    'Accuracy': [xgb_accuracy, lgbm_accuracy],
    'F1 Score': [xgb_f1, lgbm_f1],
    'Precision': [xgb_precision, lgbm_precision],
    'Recall': [xgb_recall, lgbm_recall]
})

results.to_csv(eval_save_path+"integrated_for_use_model_evaluation_results.csv", index=False)
print("모델 평가 결과가 'model_for_use_evaluation_results.csv' 파일에 저장되었습니다.")

crosstab_data = pd.crosstab(Y_test, xgb_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(crosstab_data, annot=True, fmt="d", cmap="Blues")
plt.title("Crosstab XGB model")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# 플롯을 파일로 저장 (dpi: 해상도, bbox_inches: 레이아웃 여백 설정)
plt.savefig(save_path+"xgb_crosstab_plot.png", dpi=300, bbox_inches='tight')
print("crosstap이 저장되었습니다")
plt.show()

crosstab_data = pd.crosstab(Y_test, lgbm_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(crosstab_data, annot=True, fmt="d", cmap="Blues")
plt.title("Crosstab LGBM model")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(save_path+"lgbm_crosstab_plot.png", dpi=300, bbox_inches='tight')
print("crosstap이 저장되었습니다")
plt.show()

joblib.dump(xgb_model, save_path+"integrated_for_use_xgb_model.pkl")
joblib.dump(lgbm_model, save_path+"integrated_for_use_lgbm_model.pkl")