import pandas as pd
import os
import numpy as np
import ast
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

base_path = 'D:/KHH/team_project/ML_data/'

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
X_train, X_test, Y_train, Y_test = train_test_split(X_padded, y_data, test_size=0.3, shuffle=True)

xgb_model = XGBClassifier(eval_metric="logloss")
xgb_model.fit(X_train, Y_train)

lgbm_model = LGBMClassifier()
lgbm_model.fit(X_train, Y_train)

xgb_pred = xgb_model.predict(X_test)
lgbm_pred = lgbm_model.predict(X_test)
xgb_accuracy = accuracy_score(Y_test, xgb_pred)
xgb_f1 = f1_score(Y_test, xgb_pred, average='weighted')

lgbm_accuracy = accuracy_score(Y_test, lgbm_pred)
lgbm_f1 = f1_score(Y_test, lgbm_pred, average='weighted')

results = pd.DataFrame({
'Model': ['XGBoost', 'LightGBM'],
'Accuracy': [xgb_accuracy, lgbm_accuracy],
'F1 Score': [xgb_f1, lgbm_f1]
})

save_path = "D:/KHH/team_project/proseced_logic/ML_integrated_model/"
eval_save_path = "D:/KHH/team_project/proseced_logic/ML_integrated_model/model_evaluation/"
results.to_csv(eval_save_path+"integrated_model_evaluation_results.csv", index=False)
print("모델 평가 결과가 'model_evaluation_results.csv' 파일에 저장되었습니다.")

joblib.dump(xgb_model, save_path+"integrated_xgb_model.pkl")
joblib.dump(lgbm_model, save_path+"integrated_lgbm_model.pkl")