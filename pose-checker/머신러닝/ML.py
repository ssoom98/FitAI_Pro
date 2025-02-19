import pandas as pd
import os
import numpy as np
import ast
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

def convert_to_list(x):
    # 정규 표현식을 사용하여 숫자만 추출
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", x)
    # 추출된 문자열을 실수(float)로 변환
    return list(map(float, numbers))

base_path = 'D:/KHH/team_project/ML_data/'

ML_df = pd.DataFrame([])
for csv_name in os.listdir(base_path):
    if not('._' in csv_name or 'ML_X_data.csv' in csv_name or 'ML_Y_data.csv' in csv_name or 'ML_proseced_data.csv' in csv_name) : # 임시파일 거르는 용도
        print(csv_name + '병합중')
        ML_df = pd.concat([pd.read_csv(base_path + csv_name), ML_df], axis=0, ignore_index=True)

ML_df.drop(columns=['babel_04','babel_03','babel_02','babel_01'], inplace=True)
nan_index = ML_df[ML_df['keypoints'].apply(lambda x: x is None or (isinstance(x, float) and np.isnan(x)))].index
ML_df.drop(index=nan_index, inplace=True)
ML_df.reset_index(drop=True, inplace=True)

ML_df['folder'] = ML_df['img_key'].apply(lambda x: "/".join(x.split("/")[:4])) # 이미지 파일 경로기준으로 데이터 프레임을 묶기 위해
folder_workout_df = ML_df.groupby("folder")["workout"].unique().reset_index()

ML_X_data = pd.read_csv('D:/KHH/team_project/ML_data/ML_X_data.csv')
ML_Y_data = pd.read_csv('D:/KHH/team_project/ML_data/ML_Y_data.csv')
train_ML_data = pd.concat([ML_X_data, ML_Y_data], axis=1)

train_ML_data['workout'] = folder_workout_df['workout']
train_ML_data['workout'] = train_ML_data['workout'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, (list, np.ndarray)) else str(x))
train_ML_data['keypoints'] = train_ML_data['keypoints'].apply(convert_to_list)
train_ML_data['0'] = train_ML_data.loc[:, '0'].map(ast.literal_eval)

for workout in train_ML_data['workout'].value_counts().index:
    # numpy 배열로 변환 (dtype=object)
    X_data = np.array(train_ML_data.loc[train_ML_data['workout'] == workout, 'keypoints'].tolist(), dtype=object)
    Y_data = np.array(train_ML_data.loc[train_ML_data['workout'] == workout, '0'].tolist())

    # 패딩 적용 (모든 샘플을 동일한 길이로 맞춤)
    X_data_padded = pad_sequences(X_data, padding='post', dtype='float32')

    # 데이터 셔플 및 7:3 비율로 나누기
    X_train, X_test, Y_train, Y_test = train_test_split(X_data_padded, Y_data, test_size=0.3, shuffle=True)
    print(workout, X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    xgb_model = MultiOutputClassifier(XGBClassifier(eval_metric="logloss"))
    # 모델 학습
    xgb_model.fit(X_train, Y_train)
    # 예측 수행
    Y_pred_xgb = xgb_model.predict(X_test)
    # 정확도 평가 (각 라벨별 정확도 평균)
    accuracy_xgb = np.mean([accuracy_score(Y_test[:, i], Y_pred_xgb[:, i]) for i in range(Y_test.shape[1])])
    print(f"XGBoost Multi-Label Accuracy: {accuracy_xgb:.4f}")
    f1_xgb = f1_score(Y_test, Y_pred_xgb, average='macro')

    # LightGBM을 MultiOutputClassifier로 래핑하여 다중 라벨 분류 수행
    lgbm_model = MultiOutputClassifier(LGBMClassifier())
    # 모델 학습
    lgbm_model.fit(X_train, Y_train, feature_name="auto")
    # 예측 수행
    Y_pred_lgbm = lgbm_model.predict(X_test)
    # 정확도 평가
    accuracy_lgbm = np.mean([accuracy_score(Y_test[:, i], Y_pred_lgbm[:, i]) for i in range(Y_test.shape[1])])
    print(f"LightGBM Multi-Label Accuracy: {accuracy_lgbm:.4f}")
    f1_lgbm = f1_score(Y_test, Y_pred_lgbm, average='macro')

    if accuracy_lgbm <= accuracy_xgb:
        print(f"XGB모델을 저장합니다 f1 : {f1_xgb}, acc : {accuracy_xgb}")
        joblib.dump(xgb_model, f"ML_model/{workout}_{accuracy_xgb}_{f1_xgb}_xgboost_model.pkl")

    elif accuracy_xgb <= accuracy_lgbm:
        print(f"LGBM모델을 저장합니다 f1 : {f1_lgbm}, acc : {accuracy_lgbm}")
        joblib.dump(lgbm_model, f"ML_model/{workout}_{accuracy_lgbm}_{f1_lgbm}_lgbm_model.pkl")