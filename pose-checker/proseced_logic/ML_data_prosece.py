import pandas as pd
import os
import numpy as np
import ast

base_path = 'D:/KHH/team_project/ML_data/'


# 조건을 변환하는 함수 정의
def convert_conditions(cond_str):
    try:
        # 문자열을 리스트로 변환
        cond_list = ast.literal_eval(cond_str) if isinstance(cond_str, str) else cond_str
        
        # "value" 값이 True면 1, False면 0으로 변환
        binary_values = [1 if cond["value"] else 0 for cond in cond_list]
        descriptions = [cond["condition"] for cond in cond_list]
        return pd.Series([binary_values, descriptions])
    except:
        return pd.Series([[], []])  # 변환 실패 시 빈 리스트 반환
print("convert_conditions 함수 생성 완료")

ML_df = pd.DataFrame([])
for csv_name in os.listdir(base_path):
    print(csv_name + '병합중')
    if not('._' in csv_name) : # 임시파일 거르는 용도
        ML_df = pd.concat([pd.read_csv(base_path + csv_name), ML_df], axis=0, ignore_index=True)
print("ML_df생성완료")

ML_df.drop(columns=['babel_04','babel_03','babel_02','babel_01'], inplace=True)
nan_index = ML_df[ML_df['keypoints'].apply(lambda x: x is None or (isinstance(x, float) and np.isnan(x)))].index
ML_df.drop(index=nan_index, inplace=True)
ML_df.reset_index(drop=True, inplace=True)
print("ML_df의 Nan행 삭제후 index초기화 완료")

# apply(pd.Series)를 사용하여 두 개의 컬럼에 분리 저장
ML_df[['conditions_binary', 'conditions_description']] = ML_df['conditions'].apply(convert_conditions)
print("ML_df의 conditions_binary, conditions_description행 추가 완료")
ML_df['folder'] = ML_df['img_key'].apply(lambda x: "/".join(x.split("/")[:4])) # 이미지 파일 경로기준으로 데이터 프레임을 묶기 위해
print("ML_df의 folder행 추가 완료")

ML_X_data = ML_df.groupby('folder')['keypoints'].sum().reset_index()
print("ML_X_data 생성 완료")

condition_list = []
for row in ML_X_data.itertuples(index=False):
    print(row.folder)
    matched_rows = ML_df[ML_df["folder"] == row.folder]
    
    # 만약 일치하는 데이터가 있다면 conditions_binary 값 추가
    if not matched_rows.empty:
        condition_list.append(matched_rows["conditions_binary"].iloc[0])  # 첫 번째 값 추가
        print('추가완료')

pd.DataFrame(np.array(condition_list)).to_csv(base_path+"ML_Y_data.csv", index=False)
print("ML_Y_data 저장 완료")
ML_X_data.to_csv(base_path+"ML_X_data.csv", index=False)
print("ML_X_data 저장 완료")