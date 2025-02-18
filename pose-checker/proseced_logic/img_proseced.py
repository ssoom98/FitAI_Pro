import os
import json
import pandas as pd
import tarfile
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from data_proseced import ImageProcessor

processor = ImageProcessor()
img_path = 'E:/project_data/013.피트니스자세/1.Training/원시데이터/'
CNN_img_save_path = 'D:/KHH/team_project/img/'
default_path = 'E:/project_data/unziped_file/img_temp/'
folder_dict = {'바벨/덤벨':'Barbell_Dumbbell/', '맨몸 운동':'Bodyweight/', '기구':'Equipment/'}
train_data = pd.read_csv('D:/KHH/team_project/json_data/Barbell_Dumbbell.csv')
train_data = pd.concat([train_data, pd.read_csv('D:/KHH/team_project/json_data/Bodyweight.csv'), pd.read_csv('D:/KHH/team_project/json_data/Equipment.csv')], axis=0, ignore_index=True)

for tarfile in os.listdir(img_path)[18:]:
    if 'tar' in tarfile:
        ML_data_list = []
        file_name = tarfile.split('.')[0] + '.csv'
        print(img_path+tarfile)
        print(file_name)
        processor.unziped_tar(img_path+tarfile) # 압축해제
        print(tarfile,'압축 해제중')

        for row in train_data.itertuples():
            img_path_ = default_path + row.img_key
            if os.path.exists(img_path_):
                print("파일이 존재합니다!")
                img = cv2.imread(img_path_)
                keypoints = processor.detect_pose(img).reshape(-1).tolist()
                keypoints_dict = {'keypoints':keypoints, # 각 관절마다 포인트 좌표를 받은 후 ML학습용 데이터로 사용하기 위해 csv파일로 저장
                                   'img_key':row.img_key,
                                   'type':row.type,
                                   'workout':row.workout,
                                   'conditions':row.conditions,
                                   'description':row.description}
                ML_data_list.append(keypoints_dict)
                safe_file_name = re.sub(r'[\\/:"*?<>|]', "@", row.img_key)
                cropped_img = processor.detect_person_yolov8_square_crop(img_path_)
                if cropped_img is not None:
                    cv2.imwrite(CNN_img_save_path + folder_dict[row.type] + safe_file_name, cropped_img) # 사람 객체를 탐지하여 256 * 256 크기의 흑백사진으로 변환 후 CNN학습을 위해 저장
                else:
                    print(f" 이미지 크롭 실패: {row.img_key}")
            else:
                print("파일이 존재하지 않습니다.")
        
        for del_path in os.listdir('E:/project_data/unziped_file/img_temp/'): # 저장 공간 확보를 위해 위에서 풀었던 압축파일 삭제
            processor.remove_file('E:/project_data/unziped_file/img_temp/'+del_path)
            print(del_path, '폴더 또는 파일 삭제완료')

        ML_data = pd.DataFrame(ML_data_list)
        ML_data.to_csv(r"D:\KHH\team_project\ML_data/" + file_name, index=False)