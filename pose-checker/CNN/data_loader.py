import os
import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class DatasetLoader:
    """
    cv2로 이미지 로드 및 x, y로 분해서 return
    """
    def find_paths_with_keyword(self, root_dir, keyword):
        matched_paths = []
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # 디렉터리와 파일명에서 keyword가 포함된 경우 저장
            for name in dirnames + filenames:
                if keyword in name:
                    matched_paths.append(os.path.join(dirpath, name))
        
        return matched_paths
    
    def random_sampling_data(self, df, categorys, size):
        random_sample_df = pd.DataFrame([])
        for category in df[categorys].value_counts().index:
            temp = df[df[categorys] == category].sample(n=size)
            random_sample_df = pd.concat([temp, random_sample_df])
        return random_sample_df.sample(frac=1).reset_index(drop=True)
    
    def pad_images(self, image_list, target_size=(64, 64, 1)):
        """이미지 리스트를 패딩하여 동일한 크기로 맞춤 (제로 패딩 적용)"""
        max_length = max(len(img_list) for img_list in image_list)  # 가장 긴 이미지 리스트 길이
        padded_list = []
        
        for img_list in image_list:
            padded = img_list[:]  # 원본 리스트 복사
            while len(padded) < max_length:  # 부족한 부분을 0으로 채움
                padded.append(np.zeros(target_size, dtype=np.uint8))  
            
            padded_list.append(np.array(padded, dtype=np.uint8))
        
        return np.array(padded_list, dtype=np.float32) / 255.0  # 정규화 후 반환

    def sub_load_img(self, df, img_paths, category_paths, split_category):
    # df.category_paths.value_counts().index
        category_mapping = {
            '바벨/덤벨': 'Barbell_Dumbbell/',
            '맨몸 운동': 'Bodyweight/',
            '기구': 'Equipment/'
        }
        img_base_dir = 'D:/KHH/team_project/img/'
        data_X = []
        data_Y = []
        for category_path, img_path, category in zip(df[category_paths], df[img_paths], df[split_category]):
            category_dir = category_mapping.get(category_path)
            img_paths = self.find_paths_with_keyword(img_base_dir+category_dir,img_path)
            print(img_paths)
            img_list = []
            if len(img_paths) > 0:
                for img_path in img_paths[3:-3]:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (64,64))
                        img_list.append(img.reshape(64,64,1))
                    print(img_path, '로드 완료')
                data_X.append(img_list)
                data_Y.append(category)
            else:
                print('없는 사진입니다')
        return self.pad_images(data_X), np.array(data_Y)
    
    def main_load_img(self, df, img_paths, category_paths):
    # df.category_paths.value_counts().index
        category_mapping = {
            '바벨/덤벨': 'Barbell_Dumbbell/',
            '맨몸 운동': 'Bodyweight/',
            '기구': 'Equipment/'
        }
        img_base_dir = 'D:/KHH/team_project/img/'
        data_X = []
        data_Y = []
        for category_path, img_path in zip(df[category_paths], df[img_paths]):
            category_dir = category_mapping.get(category_path)
            img_paths = self.find_paths_with_keyword(img_base_dir+category_dir,img_path)
            print(img_paths)
            img_list = []
            if len(img_paths) > 0:
                for img_path in img_paths[3:-3]:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (64,64))
                        img_list.append(img.reshape(64,64,1))
                    print(img_path, '로드 완료')
                data_X.append(img_list)
                data_Y.append(category_path)
            else:
                print('없는 사진입니다')
        return self.pad_images(data_X), np.array(data_Y)

    def one_hot_encode(self, y):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(y)
        onehot_encoder = OneHotEncoder(sparse_output=False)  # sparse=False → numpy 배열 반환
        integer_encoded = integer_encoded.reshape(-1, 1)  # 2D 배열로 변환해야 함
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        return onehot_encoded

    def train_test_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        return X_train, X_test, y_train, y_test
