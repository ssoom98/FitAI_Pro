import os
import numpy as np
import json
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.applications import NASNetMobile, ResNet101, EfficientNetB3, EfficientNetB7
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, ConvLSTM2D, TimeDistributed, BatchNormalization,
    Conv2D, Dense, Flatten, GlobalAveragePooling2D, 
    GlobalAveragePooling1D, Dropout, Reshape
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score


class DatasetLoader:
    """
    데이터 로딩 및 전처리 클래스 (Train/Test 자동 분할 포함)
    """
    def __init__(self, base_dir, img_base_dir, img_size=(64, 64), test_split=0.2):
        """
        초기화
        :param base_dir: JSON 데이터셋이 있는 폴더
        :param img_base_dir: 이미지 파일이 저장된 폴더
        :param img_size: 이미지 크기 (기본값: 64x64)
        :param test_split: Train/Test 데이터 분할 비율 (기본값: 0.2, 즉 80% Train / 20% Test)
        """
        self.base_dir = base_dir
        self.img_base_dir = img_base_dir
        self.img_size = img_size
        self.test_split = test_split  # Train/Test 비율 설정

    def stratified_sampling_and_split(self, df, category_column, sample_size=None):
        """
        각 카테고리에서 지정된 개수만큼 층화추출 (자동 샘플링 가능)
        :param df: 입력 데이터프레임
        :param category_column: 카테고리 컬럼명 (예: 'type')
        :param sample_size: 샘플링할 개수 (None이면 전체 사용)
        """
        sampled_dfs = []
        for category in df[category_column].unique():
            category_df = df[df[category_column] == category]
            
            # sample_size가 None이면 전체 데이터를 사용
            if sample_size is None:
                sampled_df = category_df
            else:
                sampled_df = category_df.sample(n=min(sample_size, len(category_df)))
            
            sampled_dfs.append(sampled_df)
        
        return pd.concat(sampled_dfs).sample(frac=1).reset_index(drop=True)
    
    def stratified_sampling_and_split_categorys(self, df, category_1, category_2, sample_size=None):
        
        """
        category_1 내에서 category_2의 중복 없이 샘플링하고, 각 category_1에서 sample_size만큼 데이터 선택

        :param df: 입력 데이터프레임
        :param category_1: 첫 번째 카테고리 컬럼명 (예: '대분류')
        :param category_2: 두 번째 카테고리 컬럼명 (예: '소분류')
        :param sample_size: category_1별 샘플링할 개수 (None이면 전체 사용)
        :return: 샘플링된 데이터프레임
        """
        sampled_dfs = []
        total_samples = 0  # 총 샘플 개수 추적

        print(f"{category_1} 기준으로 {category_2} 샘플링 시작")

        # 첫 번째 카테고리(`category_1`)별 그룹화
        for cat1_value in df[category_1].unique():
            cat1_df = df[df[category_1] == cat1_value]
            print(f" {category_1}: {cat1_value} 처리 중")

            # `category_2`의 중복 제거
            unique_category_2 = cat1_df[category_2].unique()

            # category_2 중복 없이 category_1 별 sample_size 만큼 샘플링
            num_samples = min(sample_size, len(unique_category_2)) if sample_size else len(unique_category_2)

            # category_2 랜덤 샘플링 (중복 방지)
            selected_category_2 = np.random.choice(unique_category_2, size=num_samples, replace=False)

            # 선택된 category_2에 해당하는 데이터 중 랜덤 sample_size 샘플링
            sampled_df = cat1_df[cat1_df[category_2].isin(selected_category_2)].sample(n=num_samples)

            sampled_dfs.append(sampled_df)
            total_samples += len(sampled_df)

            print(f" - {category_2} 중복 없이 {num_samples}개 샘플링 완료")

        # 🔹 최종 데이터프레임 생성 (셔플)
        final_df = pd.concat(sampled_dfs).sample(frac=1).reset_index(drop=True)

        print(f"최종 샘플링 완료! 총 {total_samples}개의 데이터가 선택되었습니다.")

        return final_df


    def load_images(self, folder_image_paths, categories):
        """
        특정 파일 그룹 (folder_image_path 기준)으로 이미지를 묶어 5D NumPy 배열로 변환
        """
        category_mapping = {
            '바벨/덤벨': 'Barbell_Dumbbell/',
            '맨몸 운동': 'Bodyweight/',
            '기구': 'Equipment/'
        }

        image_groups = []
        max_images_per_group = 0
        all_groups = []
        category_list = []

        for folder_image_path, category in zip(folder_image_paths, categories):
            category_dir = category_mapping.get(category)
            folder_image_path = folder_image_path.replace("/", "@")
            img_list = sorted(os.listdir(self.img_base_dir + category_dir))
            img_paths = [img for img in img_list if folder_image_path in img]
            image_group = []
            category_added = False
            for img_path in img_paths:
                img = cv2.imread(self.img_base_dir + category_dir + img_path)
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    image_group.append(img)
                    if not category_added:
                        category_list.append(category)
                        category_added = True
                    print(img_path, '로드 완료')
                else:
                    print(f"Error loading: {img_path}")

            if image_group:
                max_images_per_group = max(max_images_per_group, len(image_group))
                all_groups.append(image_group)
            
        
        # Zero Padding (sahpe을 일정하게 유지지)
        for group in all_groups:
            while len(group) < max_images_per_group:
                group.append(np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.float32))

            image_groups.append(np.array(group, dtype=np.float32) / 255.0)

        return np.array(image_groups, dtype=np.float32), np.array(category_list)
    

    def sub_model_load_images(self, folder_image_paths, categories, return_categories):
        """
        특정 파일 그룹 (folder_image_path 기준)으로 이미지를 묶어 5D NumPy 배열로 변환
        """
        category_mapping = {
            '바벨/덤벨': 'Barbell_Dumbbell/',
            '맨몸 운동': 'Bodyweight/',
            '기구': 'Equipment/'
        }
        
        image_groups = []
        max_images_per_group = 0
        all_groups = []
        category_list = []

        # folder_image_paths의 index와 함께 반복
        for idx, (folder_image_path, category) in enumerate(zip(folder_image_paths, categories)):
            category_dir = category_mapping.get(category)
            folder_image_path = folder_image_path.replace("/", "@")

            # 해당 디렉토리의 이미지 파일 리스트 가져오기
            img_list = sorted(os.listdir(self.img_base_dir + category_dir))
            img_paths = [img for img in img_list if folder_image_path in img]

            image_group = []

            for img_path in img_paths:
                img = cv2.imread(self.img_base_dir + category_dir + img_path)
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    image_group.append(img)
                    print(img_path, '로드 완료')
                else:
                    print(f"Error loading: {img_path}")

            if image_group:
                max_images_per_group = max(max_images_per_group, len(image_group))
                all_groups.append(image_group)

                # return_categories에서 현재 index에 해당하는 값 추가
                if idx in return_categories.index:
                    category_list.append(return_categories.loc[idx])

        # Zero Padding (shape을 일정하게 유지)
        for group in all_groups:
            while len(group) < max_images_per_group:
                group.append(np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.float32))

            image_groups.append(np.array(group, dtype=np.float32) / 255.0)

        return np.array(image_groups, dtype=np.float32), np.array(category_list)



    def split_data(self, X_data, labels_array, save_mapping=True, mapping_filename="label_mapping.json"):
        """
        5D NumPy 배열(X)과 1차원 NumPy 문자열 배열(Y)을 입력받아 Train/Test 데이터 분할 (이미지 로드 제거)
        
        :param X_data: 5D NumPy 배열 (이미지 데이터)
        :param labels_array: 1차원 NumPy 배열 (문자열 라벨)
        :param save_mapping: json파일로 저장을 진행할지 여부 (bool)
        :param mapping_filename: 라벨인코더의 정보를 json파일로 저장할때 파일이름
        :return: train_X, train_Y, test_X, test_Y, label_mapping
        """
        # Train/Test 데이터 분할
        train_X, test_X, train_Y_raw, test_Y_raw = train_test_split(
            X_data, labels_array, test_size=self.test_split, stratify=labels_array
        )

        # 라벨 인코딩 (문자형 → 숫자형)
        label_encoder = LabelEncoder()
        train_Y_encoded = label_encoder.fit_transform(train_Y_raw)
        test_Y_encoded = label_encoder.transform(test_Y_raw)

        # 원-핫 인코딩 (숫자형 → 원-핫 벡터)
        train_Y = to_categorical(train_Y_encoded)
        test_Y = to_categorical(test_Y_encoded)

        # 라벨 인코딩 매핑 딕셔너리 생성
        label_mapping = {class_name: int(label_id) for class_name, label_id in zip(label_encoder.classes_, range(len(label_encoder.classes_)))}

        print(f"Train X {train_X.shape}, Train Y {train_Y.shape}, Test X {test_X.shape}, Test Y {test_Y.shape}")
        print(f"라벨 인코딩 매핑: {label_mapping}")

        def save_label_mapping(label_mapping, filename="label_mapping.json"):
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(label_mapping, f, indent=4, ensure_ascii=False)

        if save_mapping:
            if "/" in mapping_filename:
                safe_filename = mapping_filename.replace("/", "_")
            save_label_mapping(label_mapping, filename=safe_filename)

        return train_X, train_Y, test_X, test_Y, label_mapping




class ConvLSTM_NasNet_Model:
    """
    ConvLSTM + NASNetMobile 모델 클래스
    """
    def __init__(self, input_shape=(None, 64, 64, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        video_input = Input(shape=self.input_shape)
        x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(video_input)
        x = BatchNormalization()(x)

        # ConvLSTM → NASNetMobile 연결을 위해 채널 변환
        x = TimeDistributed(Conv2D(3, (1, 1), activation="relu"))(x)

        # NASNetMobile 불러오기 (사전 학습된 가중치 사용)
        base_model = NASNetMobile(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
        base_model.trainable = False  # 초기 가중치 동결

        for layer in base_model.layers[-20:]:  
            layer.trainable = True  # 마지막 20개 레이어만 학습 가능

        # TimeDistributed로 NASNet 적용
        x = TimeDistributed(base_model)(x)
        x = BatchNormalization()(x)
        x = TimeDistributed(GlobalAveragePooling2D())(x)

        # 차원 축소
        x = GlobalAveragePooling1D()(x)

        # Fully Connected Layer
        x = Flatten()(x)
        x = Dense(512, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(self.num_classes, activation="softmax")(x)

        model = Model(inputs=video_input, outputs=x)
        model.compile(optimizer='adam',
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        return model
    
class ConvLSTM_ResNet_Model:
    """
    ConvLSTM + ResNet101 모델 클래스
    """
    def __init__(self, input_shape=(None, 64, 64, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        video_input = Input(shape=self.input_shape)
        x = ConvLSTM2D(filters=128, kernel_size=(5, 5), padding="same", return_sequences=True, activation="relu")(video_input)
        x = BatchNormalization()(x)

        # ConvLSTM → ResNet101 연결을 위해 채널 변환
        x = TimeDistributed(Conv2D(3, (1, 1), activation="relu"))(x)

        # ResNet101 불러오기 (사전 학습된 가중치 사용)
        base_model = ResNet101(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
        base_model.trainable = False  # 초기 가중치 동결

        for layer in base_model.layers[-30:]:  
            layer.trainable = True  # 마지막 20개 레이어만 학습 가능

        # TimeDistributed로 ResNet 적용
        x = TimeDistributed(base_model)(x)
        x = BatchNormalization()(x)
        x = TimeDistributed(GlobalAveragePooling2D())(x)

        # 차원 축소
        x = GlobalAveragePooling1D()(x)

        # Fully Connected Layer
        x = Flatten()(x)
        x = Dense(512, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.2)(x)
        x = Dense(self.num_classes, activation="softmax")(x)

        model = Model(inputs=video_input, outputs=x)
        model.compile(optimizer='adam',
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        return model

class ConvLSTM_EfficientNetB7_Model:
    """
    ConvLSTM + EfficientNetB7 모델 클래스
    """
    def __init__(self, input_shape=(None, 64, 64, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        video_input = Input(shape=self.input_shape)
        x = ConvLSTM2D(filters=128, kernel_size=(5, 5), padding="same", return_sequences=True, activation="relu")(video_input)
        x = BatchNormalization()(x)

        # ConvLSTM → ResNet101 연결을 위해 채널 변환
        x = TimeDistributed(Conv2D(3, (1, 1), activation="relu"))(x)

        # ResNet101 불러오기 (사전 학습된 가중치 사용)
        base_model = EfficientNetB7(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
        base_model.trainable = False  # 초기 가중치 동결

        for layer in base_model.layers[-30:]:  
            layer.trainable = True  # 마지막 20개 레이어만 학습 가능

        # TimeDistributed로 ResNet 적용
        x = TimeDistributed(base_model)(x)
        x = BatchNormalization()(x)
        x = TimeDistributed(GlobalAveragePooling2D())(x)

        # 차원 축소
        x = GlobalAveragePooling1D()(x)

        # Fully Connected Layer
        x = Flatten()(x)
        x = Dense(512, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.2)(x)
        x = Dense(self.num_classes, activation="softmax")(x)

        model = Model(inputs=video_input, outputs=x)
        model.compile(optimizer='adam',
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        return model

# class ConvLSTM_EfficientNetB3_Model:
#     """
#     ConvLSTM + EfficientNetB3 모델 클래스
#     """
#     def __init__(self, input_shape=(None, 64, 64, 3), num_classes=3):
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#         self.model = self.build_model()

#     def build_model(self):
#         video_input = Input(shape=self.input_shape)
#         x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(video_input)
#         x = BatchNormalization()(x)

#         x = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
#         x = BatchNormalization()(x)

#         x = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding="same", return_sequences=False, activation="relu")(x)
#         x = BatchNormalization()(x)

#         x = Reshape((1, 64, 64, 256))(x)
#         # ConvLSTM → EfficientNetB3 연결을 위해 채널 변환
#         x = TimeDistributed(Conv2D(3, (1, 1), activation="relu"))(x)

#         # NASNetMobile 불러오기 (사전 학습된 가중치 사용)
#         base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
#         base_model.trainable = False  # 초기 가중치 동결

#         for layer in base_model.layers[-20:]:  
#             layer.trainable = True  # 마지막 20개 레이어만 학습 가능

#         # TimeDistributed로 NASNet 적용
#         x = TimeDistributed(base_model)(x)
#         x = BatchNormalization()(x)
#         x = TimeDistributed(GlobalAveragePooling2D())(x)

#         # 차원 축소
#         x = GlobalAveragePooling1D()(x)

#         # Fully Connected Layer
#         x = Flatten()(x)
#         x = Dense(512, activation="relu", kernel_regularizer=l2(0.01))(x)
#         x = Dropout(0.3)(x)
#         x = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(x)
#         x = Dropout(0.3)(x)
#         x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
#         x = Dropout(0.3)(x)
#         x = Dense(self.num_classes, activation="softmax")(x)

#         model = Model(inputs=video_input, outputs=x)
#         model.compile(optimizer='adam',
#                       loss="categorical_crossentropy",
#                       metrics=["accuracy"])
#         return model

def build_model(self):
    video_input = Input(shape=self.input_shape)
    
    # ConvLSTM2D 스택 적용
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(video_input)
    x = BatchNormalization()(x)

    x = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
    x = BatchNormalization()(x)

    x = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding="same", return_sequences=False, activation="relu")(x)
    x = BatchNormalization()(x)

    # ConvLSTM → EfficientNetB3 연결을 위해 채널 변환
    x = Conv2D(3, (1, 1), activation="relu")(x)

    # EfficientNetB3 불러오기 (사전 학습된 가중치 사용)
    base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
    base_model.trainable = False  # 초기 가중치 동결

    # 일부 레이어만 학습 가능하도록 설정
    for layer in base_model.layers[-20:]:  
        layer.trainable = True  # 마지막 20개 레이어만 학습 가능

    # EfficientNetB3 적용 (TimeDistributed 필요 없음)
    x = base_model(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layer
    x = Dense(512, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(self.num_classes, activation="softmax")(x)

    model = Model(inputs=video_input, outputs=x)
    model.compile(optimizer='adam',
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


class CustomCheckpoint(Callback):
    """
    에포크 20 이상 & val_accuracy 향상 시에만 모델 저장
    """
    def __init__(self, save_dir="models"):
        super(CustomCheckpoint, self).__init__()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)  # 저장 폴더 생성
        self.best_val_acc = 0  # 최고 val_accuracy 저장

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        val_acc = logs.get("val_accuracy", 0)
        val_loss = logs.get("val_loss", 0)

        # 에포크가 20 이상 & val_accuracy 향상 시 저장
        if (epoch + 1 >= 20 and val_acc > self.best_val_acc) or (epoch + 1 >= 80 and val_acc > 70):
            self.best_val_acc = val_acc  # 최고 val_accuracy 업데이트

            # 파일명에 epoch, val_acc, val_loss 포함
            filename = f"model_epoch-{epoch+1:03d}_val-acc-{val_acc:.4f}_val-loss-{val_loss:.4f}.h5"
            filepath = os.path.join(self.save_dir, filename)

            # 모델 저장
            self.model.save(filepath)
            print(f"모델 저장됨: {filename}")


class Trainer:
    """
    모델 학습, 평가 및 시각화를 위한 클래스
    """
    def __init__(self, model, train_X, train_y, test_X=None, test_y=None, batch_size=2, epochs=100, save_path="CNN_model/"):
        self.model = model
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_path = save_path

    def train(self):
        # EarlyStopping 및 ReduceLROnPlateau 적용
        checkpoint_callback = CustomCheckpoint(save_dir=self.save_path)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', # val_loss기준
                                      factor=0.5, # 개선되지 않으면 학습률 50프로 감소
                                      patience=5, # 5번동안 개선되지 않으면 학습률 감소
                                      min_lr=1e-6) # 과적합을 줄이기 위해 학습률 조절절

        hist = self.model.fit(
            self.train_X, self.train_y,
            validation_split=0.3,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[checkpoint_callback]
        )

        self.model.save("cnn_model.h5")
        self.plot_results(hist)

    def plot_results(self, hist):
        """ 학습 곡선 시각화 """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig, loss_ax = plt.subplots(figsize=(10, 5))
        loss_ax.plot(hist.history['loss'], 'r', label='train loss')
        loss_ax.plot(hist.history['val_loss'], 'y', label='validation loss')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')

        acc_ax = loss_ax.twinx()
        acc_ax.plot(hist.history['accuracy'], 'b', label='train accuracy')
        acc_ax.plot(hist.history['val_accuracy'], 'g', label='validation accuracy')
        acc_ax.set_ylabel('accuracy')

        loss_ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))
        acc_ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98))

        plt.savefig(f"{timestamp}_plot.jpg", dpi=300)
        plt.show()

    def evaluate(self):
        """ 테스트 데이터셋으로 예측 후 성능 평가 """
        if self.test_X is None or self.test_y is None:
            print("테스트 데이터가 없습니다! `test_X`와 `test_y`를 설정하세요.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 예측 수행
        y_pred_prob = self.model.predict(self.test_X)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(self.test_y, axis=1)

        # 혼동 행렬 (Confusion Matrix) 생성
        cm = confusion_matrix(y_true, y_pred)

        # Pandas DataFrame으로 변환하여 crosstab 시각화
        labels = list(range(self.test_y.shape[1]))  # 클래스 개수
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix (Crosstab)")
        plt.savefig(f"{timestamp}_crosstab.jpg", dpi=300)
        plt.show()

        # 평가 지표 계산
        f1 = f1_score(y_true, y_pred, average=None)  # 클래스별 F1 Score
        f1_macro = f1_score(y_true, y_pred, average="macro")  # 매크로 평균
        f1_weighted = f1_score(y_true, y_pred, average="weighted")  # 가중 평균
        precision = precision_score(y_true, y_pred, average="macro")  # Precision
        recall = recall_score(y_true, y_pred, average="macro")  # Recall
        accuracy = accuracy_score(y_true, y_pred)  # Accuracy

        # 평가 지표 출력
        print("\n F1 Score (클래스별):", f1)
        print(f" F1 Score (Macro 평균): {f1_macro:.4f}")
        print(f" F1 Score (Weighted 평균): {f1_weighted:.4f}")
        print(f" Precision (Macro 평균): {precision:.4f}")
        print(f" Recall (Macro 평균): {recall:.4f}")
        print(f" Accuracy: {accuracy:.4f}")

        # 현재 시간 기반 파일 이름 생성
        txt_filename = f"{timestamp}_CNN.txt"
        csv_filename = f"{timestamp}_crosstab.csv"

        # Confusion Matrix CSV 저장
        cm_df.to_csv(csv_filename, index=True)
        print(f" Confusion Matrix 저장됨: {csv_filename}")

        # 평가 지표 TXT 파일 저장
        with open(txt_filename, "w") as f:
            f.write(" 모델 평가 지표\n")
            f.write("=" * 30 + "\n")
            f.write(f" F1 Score (클래스별): {f1.tolist()}\n")
            f.write(f" F1 Score (Macro 평균): {f1_macro:.4f}\n")
            f.write(f" F1 Score (Weighted 평균): {f1_weighted:.4f}\n")
            f.write(f" Precision (Macro 평균): {precision:.4f}\n")
            f.write(f" Recall (Macro 평균): {recall:.4f}\n")
            f.write(f" Accuracy: {accuracy:.4f}\n")
            f.write("=" * 30 + "\n")

        print(f"평가 지표 저장됨: {txt_filename}")



if __name__ == '__main__':
    base_dir = 'D:/KHH/team_project/json_data/'
    img_base_dir = 'D:/KHH/team_project/img/'
    dataset_loader = DatasetLoader(base_dir, img_base_dir, img_size=(64, 64), test_split=0.2)

    json_df = pd.concat([pd.read_csv(base_dir+file) for file in os.listdir(base_dir) if 'validation' not in file])

    # print(json_df)

    dataset_df = dataset_loader.stratified_sampling_and_split(json_df, 'type', 10)
    dataset_df['folder'] = dataset_df['img_key'].apply(lambda x: "/".join(x.split("/")[:4])).str.replace('/', '@')

    # print(dataset_df.columns)

    img_dataset, drop_index = dataset_loader.load_images(dataset_df['folder'], dataset_df['type'])
    dataset_df.drop(index=drop_index, inplace=True)

    print(img_dataset.shape)

    train_X, train_Y, test_X, test_Y, train_LABELS, test_LABELS = dataset_loader.split_data(img_dataset, dataset_df[['type']])
    print(train_Y)
    print('-----------------------------------------------------------')
    print(test_Y)

    model = ConvLSTM_NasNet_Model().build_model()

    trainer = Trainer(model=model, train_X=train_X, train_y=train_Y, test_X=test_X, test_y=test_Y,epochs=10)
    trainer.train()
    trainer.evaluate()