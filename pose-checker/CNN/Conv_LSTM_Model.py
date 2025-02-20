import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, ConvLSTM2D, TimeDistributed, BatchNormalization,
    Conv2D, Dense, Flatten, GlobalAveragePooling2D, 
    GlobalAveragePooling1D, Dropout
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

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
                sampled_df = category_df.sample(n=min(sample_size, len(category_df)), random_state=42)
            
            sampled_dfs.append(sampled_df)
        
        return pd.concat(sampled_dfs).sample(frac=1).reset_index(drop=True)

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
        non_img_idx = []
        max_images_per_group = 0
        all_groups = []
        img_idx = 0

        for folder_image_path, category in zip(folder_image_paths, categories):
            category_dir = category_mapping.get(category)
            folder_image_path = folder_image_path.replace("/", "@")
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
                    non_img_idx.append(folder_image_path)

            if image_group:
                max_images_per_group = max(max_images_per_group, len(image_group))
                all_groups.append(image_group)
            
            img_idx += 1
        
        # Zero Padding (sahpe을 일정하게 유지지)
        for group in all_groups:
            while len(group) < max_images_per_group:
                group.append(np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.float32))

            image_groups.append(np.array(group, dtype=np.float32) / 255.0)

        return np.array(image_groups, dtype=np.float32), non_img_idx


    def split_data(self, X_data, labels_df, category_column='type'):
        """
        5D NumPy 배열(X)과 Pandas DataFrame(Y)를 입력받아 Train/Test 데이터 분할 (이미지 로드 제거)
        
        :param X_data: 5D NumPy 배열 (이미지 데이터)
        :param labels_df: Pandas DataFrame (라벨 정보 포함)
        :param category_column: 분류할 라벨 컬럼명 (기본값: 'type')
        :return: train_X, train_Y, test_X, test_Y, train_LABELS, test_LABELS (X: NumPy 배열, Y: Pandas DataFrame)
        """
        # Train/Test 데이터 분할
        train_X, test_X, train_LABELS, test_LABELS = train_test_split(
            X_data, labels_df, test_size=self.test_split, stratify=labels_df[category_column]
        )

        # 라벨 인코딩 (문자형 → 숫자형)
        label_encoder = LabelEncoder()
        train_Y = to_categorical(label_encoder.fit_transform(train_LABELS[category_column]))
        test_Y = to_categorical(label_encoder.transform(test_LABELS[category_column]))

        print(f"Train X {train_X.shape}, Train Y {train_Y.shape}, Test X {test_X.shape}, Test Y {test_Y.shape}")

        return train_X, train_Y, test_X, test_Y, train_LABELS, test_LABELS



class ConvLSTMModel:
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

        # ✅ 에포크가 20 이상 & val_accuracy 향상 시 저장
        if epoch + 1 >= 20 and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc  # 최고 val_accuracy 업데이트

            # 파일명에 epoch, val_acc, val_loss 포함
            filename = f"model_epoch-{epoch+1:03d}_val-acc-{val_acc:.4f}_val-loss-{val_loss:.4f}.h5"
            filepath = os.path.join(self.save_dir, filename)

            # 모델 저장
            self.model.save(filepath)
            print(f"✅ 모델 저장됨: {filename}")


class Trainer:
    """
    모델 학습, 평가 및 시각화를 위한 클래스
    """
    def __init__(self, model, train_X, train_y, test_X=None, test_y=None, batch_size=2, epochs=100):
        self.model = model
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self):
        # EarlyStopping 및 ReduceLROnPlateau 적용
        checkpoint_callback = CustomCheckpoint(save_dir='CNN_model/')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        hist = self.model.fit(
            self.train_X, self.train_y,
            validation_split=0.3,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[checkpoint_callback, reduce_lr]
        )

        self.model.save("cnn_model.h5")
        self.plot_results(hist)

    def plot_results(self, hist):
        """ 학습 곡선 시각화 """
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

        plt.savefig("plot.jpg", dpi=300)
        plt.show()

    def evaluate(self):
        """ 테스트 데이터셋으로 예측 후 성능 평가 """
        if self.test_X is None or self.test_y is None:
            print("테스트 데이터가 없습니다! `test_X`와 `test_y`를 설정하세요.")
            return
        
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
        plt.savefig("crosstab.jpg", dpi=300)
        plt.show()


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

    model = ConvLSTMModel().build_model()

    trainer = Trainer(model=model, train_X=train_X, train_y=train_Y, test_X=test_X, test_y=test_Y,epochs=10)
    trainer.train()
    trainer.evaluate()