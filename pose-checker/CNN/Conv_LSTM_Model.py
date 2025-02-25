import os
import tensorflow as tf
from tensorflow.keras.applications import NASNetMobile, ResNet101, EfficientNetB3, EfficientNetB7
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, ConvLSTM2D, TimeDistributed, BatchNormalization,
    Conv2D, Dense, Flatten, GlobalAveragePooling2D, 
    GlobalAveragePooling1D, Dropout, Conv3D, MaxPooling3D,
    RandomFlip, RandomRotation, RandomZoom, RandomContrast
)
from tensorflow.keras.regularizers import l2
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score




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
        x = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
        x = BatchNormalization()(x)
        x = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
        x = BatchNormalization()(x)
        x = ConvLSTM2D(filters=512, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
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
        x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(self.num_classes, activation="softmax")(x)

        model = Model(inputs=video_input, outputs=x)
        model.compile(optimizer='adam',
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        return model

class ConvLSTM_NasNet_IMG_Gen_Model:
    """
    RandomLayer 데이터 증강 + ConvLSTM + NASNetMobile 모델 클래스
    """
    def __init__(self, input_shape=(None, 64, 64, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        video_input = Input(shape=self.input_shape)
        data_augmentation = TimeDistributed(tf.keras.Sequential([
            RandomFlip("horizontal"),
            RandomRotation(0.2),
            RandomZoom(0.1),
            RandomContrast(0.2)
        ]))
        augmented_output = data_augmentation(video_input)
        x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(augmented_output)
        x = BatchNormalization()(x)
        x = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
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
        x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(x)
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
        x = ConvLSTM2D(filters=256, kernel_size=(5, 5), padding="same", return_sequences=True, activation="relu")(video_input)
        x = BatchNormalization()(x)
        x = ConvLSTM2D(filters=512, kernel_size=(5, 5), padding="same", return_sequences=True, activation="relu")(video_input)
        x = BatchNormalization()(x)

        # ConvLSTM → ResNet101 연결을 위해 채널 변환
        x = TimeDistributed(Conv2D(3, (1, 1), activation="relu"))(x)

        # ResNet101 불러오기 (사전 학습된 가중치 사용)
        base_model = ResNet101(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
        base_model.trainable = False  # 초기 가중치 동결

        for layer in base_model.layers[-30:]:  
            layer.trainable = True  # 마지막 30개 레이어만 학습 가능

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
        x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(x)
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

class ConvLSTM_EfficientNetB3_Model:
    """
    ConvLSTM + EfficientNetB3 모델 클래스
    """
    def __init__(self, input_shape=(None, 64, 64, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        video_input = Input(shape=self.input_shape)
        
        # ConvLSTM2D 스택 적용
        x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(video_input)
        x = BatchNormalization()(x)

        x = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
        x = BatchNormalization()(x)

        x = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding="same", return_sequences=False, activation="relu")(x)
        x = BatchNormalization()(x)

        x = ConvLSTM2D(filters=512, kernel_size=(3, 3), padding="same", return_sequences=False, activation="relu")(x)
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
        x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(self.num_classes, activation="softmax")(x)

        model = Model(inputs=video_input, outputs=x)
        model.compile(optimizer='adam',
                    loss="categorical_crossentropy",
                    metrics=["accuracy"])
        return model

class ConvLSTM_EfficientNetB3_OvA_Model:
    """
    ConvLSTM + EfficientNetB3 모델 (One-vs-All Multi-Head 방식)
    """
    def __init__(self, input_shape=(None, 64, 64, 3), num_classes=17):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        video_input = Input(shape=self.input_shape)
        
        # ConvLSTM2D 스택 적용
        x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(video_input)
        x = BatchNormalization()(x)

        x = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
        x = BatchNormalization()(x)

        x = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding="same", return_sequences=False, activation="relu")(x)
        x = BatchNormalization()(x)

        x = ConvLSTM2D(filters=512, kernel_size=(3, 3), padding="same", return_sequences=False, activation="relu")(x)
        x = BatchNormalization()(x)

        # ConvLSTM → EfficientNetB3 연결을 위해 채널 변환
        x = Conv2D(3, (1, 1), activation="relu")(x)

        # EfficientNetB3 불러오기 (사전 학습된 가중치 사용)
        base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
        base_model.trainable = True  # 초기 가중치 동결

        # 일부 레이어만 학습 가능하도록 설정
        # for layer in base_model.layers[-30:]:  
        #     layer.trainable = True  # 마지막 30개 레이어만 학습 가능

        # EfficientNetB3 적용 (TimeDistributed 필요 없음)
        x = base_model(x)
        x = BatchNormalization()(x)
        x = GlobalAveragePooling2D()(x)

        # Fully Connected Layer (공통 Feature Extractor)
        x = Dense(512, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)

        # Multi-Head 이진 분류기 (OvA 방식, sigmoid 활성화)
        output = Dense(self.num_classes, activation="sigmoid")(x)  # OvA 방식 적용

        model = Model(inputs=video_input, outputs=output)
        model.compile(optimizer='adam',
                      loss="binary_crossentropy",  # OvA에서는 binary_crossentropy 사용
                      metrics=["accuracy"])
        return model


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