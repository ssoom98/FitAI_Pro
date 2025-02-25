from Conv_LSTM_Model import ConvLSTM_NasNet_IMG_Gen_Model
from data_loader import DatasetLoader
from trainer import Trainer
import pandas as pd
import numpy as np

img_base_dir = 'D:/KHH/team_project/img/'

category_mapping = {
            '바벨/덤벨': 'Barbell_Dumbbell/',
            '맨몸 운동': 'Bodyweight/',
            '기구': 'Equipment/'
        }

json_df = pd.read_csv('D:/KHH/team_project/json_data/cnn_data.csv')
print(json_df)

data_loader = DatasetLoader()
dataset_df = data_loader.random_sampling_data(json_df, 'workout', size=10)

types = ['바벨/덤벨', '맨몸 운동', '기구']

for tp in types:
    tp_dataset_df = dataset_df[dataset_df['type'] == tp]
    data_X, data_y = data_loader.sub_load_img(tp_dataset_df, 'folder', 'type', 'workout')
    data_Y = data_loader.one_hot_encode(data_y)
    print(data_X.shape, data_Y.shape)

    # 고유값과 개수 계산
    unique_values, counts = np.unique(data_y, return_counts=True)

    # 결과 출력
    print(unique_values)  # [1 2 3 4]
    print(counts)  # [1 2 3 4]

    unique_values, counts = np.unique(data_Y, return_counts=True)

    # 결과 출력
    print(unique_values)  # [1 2 3 4]
    print(counts)  # [1 2 3 4]


    X_train, X_test, y_train, y_test = data_loader.train_test_split(data_X, data_Y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    modle = ConvLSTM_NasNet_IMG_Gen_Model(input_shape=data_X.shape[1:], num_classes=data_Y.shape[1]).build_model()

    trainder = Trainer(model=modle, train_X=X_train, train_y=data_Y, test_X=X_test, test_y=y_test, save_path='D:/KHH/team_project/proseced_logic/Conv_LSTM_Rasnet_sub/ConvLSTM_NasNet_IMG_Gen_Model' +  category_mapping.get(tp))
    trainder.train()
    trainder.evaluate()