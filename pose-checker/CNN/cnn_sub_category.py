from Conv_LSTM_Model import DatasetLoader, ConvLSTM_ResNet_Model, Trainer
import pandas as pd
import os

base_dir = 'D:/KHH/team_project/json_data/'
img_base_dir = 'D:/KHH/team_project/img/'


json_df = pd.concat([pd.read_csv(base_dir+file) for file in os.listdir(base_dir) if 'validation' not in file])
json_df['folder'] = json_df['img_key'].apply(lambda x: "/".join(x.split("/")[:4])).str.replace('/', '@')

# print(json_df)
type_mapping = {
            '바벨/덤벨': 'Barbell_Dumbbell/',
            '맨몸 운동': 'Bodyweight/',
            '기구': 'Equipment/'
        }
data_type = ['바벨/덤벨', '맨몸 운동', '기구']
columns_to_read = ["img_key", "type", "workout", "conditions"]


for tp in data_type:

    temp_df = json_df[json_df['type']=='맨몸 운동']

    dataset_loader = DatasetLoader(base_dir, img_base_dir, img_size=(64, 64), test_split=0.2)

    dataset_df = dataset_loader.stratified_sampling_and_split_categorys(temp_df, 'workout', 'folder', 50)


    print(dataset_df)

    dataset_X, dataset_Y = dataset_loader.sub_model_load_images(dataset_df['folder'], dataset_df['type'], dataset_df['workout'])

    print(dataset_X.shape, dataset_Y.shape)
    print(dataset_Y)

    train_X, train_Y, test_X, test_Y, _ = dataset_loader.split_data(dataset_X, dataset_Y, mapping_filename="type_라벨인코더.json")
    print(train_Y)
    print('-----------------------------------------------------------')
    print(test_Y)

    model = ConvLSTM_ResNet_Model(num_classes=train_Y.shape[1]).build_model()

    print(train_X.shape, train_Y.shape)
    print(test_X.shape, test_Y.shape)

    trainer = Trainer(model=model, train_X=train_X, train_y=train_Y, test_X=test_X, test_y=test_Y,epochs=100, save_path='D:/KHH/team_project/proseced_logic/Conv_LSTM_Rasnet_sub/'+type_mapping.get(tp))
    trainer.train()
    trainer.evaluate()