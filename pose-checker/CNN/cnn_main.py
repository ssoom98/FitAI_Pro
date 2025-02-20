from Conv_LSTM_Model import DatasetLoader, ConvLSTMModel, Trainer
import pandas as pd
import os

base_dir = 'D:/KHH/team_project/json_data/'
img_base_dir = 'D:/KHH/team_project/img/'
dataset_loader = DatasetLoader(base_dir, img_base_dir, img_size=(64, 64), test_split=0.2)

json_df = pd.concat([pd.read_csv(base_dir+file) for file in os.listdir(base_dir) if 'validation' not in file])

# print(json_df)

dataset_df = dataset_loader.stratified_sampling_and_split(json_df, 'type', 200)
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

print(train_X.shape, train_Y.shape)
print(test_X.shape, test_Y.shape)

trainer = Trainer(model=model, train_X=train_X, train_y=train_Y, test_X=test_X, test_y=test_Y,epochs=200)
trainer.train()
trainer.evaluate()