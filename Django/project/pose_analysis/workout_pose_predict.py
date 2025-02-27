import joblib
import numpy as np
import os


class Predictor:
    def __init__(self, model_base_dir="pose_analysis/models/"):
        self.model_base_dir = model_base_dir

    def pad_features(self, features, expected_size=1071):
        # features를 1차원 배열로 평탄화합니다.
        features = np.array(features).flatten()
        current_size = features.shape[0]
        if current_size < expected_size:
            padding = expected_size - current_size
            # 1차원 배열이므로 pad_width는 (0, padding)만 지정합니다.
            features = np.pad(features, (0, padding), mode='constant', constant_values=0)
        return features

    def workout_predict(self, keypoint):
        model_path = os.path.join(self.model_base_dir, "gym_exercise_classifier.pkl")
        model = joblib.load(model_path)
        prediction = model.predict(keypoint)
        print(prediction)
        print(type(prediction))
        return prediction

    def pose_predict(self, keypoint, workout):
        model_pose_dir = os.path.join(self.model_base_dir, "workout_form_analyzer")
        model_list = os.listdir(model_pose_dir)
        model_path = None

        for model_file in model_list:
            if workout in model_file:
                model_path = os.path.join(model_pose_dir, model_file)
                break

        if model_path is None:
            raise ValueError(f"No model found for workout: {workout}")

        model = joblib.load(model_path)
        pred_X = keypoint[:model.n_features_in_] # 모델마다 요구하는 X값이 달라 배열을 자르기 위함
        pred_X = pred_X.reshape(1,-1)
        prediction = model.predict(pred_X)
        return prediction