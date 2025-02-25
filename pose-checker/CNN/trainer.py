
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score



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

        # EagerTensor를 float으로 변환 (오류 방지)
        val_acc = float(val_acc) if isinstance(val_acc, tf.Tensor) else val_acc
        val_loss = float(val_loss) if isinstance(val_loss, tf.Tensor) else val_loss

        # 에포크가 20 이상 & val_accuracy 향상 시 저장
        if (epoch + 1 >= 20 and val_acc > self.best_val_acc) or (epoch + 1 >= 80 and val_acc > 70):
            self.best_val_acc = val_acc  # 최고 val_accuracy 업데이트

            # 파일명에 epoch, val_acc, val_loss 포함
            filename = f"model_epoch-{epoch+1:03d}_val-acc-{val_acc:.4f}_val-loss-{val_loss:.4f}.h5"
            filepath = os.path.join(self.save_dir, filename)

            self.model.save_weights(filepath)

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

        plt.savefig(self.save_path + f"{timestamp}_plot.jpg", dpi=300)
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
        plt.savefig(self.save_path + f"{timestamp}_crosstab.jpg", dpi=300)
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
        txt_filename = self.save_path + f"{timestamp}_CNN.txt"
        csv_filename = self.save_path + f"{timestamp}_crosstab.csv"

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