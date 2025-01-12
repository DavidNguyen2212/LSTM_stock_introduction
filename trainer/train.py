from datetime import timedelta
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tqdm import tqdm
from trainer.StockModel import StockModel
from trainer.pandasProcessing import convert_df
from trainer.utils import anchor, calculate_accuracy

test_size = 30
num_layers = 1
size_layer = 128
timestamp = 90
epoch = 10
dropout_rate = 0.1
future_day = 30
learning_rate = 0.01

def forecast(daily_df: DataFrame, df_train: DataFrame, scaler: MinMaxScaler):
    # Khởi tạo model
    model = StockModel(learning_rate, num_layers, size_layer, df_train.shape[1], dropout_rate)

    # Convert data to tensor
    train_data = tf.convert_to_tensor(df_train.values, dtype=tf.float32)

    date_ori = pd.to_datetime(daily_df.iloc[:, -1]).tolist()

    # training
    for epoch_idx in tqdm(range(epoch), desc="Training"):
        init_value = [tf.zeros(1, size_layer) for _ in range(num_layers)]
        total_loss, total_acc = [], []

        for k in range(0, df_train.shape[0] - 1, timestamp):
            index = min(k + timestamp, df_train.shape[0] - 1)
            batch_x = train_data[k : index]
            batch_y = train_data[k + 1 : index + 1]
            with tf.GradientTape() as tape:
                logits, last_state = model(tf.expand_dims(batch_x, axis=0), init_value)
                loss = model.compute_loss(logits, tf.expand_dims(batch_y, axis=0))

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Update hidden states
            init_value = last_state
            total_loss.append(loss.numpy())
            total_acc.append(calculate_accuracy(batch_y.numpy()[:, 0], logits.numpy()[:, 0]))
        tqdm.write(f"Epoch {epoch_idx + 1}, Loss: {np.mean(total_loss)}, Accuracy: {np.mean(total_acc)}")

    # Dự báo tương lai
    output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
    output_predict[0] = df_train.iloc[0].values
    upper_b = (df_train.shape[0] // timestamp) * timestamp
    init_value = [tf.zeros((1, size_layer)) for _ in range(num_layers)]  # Reset trạng thái
        # Dự đoán phần dữ liệu đã biết
    for k in range(0, upper_b, timestamp):
        batch_x = tf.expand_dims(train_data[k : k + timestamp], axis=0)
        logits, last_state = model(batch_x, init_value)
        init_value = last_state
        output_predict[k + 1 : k + timestamp + 1] = logits.numpy()

    if upper_b != df_train.shape[0]:
        batch_x = tf.expand_dims(train_data[upper_b:], axis=0)
        logits, last_state = model(batch_x, init_value)
        output_predict[upper_b + 1 : df_train.shape[0] + 1] = logits.numpy()
        future_day -= 1
        date_ori.append(date_ori[-1] + timedelta(days=1))

    init_value = last_state

    # Dự đoán dữ liệu tương lai
    for i in range(future_day):
        o = output_predict[-future_day - timestamp + i : -future_day + i]
        batch_x = tf.expand_dims(o, axis=0)
        logits, last_state = model(batch_x, init_value)
        init_value = last_state
        output_predict[-future_day + i] = logits.numpy()[-1]
        date_ori.append(date_ori[-1] + timedelta(days=1))

    # Đưa kết quả về giá trị ban đầu
    output_predict = scaler.inverse_transform(output_predict)
    deep_future = anchor(output_predict[:, 0], 0.3)

    return deep_future[-test_size:]


def main():
    df, close_scaler = convert_df('../data/PNJ.csv') 
    df_train = df.iloc[:-test_size]
    df_test = df.iloc[-test_size:]
    results = []
    for i in range(2):
        print('simulation %d'%(i + 1))
        results.append(forecast(df, df_train, close_scaler))
    accuracies = [calculate_accuracy(df_test.values, r) for r in results]

    plt.figure(figsize = (15, 5))
    for no, r in enumerate(results):
        plt.plot(r, label = 'forecast %d'%(no + 1))
    plt.plot(df['Close'].iloc[-test_size:].values, label = 'true trend', c = 'black')
    plt.legend()
    plt.title('average accuracy: %.4f'%(np.mean(accuracies)))
    plt.show()

main()


    
