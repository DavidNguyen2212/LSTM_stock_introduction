from matplotlib import pyplot as plt
import numpy as np
from trainer.pandasProcessing import convert_df, extract_features
from trainer.train import forecast
from trainer.utils import calculate_accuracy


test_size = 30
num_layers = 1
size_layer = 128
timestamp = 120
epoch = 5
dropout_rate = 0.1
future_day = 30
learning_rate = 0.005

def main():
    df, daily_df, close_scaler = convert_df('data/PNJ.csv')
    df_log  = extract_features(df) 
    df_train = df_log.iloc[:-test_size]
    df_test = df_log.iloc[-test_size:]

    results = []
    for i in range(2):
        print('simulation %d'%(i + 1))
        results.append(forecast(
                daily_df=daily_df, df_train=df_train, scaler=close_scaler,
                learning_rate=learning_rate, num_layers=num_layers, size_layer=size_layer, dropout_rate=dropout_rate,
                test_size=test_size, epoch=epoch, timestamp=timestamp
            )
        )
    accuracies = [calculate_accuracy(daily_df['Close'].values, r) for r in results]

    plt.figure(figsize = (15, 5))
    for no, r in enumerate(results):
        plt.plot(r, label = 'forecast %d'%(no + 1))
    plt.plot(df['Close'].iloc[-test_size:].values, label = 'True trend', c = 'black')
    plt.legend()
    plt.title('Average accuracy: %.4f'%(np.mean(accuracies)))
    plt.savefig()
    plt.show()

if __name__ == "__main__":
    main()