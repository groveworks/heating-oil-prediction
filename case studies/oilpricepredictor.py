import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import Sequential
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def main():
    file = Path("data/monthly-heating-oil-price.csv")
    if file.is_file():
        df = pd.read_csv("data/monthly-heating-oil-price.csv")
        df['date'] = pd.to_datetime(df['date'])
        df['months'] = pd.DatetimeIndex(df['date']).month
        df['years'] = pd.DatetimeIndex(df['date']).year
        dff = df.drop(columns=['date'])
        dff.to_csv("data/mhoprice.csv", index=False)
    else:
        dff = pd.read_csv("data/mhoprice.csv")
    Xy = np.array(dff)
    scaler = MinMaxScaler().fit(Xy)
    Xyt = scaler.transform(Xy)
    X = Xyt[:, 1:4]
    y = Xyt[:,0]
    look_back = 1
    Xs, ys = [], []
    for i in range(len(Xyt) - look_back):
        v = Xyt[i:i+look_back]
        Xs.append(v)
        ys.append(Xyt[i+look_back])
    Xs = np.array(Xs)
    ys = np.array(ys)
    # Design the machine learning algorithm
    xsize = Xs.shape
    ysize = ys.shape
    X_train = Xs[:round(0.75*xsize[0])]
    X_test = Xs[:round(0.25*xsize[0])]
    y_train = ys[:round(0.75*ysize[0])]
    y_test = ys[:round(0.25*ysize[0])]
    
    model = SimpPredictor(64)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01), loss='mse', metrics=['accuracy'])
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(16)
    testset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1)
    model.fit(dataset, epochs=100, shuffle=False)
    evaluation = model.evaluate(testset, return_dict=True)
    for name, value in evaluation.items():
        print(f"{name}: {value:.4f}")

class SimpPredictor(tf.keras.models.Model):
    def __init__(self, in_dims):
        super().__init__(self)
        self.l1 = layers.GRU(in_dims, return_sequences=True)
        self.l2 = layers.GRU(1_460, return_sequences=True)
        self.l3 = layers.Dropout(0.3)
        self.l4 = layers.GRU(940)
        self.l5 = layers.Dense(80)
        self.l6 = layers.Dense(3)
        self.l7 = layers.Dense(1)
    
    def call(self, inputs):
        x = inputs
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        return x


if __name__ == "__main__":
    main()
