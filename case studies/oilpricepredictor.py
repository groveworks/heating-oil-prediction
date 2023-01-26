import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import Sequential
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def main():
    file = Path("data/ctoilpricesfeats.csv")
    if file.is_file():
        df = pd.read_csv("data/heating oil prices.csv")
        df['date'] = pd.to_datetime(df['date'])
        df['days'] = pd.DatetimeIndex(df['date']).day
        df['months'] = pd.DatetimeIndex(df['date']).month
        df['years'] = pd.DatetimeIndex(df['date']).year
        dff = df.drop(columns=['date'])
        dff.to_csv("data/ctoilpricesfeats.csv", index=False)
    else:
        dff = pd.read_csv("data/ctoilpricesfeats.csv")
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
    # Design the machine learning algorithm
    model = SimpPredictor(64)
    model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
    dataset = tf.data.Dataset.from_tensor_slices((Xs, ys)).batch(16)
    model.fit(dataset, epochs=100, shuffle=False)

class SimpPredictor(tf.keras.models.Model):
    def __init__(self, in_dims):
        super().__init__(self)
        self.l1 = layers.GRU(in_dims, return_sequences=True)
        self.l2 = layers.GRU(64, return_sequences=True)
        self.l3 = layers.Dropout(0.2)
        self.l4 = layers.GRU(32, return_sequences=True)
        self.l5 = layers.Dense(16)
        self.l6 = layers.Dense(8)
        self.l6 = layers.Dense(1)
    
    def call(self, inputs):
        x = inputs
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        return x


if __name__ == "__main__":
    main()
