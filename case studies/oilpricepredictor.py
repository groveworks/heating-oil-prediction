import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("B:/Documents/Github/grove-cost-predictors/data/heating oil prices.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['days'] = pd.DatetimeIndex(df['date']).day
    df['months'] = pd.DatetimeIndex(df['date']).month
    df['years'] = pd.DatetimeIndex(df['date']).year
    dff = df.drop(columns=['date'])
    Xy = np.array(dff)
    print(Xy.shape) # <-- work from there so that the shape of the data is reformatted in such a way that the previous price gets indexed
    print(Xy)
    exit()
    for i, row in df.iterrows():
        if i == 0:
            pass
        else:
            df.at[i, 'previous_price'] = df.at[i-1, ['weekly__dollars_per_gallon', 'day', 'month', 'year']]
    # Design the machine learning algorithm
    model = SimpPredictor(4)
    model.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])
    y_train = df['weekly__dollars_per_gallon']
    X_train = df.drop(columns=['weekly__dollars_per_gallon', 'date'])
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(600).batch(64)
    model.fit(dataset, epochs=10)

class SimpPredictor(tf.keras.models.Model):
    def __init__(self, in_dims):
        super().__init__(self)
        self.l1 = layers.GRU(in_dims, return_sequences=True)
        self.l2 = layers.GRU(20)
        self.l3 = layers.Dropout(0.3)
        self.l4 = layers.GRU(10)
        self.l5 = layers.Dense(2)
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