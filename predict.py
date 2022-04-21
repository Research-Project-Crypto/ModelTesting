import numpy as np
import os
from multiprocessing import Process
import pandas as pd
import json
import argparse
import csv

import keras
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import models

### read vars
parser = argparse.ArgumentParser()
parser.add_argument("modelfile")
parser.add_argument("start_index")
parser.add_argument("stop_index")
parser.add_argument("frame_size")
parser.add_argument("data_file")
parser.add_argument("file_name")

args = parser.parse_args()

column_names = ['timestamp','open','close','high','low','volume','adosc','atr','macd','macd_signal','macd_hist','mfi','upper_band','middle_band','lower_band','rsi','difference_low_high','difference_open_close','target']

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"] = 'true'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], False)

field_info = [
    { "type": np.uint64, "count": 1 },
    { "type": np.double, "count": 17 },
    { "type": np.int64, "count": 1 }
]
BYTES_EIGHT = 8

def read_bin_full_file(file):
    f = open(file, 'rb')
    b = f.read(-1)

    BYTES_TO_READ = 0
    for field in field_info:
        BYTES_TO_READ += BYTES_EIGHT * field["count"]

    data = []
    BYTES_READ = 0
    for i in range(0, int(os.path.getsize(file) / BYTES_TO_READ)):
        row = []

        for idx, field in enumerate(field_info):
            row += np.frombuffer(b, dtype=field["type"], count=field["count"], offset=BYTES_READ).tolist()

            BYTES_READ += BYTES_EIGHT * field["count"]

        data.append(row)
    return np.array(data)


# train-test split
def df_split(df, frame_size):  
    X = df.drop(columns=['timestamp','target'], axis=0, inplace=False).to_numpy()
    Y = df['target'].to_numpy()

    X_test = []
    y_test = []
    for i in range(frame_size, X.shape[0]): 
        X_test.append(X[i-frame_size:i])
        y_test.append(Y[i])
    X_test, y_test = np.array(X_test), np.array(y_test)

    return X_test, y_test


# load model
model = models.load_model(args.modelfile)

data = read_bin_full_file(args.data_file)
df2 = pd.DataFrame(data, columns=column_names)

if int(args.start_index) == 0:
    df = df2.iloc[int(args.start_index):int(args.stop_index)]
else: 
    df = df2.iloc[int(args.start_index)-int(args.frame_size):int(args.stop_index)]


X_test, y_test = df_split(df, int(args.frame_size))

# normal extra long
## predict
y_pred = np.array(model.predict(X_test))

y_pred = np.argmax(y_pred, axis=-1)
y_pred = y_pred.flatten()

df = df.iloc[int(args.frame_size):]
length = len(df)/60
df['predictions'] = np.array(y_pred)
# print(df.head())

if os.path.exists(f"/home/joren/Coding/cryptodata/predictions/{args.modelfile.split('/')[-1]}/{args.file_name[:-4]}.csv"):
    df3 = pd.read_csv(f"/home/joren/Coding/cryptodata/predictions/{args.modelfile.split('/')[-1]}/{args.file_name[:-4]}.csv")
    df3 = df3.append(df, ignore_index=True)
    df3.drop_duplicates(subset="timestamp", keep='first', inplace=True)
    # df3.drop(columns=['open','close','high','low','volume','adosc','atr','macd','macd_signal','macd_hist','mfi','upper_band','middle_band','lower_band','rsi','difference_low_high','difference_open_close','target'])
    df3.to_csv(f"/home/joren/Coding/cryptodata/predictions/{args.modelfile.split('/')[-1]}/{args.file_name[:-4]}.csv", index=False, quoting=csv.QUOTE_NONE)
else:
    df.to_csv(f"/home/joren/Coding/cryptodata/predictions/{args.modelfile.split('/')[-1]}/{args.file_name[:-4]}.csv", index=False, quoting=csv.QUOTE_NONE)

# calculate profit/loss
total_percentage = 0
percentage = 0
position = 0
trades = 0
negative = False

for index, row in df.iterrows():

    if position == 1:
        percentage += row['close']

    if row['predictions'] == 1:
        position = 1
    elif row['predictions'] == 2:
        position = 0
        total_percentage += percentage
        trades += 1
        percentage = 0

if total_percentage < 0:
    negative = True

print({
    "percentage" : total_percentage,
    "negative": negative,
    "length_in_hours": length,
    "trades": trades
})

dictionary ={
    "percentage" : total_percentage,
    "negative": negative,
    "length_in_hours": length,
    "trades": trades
}

test = ''
with open(f"test_results/{args.modelfile.split('/')[-1]}_result.json", 'r+') as file:
    test = json.load(file)
    test["results"].append(dictionary)

# json_object = json.dumps(dictionary1, indent = 4)
with open(f"test_results/{args.modelfile.split('/')[-1]}_result.json", "w") as outfile:
    outfile.write(json.dumps(test, indent = 4))