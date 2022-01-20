import numpy as np
import time
import os

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

    print("Reading", BYTES_TO_READ, "bytes / row")

    data = []
    BYTES_READ = 0
    for i in range(0, int(os.path.getsize(file) / BYTES_TO_READ)):
        row = []

        for idx, field in enumerate(field_info):
            row += np.frombuffer(b, dtype=field["type"], count=field["count"], offset=BYTES_READ).tolist()

            BYTES_READ += BYTES_EIGHT * field["count"]

        data.append(row)

    return data

start_time = time.time()
data = read_bin_full_file('./data/output/PERLUSDT.bin')

if True:
    for candle in data:
        print(candle)

        time.sleep(0.3)

print(f"Difference: {(time.time() - start_time) * 1000}ms")