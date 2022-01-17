import numpy as np
import time
import os

BYTE_SIZE = 8
FIELD_COUNT = 16

BYTES_TO_READ = BYTE_SIZE * FIELD_COUNT

def read_bin_full_file(file):
    f = open(file, 'rb')
    b = f.read(-1)

    data = []
    for i in range(0, int(os.path.getsize(file) / BYTES_TO_READ)):
        data.append(np.frombuffer(b, dtype=np.uint64, count=1, offset=i * BYTES_TO_READ).tolist() + np.frombuffer(b, dtype=np.double, count=15, offset=BYTES_TO_READ * i + 8).tolist())

    return data

start_time = time.time()
data = read_bin_full_file('./data/output/VETUSDT.bin')
print(f"Difference: {(time.time() - start_time) * 1000}ms")
