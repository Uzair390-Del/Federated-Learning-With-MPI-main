import numpy as np
import time
import psutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import socket
import os
import struct

# Use localhost for single-PC testing
SERVER_HOST = '127.0.0.1'  
SERVER_PORT = 5001
BUFFER_SIZE = 4096

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1) / 255.0
x_test = np.expand_dims(x_test, axis=-1) / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Track CPU/time for training
cpu_start = psutil.cpu_percent(interval=None)
start_time = time.time()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

training_time = time.time() - start_time
cpu_used = psutil.cpu_percent(interval=None) - cpu_start

print(f"[CLIENT] Training CPU used: {cpu_used:.2f}%")
print(f"[CLIENT] Training time: {training_time:.2f} sec")

# Save model (simplified for demo)
model.save('client_model.h5')

def send_file(filename):
    cpu_start = psutil.cpu_percent(interval=None)
    start_time = time.time()

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_HOST, SERVER_PORT))

    filename_bytes = filename.encode()
    client_socket.send(struct.pack('I', len(filename_bytes)))
    client_socket.send(filename_bytes)

    with open(filename, 'rb') as f:
        while True:
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                break
            client_socket.sendall(bytes_read)

    client_socket.close()
    transfer_time = time.time() - start_time
    cpu_used = psutil.cpu_percent(interval=None) - cpu_start

    print(f"[CLIENT] File sent: {filename}")
    print(f"[CLIENT] Transfer CPU used: {cpu_used:.2f}%")
    print(f"[CLIENT] Transfer time: {transfer_time:.2f} sec")

if __name__ == "__main__":
    send_file('client_model.h5')