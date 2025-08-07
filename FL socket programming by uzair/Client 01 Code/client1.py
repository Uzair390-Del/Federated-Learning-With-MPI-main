import numpy as np
import socket
import pickle
import struct
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# --- Configuration ---
serverIP = input("Enter Server IP: ")
machineName = int(input("Enter Machine Name (1 or 2): "))
SERVER_PORT = 5001

# --- Data Preparation ---
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # Normalize pixel values
y = y.astype(int)

# Partition the training data
if machineName == 1:
    X_train = X[:30000]
    y_train = y[:30000]
elif machineName == 2:
    X_train = X[30000:60000]
    y_train = y[30000:60000]
else:
    raise ValueError("Invalid machine name. Please use 1 or 2.")

# --- Model Definition and Training ---
# SGDClassifier is a linear classifier that uses SGD for training.
# It's an excellent choice for large-scale datasets.
model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3, random_state=42)

stTime = time.time()
model.fit(X_train, y_train)
endTime = time.time()
training_time = endTime - stTime

print(f"Client {machineName} Training Time: {training_time:.2f} seconds")

# --- Get the model parameters ---
# Scikit-learn models don't have a simple get_weights() method like Keras.
# We'll get the coefficients (weights) and intercept (bias).
weights = model.coef_
intercept = model.intercept_

# --- Socket Communication ---
def send_model_params(params):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((serverIP, SERVER_PORT))
        print(f"Connected to server at {serverIP}:{SERVER_PORT}")
        
        serialized_params = pickle.dumps(params)
        
        data_size = len(serialized_params)
        client_socket.sendall(struct.pack('!I', data_size))
        client_socket.sendall(serialized_params)
        
        print("Model parameters sent successfully!")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client_socket.close()

if __name__ == "__main__":
    send_model_params({'coef_': weights, 'intercept_': intercept})
