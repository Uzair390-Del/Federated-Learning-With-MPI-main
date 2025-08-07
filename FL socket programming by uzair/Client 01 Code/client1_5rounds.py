import numpy as np
import socket
import pickle
import struct
import time
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Configuration ---
serverIP = input("Enter Server IP: ")
machineName = int(input("Enter Machine Name (1 or 2): "))
SERVER_PORT = 5001
NUM_ROUNDS = 5  # Number of federated learning rounds

# --- Data Preparation ---
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0
y = y.astype(int)

if machineName == 1:
    X_train = X[:30000]
    y_train = y[:30000]
elif machineName == 2:
    X_train = X[30000:60000]
    y_train = y[30000:60000]
else:
    raise ValueError("Invalid machine name. Please use 1 or 2.")

# --- Socket Communication Functions ---
def send_model_params(client_socket, params):
    serialized_params = pickle.dumps(params)
    data_size = len(serialized_params)
    client_socket.sendall(struct.pack('!I', data_size))
    client_socket.sendall(serialized_params)

def receive_model_params(client_socket):
    try:
        data_size_bytes = client_socket.recv(4)
        if not data_size_bytes:
            return None
        data_size = struct.unpack('!I', data_size_bytes)[0]
        data = b''
        while len(data) < data_size:
            packet = client_socket.recv(4096)
            if not packet:
                return None
            data += packet
        params = pickle.loads(data)
        return params
    except Exception as e:
        print(f"Error during parameter reception: {e}")
        return None

# --- Main Client Logic ---
def main():
    model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, max_iter=1, warm_start=True, random_state=42)
    model.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # Set once for the model

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Starting Round {round_num} ---")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            client_socket.connect((serverIP, SERVER_PORT))
            print(f"Connected to server. Receiving global model for round {round_num}...")

            global_params = receive_model_params(client_socket)

            if global_params:
                coef = global_params.get('coef_')
                intercept = global_params.get('intercept_')
                if coef is not None and intercept is not None:
                    if coef.shape[1] == X_train.shape[1] and coef.shape[0] == 10:
                        model.coef_ = coef
                        model.intercept_ = intercept
                    else:
                        print(f"Shape mismatch: coef_ shape {coef.shape}, expected (10, {X_train.shape[1]})")
                        continue
                else:
                    print("Invalid global parameters received.")
                    continue

            print("Starting local training...")
            stTime = time.time()
            model.fit(X_train, y_train)
            endTime = time.time()
            training_time = endTime - stTime
            print(f"Local training for round {round_num} complete in {training_time:.2f}s.")

            local_params = {'coef_': model.coef_, 'intercept_': model.intercept_}
            print("Sending local parameters back to server...")
            send_model_params(client_socket, local_params)
            print("Parameters sent successfully!")

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            client_socket.close()

if __name__ == "__main__":
    main()

