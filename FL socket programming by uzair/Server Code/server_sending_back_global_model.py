import numpy as np
import socket
import pickle
import struct
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5001
NUM_ROUNDS = 5  # Number of federated learning rounds

# --- Data Preparation (Test Data for Evaluation) ---
X_full, y_full = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X_test = X_full[60000:] / 255.0
y_test = y_full[60000:].astype(int)

# --- Aggregation ---
def aggregate_params(params_list):
    if not params_list or len(params_list) != 2:
        return None
    agg_coef = (params_list[0]['coef_'] + params_list[1]['coef_']) / 2
    agg_intercept = (params_list[0]['intercept_'] + params_list[1]['intercept_']) / 2
    return {'coef_': agg_coef, 'intercept_': agg_intercept}

# --- Socket Communication ---
def send_model_params(conn, params):
    serialized_params = pickle.dumps(params)
    data_size = len(serialized_params)
    conn.sendall(struct.pack('!I', data_size))
    conn.sendall(serialized_params)

def receive_model_params(conn):
    try:
        data_size_bytes = conn.recv(4)
        if not data_size_bytes:
            return None
        data_size = struct.unpack('!I', data_size_bytes)[0]
        data = b''
        while len(data) < data_size:
            packet = conn.recv(4096)
            if not packet:
                return None
            data += packet
        params = pickle.loads(data)
        return params
    except Exception as e:
        print(f"Error during parameter reception: {e}")
        return None

# --- Main Server Logic ---
def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(2)
    print(f"Server listening on {SERVER_HOST}:{SERVER_PORT}")
    
    # --- Initialize global model with 1 sample per class ---
    initial_model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True)

    X_init = []
    y_init = []
    for i in range(10):
        idx = np.where(y_test == i)[0][0]  # Ensure one sample per digit
        X_init.append(X_test[idx])
        y_init.append(y_test[idx])

    X_init = np.array(X_init)
    y_init = np.array(y_init)

    initial_model.fit(X_init, y_init)
    initial_model.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # Explicitly set

    global_params = {'coef_': initial_model.coef_, 'intercept_': initial_model.intercept_}

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Starting Round {round_num} ---")
        client_params = []
        connections = []

        # Accept connections from 2 clients
        for i in range(2):
            conn, addr = server_socket.accept()
            print(f"Connection from {addr} established for round {round_num}")
            send_model_params(conn, global_params)
            connections.append(conn)

        # Receive local model updates from both clients
        for conn in connections:
            params = receive_model_params(conn)
            if params:
                print(f"Received local parameters from client.")
                client_params.append(params)
            conn.close()

        if len(client_params) == 2:
            global_params = aggregate_params(client_params)
            print(f"Round {round_num} aggregation complete.")
        else:
            print(f"Round {round_num}: Missing client updates.")

    # Final evaluation
    print("\nTraining complete. Evaluating final global model...")
    final_model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True)
    final_model.fit(X_init, y_init)  # Dummy fit to initialize structure
    final_model.coef_ = global_params['coef_']
    final_model.intercept_ = global_params['intercept_']
    final_model.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    y_pred = final_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nAggregated Model Final Accuracy: {accuracy:.4f}')

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    server_socket.close()

if __name__ == "__main__":
    main()
