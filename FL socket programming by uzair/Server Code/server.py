import numpy as np
import socket
import pickle
import struct
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 6465

# --- Data Preparation (Server only loads test data) ---
X_full, y_full = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X_test = X_full[60000:] / 255.0
y_test = y_full[60000:].astype(int)

# --- Aggregation Logic ---
def aggregate_params(params_list):
    if not params_list or len(params_list) != 2:
        return None
    
    # Federated averaging for coefficients and intercepts
    agg_coef = (params_list[0]['coef_'] + params_list[1]['coef_']) / 2
    agg_intercept = (params_list[0]['intercept_'] + params_list[1]['intercept_']) / 2
    
    return {'coef_': agg_coef, 'intercept_': agg_intercept}

# --- Socket Communication ---
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
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(2)
    print(f"Server listening on {SERVER_HOST}:{SERVER_PORT}")
    
    client_params = []
    
    for i in range(2):
        conn, addr = server_socket.accept()
        print(f"Connection from {addr} has been established!")
        params = receive_model_params(conn)
        if params:
            client_params.append(params)
            print(f"Parameters from client {i+1} received successfully!")
        conn.close()

    print("\nAll client parameters received. Starting aggregation...")

    if len(client_params) == 2:
        aggregated_params = aggregate_params(client_params)
        
        # Create a new model instance and set the aggregated parameters
        final_model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True)
        # Note: We need to fit once to initialize the model before setting weights
        final_model.fit(X_test[:10], y_test[:10]) 
        final_model.coef_ = aggregated_params['coef_']
        final_model.intercept_ = aggregated_params['intercept_']
        final_model.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        y_pred = final_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'\nAggregated Model Final Accuracy: {accuracy:.4f}')

        # --- Plotting ---
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    server_socket.close()

if __name__ == "__main__":
    main()