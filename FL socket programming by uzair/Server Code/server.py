import socket
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --- Federated Averaging ---
def federated_average(weights_list, biases_list, sample_sizes):
    total_samples = sum(sample_sizes)
    avg_weights = np.sum([w * n for w, n in zip(weights_list, sample_sizes)], axis=0) / total_samples
    avg_bias = np.sum([b * n for b, n in zip(biases_list, sample_sizes)], axis=0) / total_samples
    return avg_weights, avg_bias

# --- Socket Setup ---
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 65432

# Load test data for evaluation
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# mnist_test = datasets.MNIST(root='./data', train=False, transform=transform)
# Load test data for evaluation (ADD download=True)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)  # Fix: Added download=True
X_test = mnist_test.data.numpy().reshape(-1, 784) / 255.0
y_test = mnist_test.targets.numpy()

# Initialize global model
global_model = LogisticRegression(max_iter=1, warm_start=True, solver='saga')
global_model.coef_ = np.zeros((10, 784))  # 10 classes, 784 features
global_model.intercept_ = np.zeros(10)

# Track accuracy over rounds
accuracies = []

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Server listening on {HOST}:{PORT}")

    for round in range(5):  # 5 communication rounds
        print(f"\n=== Round {round + 1} ===")
        
        # Collect weights from clients
        client_weights = []
        client_biases = []
        sample_sizes = []

        for _ in range(2):  # Expect 2 clients
            conn, addr = s.accept()
            print(f"Connected by {addr}")

            data = conn.recv(4096 * 16)  # Large buffer for weights
            weights, bias, n_samples = pickle.loads(data)
            client_weights.append(weights)
            client_biases.append(bias)
            sample_sizes.append(n_samples)

            # Send current global model to client
            conn.sendall(pickle.dumps((global_model.coef_, global_model.intercept_)))
            conn.close()

        # Aggregate weights
        global_weights, global_bias = federated_average(client_weights, client_biases, sample_sizes)
        global_model.coef_ = global_weights
        global_model.intercept_ = global_bias

        # Evaluate
        accuracy = global_model.score(X_test, y_test)
        accuracies.append(accuracy)
        print(f"Global Accuracy: {accuracy * 100:.2f}%")

# Plot results
plt.plot(range(1, 6), accuracies, marker='o')
plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.title("Federated Learning Performance")
plt.grid()
plt.show()