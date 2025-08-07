import socket
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# A simple Linear Regression model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def aggregate_weights(weights1, weights2):
    # Simple federated averaging
    aggregated_weights = {}
    for key in weights1.keys():
        aggregated_weights[key] = (weights1[key] + weights2[key]) / 2
    return aggregated_weights

def evaluate_model(model, test_loader):
    # Evaluation logic
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28 * 28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    # Model and data setup
    input_dim = 28 * 28
    output_dim = 10
    server_model = LinearRegression(input_dim, output_dim)
    
    # ... Socket setup and weight reception logic ...

    # Evaluation and plotting
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    
    # ... After aggregating weights ...
    server_model.load_state_dict(aggregated_weights)
    accuracy = evaluate_model(server_model, test_loader)
    print(f"Final Aggregated Model Accuracy: {accuracy:.2f}%")
    
    # ... Plotting logic ...

if __name__ == '__main__':
    main()