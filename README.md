ANN Classification on Fashion-MNIST using PyTorch

A complete PyTorch implementation of a classic Artificial Neural Network (ANN) to classify grayscale fashion images from the Fashion-MNIST dataset. This project demonstrates how to build, train, evaluate, and tune an ANN for image classification using PyTorch’s deep learning capabilities.

Fashion-MNIST is a benchmark dataset consisting of 70,000 grayscale 28×28 images of clothing items in 10 categories. It is widely used as a modern replacement for the original MNIST dataset for evaluating machine learning models.

🧠 Features

🧪 Load and preprocess the Fashion-MNIST dataset

🔢 Build a fully connected ANN using PyTorch

📈 Train and evaluate the model

🎛️ Experiment with hyperparameters like learning rate, batch size, epochs, and architecture

📊 Visualize performance (loss & accuracy)

📦 Repository Structure
ANN-classification-fashion-mnist-using-Pytorch/
├── Ann_using_pytorch.ipynb              # Main ANN implementation
├── Ann_hyperparameter_using_pytorch.ipynb # Hyperparameter tuning notebook
├── README.md                            # Project overview & instructions
├── models/                              # (Optional) Saved PyTorch models
└── requirements.txt                     # Python dependencies

🚀 Getting Started
1. Clone the Repository
git clone https://github.com/Abraar77/ANN-classification-fashion-mnist-using-Pytorch.git
cd ANN-classification-fashion-mnist-using-Pytorch

🛠️ Installation

Install dependencies (recommended in a virtual environment):

pip install torch torchvision matplotlib numpy


Alternatively, you can install from a requirements.txt if provided:

pip install -r requirements.txt

📌 What You’ll Learn

How to load Fashion-MNIST using torchvision.datasets

How to define a neural network with torch.nn.Module

How to train & evaluate the model using PyTorch

How to monitor training loss & accuracy

How to tune hyperparameters for better performance

🧩 Example Code Snippet

Here’s a simplified overview of how the dataset and model might be set up:

import torch
from torchvision import datasets, transforms

# Transform & load data
transform = transforms.ToTensor()
trainset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Build a simple ANN
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(28*28, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
    torch.nn.LogSoftmax(dim=1)
)

# Loss & optimizer
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


Note: The above is a representative snippet—your notebooks contain the full implementation.

📊 Results

🔥 Expect the model to learn meaningful representations of clothing images and classify them into the 10 categories like T-shirt/top, Trouser, Coat, Sneaker, etc.

Performance depends on model architecture and hyperparameter choices; thorough tuning can substantially improve accuracy.

💡 Tips for Improvement

Add normalization to input transforms

Increase network depth / hidden units

Experiment with learning rates & optimizers

Use dropout or batch normalization layers

Visualize confusion matrix & misclassified examples

📄 License

This project is open-source — feel free to share, modify, and build upon it!
