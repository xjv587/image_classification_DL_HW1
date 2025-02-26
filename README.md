# Deep Learning-Based Image Classification for SuperTuxKart Objects

This project focuses on developing a deep learning model to classify images from the SuperTuxKart dataset. The task involves distinguishing between six object categories (background, kart, pickup, nitro, bomb, and projectile) using a supervised learning approach.

# Key Components:
- Data Preprocessing: Implemented a PyTorch Dataset class to efficiently load and transform images into tensors for training.
- Model Development: Built two classification models:
  - Linear Classifier: A baseline model using a simple fully connected layer.
  - Multi-Layer Perceptron (MLP) Classifier: A non-linear classifier with multiple layers and ReLU activations.
- Loss Function: Implemented a custom softmax classification loss.
- Training Pipeline: Developed a training script to:
  - Load data efficiently.
  - Train models using Stochastic Gradient Descent (SGD).
  - Monitor performance with validation accuracy.
  - Save and load trained models for evaluation.

# Skills Demonstrated:
✅ Deep Learning: Implemented and trained neural networks using PyTorch.
✅ Data Engineering: Created a custom dataset loader, managed CSV-based labels, and applied transformations.
✅ Model Optimization: Experimented with network architectures, hyperparameters, and training strategies.
✅ End-to-End ML Pipeline: Built an entire workflow from data loading to model evaluation.
