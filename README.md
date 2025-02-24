# image_classification_DL_HW1
This is a simple deep network to classify images from SuperTuxKart.

# Data Loader
First, to implement a data loader for the SuperTuxKart dataset. Complete the __init__, __len__, and the __getitem__ of the SuperTuxDataset class in the utils.py.
-	The __len__ function should return the size of the dataset.
-	The __getitem__ function should return a tuple of image, label. The image should be a torch.Tensor of size (3,64,64) with range [0,1], and the label should be int.
-	Labels and the corresponding image paths are saved in labels.csv, their headers are file and label. There are 6 classes of objects. Make sure label background corresponds to 0, kart is 1, pickup is 2, nitro is 3, bomb is 4 and projectile 5.
Once finish, you can visualize some of the images by their classes using: python3 -m homework.visualize_data data/valid

# Linear Model
Implement the LinearClassifier class in models.py. Define the linear model and all layers in the __init__ function, then implement forward. Your forward function receives a (B,3,64,64) tensor as an input and should return a (B,6) torch.Tensor (one value per class). You can earn these full credits without training the model, just from the correct model definition. B stands for batch size.

# Classification Loss
Next, we'll implement the ClassificationLoss in models.py. We will later use this loss to train our classifiers. You should implement the log-likelihood of a softmax classifier.

# Training the linear model
Train your linear model in train.py. You should implement the full training procedure
-	create a model, loss, optimizer
-	load the data: train and valid
-	Run SGD for several epochs
-	Save your final model, using save_model

# MLP Model
Implement the MLPClassifier class in models.py. The inputs and outputs to the multi-layer perceptron are the same as the linear classifier. However, now you're learning a non-linear function.
