import numpy as np
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf  # Use tensorflow to set up neural network
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from util import func_confusion_matrix  # ensure util.py in the directory


# load (downloaded if needed) the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# transform each image from 28 by28 to a 784 pixel vector
pixel_count = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], pixel_count).astype('float32')
x_test = x_test.reshape(x_test.shape[0], pixel_count).astype('float32')

# normalize inputs from gray scale of 0-255 to values between 0-1
x_train = x_train / 255
x_test = x_test / 255

# Please write your own codes in responses to the homework assignment 4
"""
To define a FNN model, you will need to specify the following hyper-parameters:
1.number of hidden layers and number of neural units for each layer;
2.learning rate
3.activation function (sigm, tanh_opt) and
4.output function (‘sigm’,’linear’, ‘softmax’);
"""

"""
Question 1. Please further split the 60,000 training images (and labels) into two subsets: 50,000 images, and 10,000 images.
Use these two subsets for training models and validation purposes.
In particular, you will train your FNN model using the 50,000 images and labels,
and apply the trained model over the rest 10,000 images for evaluation purposes.

Please specify at least three sets of hyper-parameters (see the above).
For each set, call the third-party functions or tensorflow to train a FNN model on the training samples (50,000 images in this case)
, and apply the learned model over the validation set (10,000 images in this case).
For each model and its results, please compute its confusion matrix, average accuracy, per-class Precision and Recall.
Report the model that achieves the top accuracy.

A sample function for calculating confusion matrix is provided in ‘util.py’
"""

# We do not have to make a neural network from scratch
# Use keras library

# 1. Split the dataset
# Split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=10000, random_state=42)

# 2. Define and train the model
# Define the model function to build the model


def build_model(hidden_layers, activation, output_activation, learning_rate):
    # build the sequential model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(784,)))

    # add a given number of dense layers with specified neurons into the model
    for units in hidden_layers:

        model.add(tf.keras.layers.Dense(units, activation=activation))
     # add 10 neurons with a specific activation function
    model.add(tf.keras.layers.Dense(10, activation=output_activation))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Define hyper-parameters to try
hyperparameter_combinations = [
    ([512], 'sigmoid', 'softmax', 0.001),
    ([256, 128], 'tanh', 'softmax', 0.001),
    ([128, 128, 128], 'sigmoid', 'softmax', 0.001)
]


# 3. Evaluate the model
# Train and evaluate each model using the specified hyper-parameters
best_accuracy = 0
best_model = None
best_conf_matrix = None

# Go through the hidden layers, activation function, output Adam and learning rate in the hyperparameters.
for hidden_layers, activation, output_activation, learning_rate in hyperparameter_combinations:
    model = build_model(hidden_layers, activation,
                        output_activation, learning_rate)
    model.fit(x_train, y_train, epochs=10, batch_size=32,
              validation_data=(x_val, y_val))
    val_loss, val_accuracy = model.evaluate(x_val, y_val)
    print(
        f"Model with structure {hidden_layers} - "
        f"Validation Accuracy: {val_accuracy}"
    )

    # if val_accuracy > best_accuracy:
    #     best_accuracy = val_accuracy
    #     best_model = model

    # Predict the labels for the validation set
    y_pred = model.predict(x_val)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate the confusion matrix and related metrics
    conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(
        y_val, y_pred_classes)

    # Update best model information if current model is the best
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model = model
        best_conf_matrix = conf_matrix
        print(
            f"New best model with structure {hidden_layers}"
            f"found with accuracy: {val_accuracy}"
        )
        print(f"Confusion Matrix:\n{conf_matrix}")


# Report the best model
print(f"The best model has a validation accuracy of {best_accuracy}")

# Fixed issue with confusion matrix not showing up
if best_conf_matrix is not None:
    print(f"The best model's confusion matrix:\n{best_conf_matrix}")
else:
    print("No valid confusion matrix available.")

# Question 2
"""
Question 2. Apply the top ranked model over the testing samples (10,000 images). 
Call the above function to compute the confusion matrix, average accuracy, 
per-class precision/recall rate.  
In addition, select and visualize TEN testing images for which your mode made wrong predications. 
Try to analyze the reasons of these failure cases. 
"""

# Need to apply the best model identified from Question 1 to the testing samples
# Computer and report the performance
# Visualize the ten images
# Top ranked Model

# 1. Evaluate Model on test Set
test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
y_pred_test = best_model.predict(x_test)
y_pred_classes_test = np.argmax(y_pred_test, axis=1)

# Calculate the confusion matrix and related metrics for the test set
test_conf_matrix, test_accuracy, test_recall_array, test_precision_array = func_confusion_matrix(
    y_test, y_pred_classes_test)

# Print out the output for analysis
print("Test Confusion Matrix:")
print(test_conf_matrix)  # Show the test confusion matrix
print(f"Test Average Accuracy: {test_accuracy}")
print(f"Test Per-class Precision: {test_precision_array}")
print(f"Test Per-class Recall: {test_recall_array}")

# Figure out what was misclassified
misclassified_indices = np.where(y_pred_classes_test != y_test)[0]
selected_indices = np.random.choice(misclassified_indices, 10, replace=False)

# Plotting misclassified images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

# Go through and help plot the axes.
for i, ax in enumerate(axes):
    index = selected_indices[i]
    ax.imshow(x_test[index].reshape(28, 28), cmap='gray')
    ax.set_title(
        f'Predicted: {y_pred_classes_test[index]} / Actual: {y_test[index]}')
    ax.axis('off')

plt.tight_layout()
plt.show()
