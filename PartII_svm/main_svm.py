import numpy as np
import download_data as dl
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn import metrics
from conf_matrix import func_confusion_matrix

# step 1: load data from csv file.
data = dl.download_data('crab.csv').values

n = 200  # number of data points
# split data
S = np.random.permutation(n)  # randomly shuffle data before we split it
# 100 training samples
Xtr = data[S[:100], :6]  # feature columns
Ytr = data[S[:100], 6:]  # labels
# 100 testing samples
X_test = data[S[100:], :6]  # feature columns
Y_test = data[S[100:], 6:].ravel()  # test labels

# step 2 randomly split Xtr/Ytr into two even subsets: use one for training, another for validation.
############# placeholder 1: training/validation #######################
n2 = len(Xtr)  # number of training samples
S2 = np.random.permutation(n2)  # random shuffle
half = n2 // 2  # make sure that the model is evenly trained and processed

# subsets for training models
x_train = Xtr[S2[:half]]
y_train = Ytr[S2[:half]].ravel()  # flatten the array
# subsets for validation
x_validation = Xtr[S2[half:]]
y_validation = Ytr[S2[half:]].ravel()  # flatten the array
############# placeholder end #######################

# step 3 Model selection over validation set
# consider the parameters C, kernel types (linear, RBF etc.) and kernel
# parameters if applicable.


# 3.1 Plot the validation errors while using different values of C ( with other hyperparameters fixed)
#  keeping kernel = "linear"
############# placeholder 2: Figure 1#######################
# Create a range of C values from 0.1 to 10
c_range = np.linspace(0.1, 10, 100)
svm_c_error = []  # hold the svm errors
for c_value in c_range:  # go through the different c value range
    model = svm.SVC(kernel='linear', C=c_value)
    model.fit(X=x_train, y=y_train)
    error = 1. - model.score(x_validation, y_validation)
    svm_c_error.append(error)

# Set the figure size to match your specific requirements
plt.figure(figsize=(10, 6))  # plot the figures and the linear svm and c values
plt.plot(c_range, svm_c_error, linestyle='-', color='blue')  # Use a solid line
plt.title('Linear SVM')
plt.xlabel('C values')
plt.ylabel('Validation Error')
plt.xticks(np.arange(0, 11, 1))  # Set x-ticks to be whole numbers from 0 to 10
# Set y-ticks with a step that fits the error range
plt.yticks(np.arange(0, round(max(svm_c_error), 2) + 0.01, step=0.01))
plt.show()
############# placeholder end #######################


# 3.2 Plot the validation errors while using linear, RBF kernel, or Polynomial kernel ( with other hyperparameters fixed)
# 3.2 Plot the validation errors while using linear, RBF kernel, or Polynomial kernel (with other hyperparameters fixed)
############# placeholder 3: Figure 2#######################
kernel_types = ['linear', 'poly', 'rbf']
svm_kernel_error = []
for kernel_value in kernel_types:
    model = svm.SVC(kernel=kernel_value, C=1)  # Corrected variable name here
    model.fit(x_train, y_train)
    error = 1. - model.score(x_validation, y_validation)
    svm_kernel_error.append(error)

plt.plot(kernel_types, svm_kernel_error, marker='o')
plt.title('SVM by Kernels')
plt.xlabel('Kernel')
plt.ylabel('error')
plt.xticks(kernel_types)
plt.show()
############# placeholder end #######################


# step 4 Select the best model and apply it over the testing subset
############# placeholder 4:testing  #######################

best_kernel = 'poly'  # best kernel type due to lowest validation error
best_c = 1  # poly had many that were the "best"
model = svm.SVC(kernel=best_kernel, C=best_c)
model.fit(X=x_train, y=y_train)

############# placeholder end #######################


# step 5 evaluate your results in terms of accuracy, real, or precision.

############# placeholder 5: metrics #######################
# func_confusion_matrix is not included
# You might re-use this function for the Part I.
y_pred = model.predict(X_test)
conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(
    Y_test, y_pred)

print("Confusion Matrix: ")
print(conf_matrix)
print("Average Accuracy: {}".format(accuracy))
print("Per-Class Precision: {}".format(precision_array))
print("Per-Class Recall: {}".format(recall_array))

############# placeholder end #######################

############# placeholder 6: success and failure examples #######################
# Success samples: samples for which you model can correctly predict their labels
# Failure samples: samples for which you model can not correctly predict their labels
# Identify success and failure cases
correct_indices = np.where(y_pred == Y_test)[0]
incorrect_indices = np.where(y_pred != Y_test)[0]

# Select up to 5 examples from each category for visualization
success_samples = np.random.choice(correct_indices, 5, replace=False)
failure_samples = np.random.choice(incorrect_indices, 5, replace=False)

# Define a function to plot crab data samples


def plot_crab_samples(indices, title):
    plt.figure(figsize=(10, 2))
    for i, idx in enumerate(indices):
        plt.subplot(1, 5, i + 1)
        # Assuming that the first feature in the dataset is significant for visualization,
        # it might represent some size measure or similar
        # Here, we just plot the index as a proxy for lacking specific visualization details
        plt.bar(['Species', 'FL', 'RW', 'L', 'W', 'D'], X_test[idx])
        plt.title(f'Label: {Y_test[idx]}\nPred: {y_pred[idx]}')
        plt.xticks(rotation=45)
    plt.suptitle(title)
    plt.tight_layout()


# Plot successful cases
plot_crab_samples(success_samples, "Success Examples")

# Plot failure cases
plot_crab_samples(failure_samples, "Failure Examples")
plt.show()

############# placeholder end #######################
