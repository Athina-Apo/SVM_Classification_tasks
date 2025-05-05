#SVM δυο κλάσεων με τη χρήση τετραγωνικού προγραμματισμού
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import copy
import time

# Η δική σου κλάση SVM
class SVM:
    def __init__(self, C=1.0):
        self.C = C
        self.alpha = None
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_labels = None

    def fit(self, X, y):
        print("Starting training of the SVM model...")
        n_samples, n_features = X.shape

        # Υπολογισμός του πίνακα Gram
        print("Computing the Gram matrix...")
        K = np.dot(X, X.T)
        P = matrix(np.outer(y, y) * K)  # P = Q_kl
        q = matrix(-np.ones(n_samples))  # q_i = -1 for all i
        A = matrix(y.reshape(1, -1), (1, n_samples), tc='d')  # A = y^T
        b = matrix(0.0)  # b = 0

        # Περιορισμοί: 0 <= alpha_k <= C
        print("Setting up constraints...")
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

        # Επίλυση του προβλήματος QP
        print("Solving the quadratic programming problem...")
        solution = solvers.qp(P, q, G, h, A, b)
        self.alpha = np.ravel(solution['x'])  # Lagrange multipliers
        print("QP problem solved.")

        # Εντοπισμός support vectors
        print("Identifying support vectors...")
        support_indices = self.alpha > 1e-5
        self.support_vectors = X[support_indices]
        self.support_labels = y[support_indices]
        self.alpha = self.alpha[support_indices]

        # Υπολογισμός του διανύσματος βαρών w
        print("Computing weight vector w...")
        self.w = np.sum(self.alpha[:, None] * self.support_labels[:, None] * self.support_vectors, axis=0)

        # Υπολογισμός του bias b
        print("Computing bias b...")
        K_index = np.argmax(self.alpha)  # Find the largest alpha
        x_k = self.support_vectors[K_index]
        y_k = self.support_labels[K_index]
        epsilon_k = 1e-5
        self.b = np.mean(self.support_labels - np.dot(self.support_vectors, self.w))
        print("Training completed.")

    def predict(self, X):
        print("Predicting labels...")
        decision = np.dot(X, self.w) + self.b
        return np.sign(decision)

# Βοηθητική συνάρτηση για την αποσυμπίεση των δεδομένων
def unpickle(file):
    print(f"Unpacking file: {file}")
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    print(f"File {file} unpacked.")
    return data

# Φόρτωση δεδομένων CIFAR-10
def load_cifar_batch(file: str):
    print(f"Loading CIFAR-10 batch: {file}")
    data_dict = unpickle(file)
    images = data_dict[b'data']
    labels = data_dict[b'labels']
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Rearranging to (N, 32, 32, 3)
    print(f"Batch {file} loaded successfully.")
    return images, labels

# CIFAR-10 paths
cifar_data_dir = ""  # Update this path to your CIFAR-10 data folder
x_train, y_train = [], []
print("Loading CIFAR-10 training batches...")
for i in range(1, 6):  # CIFAR-10 training batches
    file_path = f"{cifar_data_dir}data_batch_{i}"
    images, labels = load_cifar_batch(file_path)
    x_train.append(images)
    y_train.extend(labels)
print("CIFAR-10 training batches loaded.")
x_train = np.concatenate(x_train)
y_train = np.array(y_train)


print("Loading CIFAR-10 test batch...")
x_test, y_test = load_cifar_batch(f"{cifar_data_dir}test_batch")
y_test = np.array(y_test)
print("CIFAR-10 test batch loaded.")



# Φιλτράρισμα για 2 κλάσεις
class_0, class_1 = 0, 1
print(f"Filtering classes {class_0} and {class_1}...")
def filter_classes(x, y, class_0, class_1):
    mask = (y == class_0) | (y == class_1)
    x, y = x[mask], y[mask]
    y = np.where(y == class_0, -1, 1)  #Μετατροπή των ετικετών σε -1 και +1
    return x, y

x_train, y_train = filter_classes(x_train, y_train, class_0, class_1)
x_test, y_test = filter_classes(x_test, y_test, class_0, class_1)
print(f"Filtered data to include only classes {class_0} and {class_1}.")

x_train_cop = copy.deepcopy(x_train)
x_test_cop = copy.deepcopy(x_test)

# Επίπεδες και κανονικοποιημένες εικόνες
print("Flattening and normalizing images...")
start_time = time.time()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
print("Images flattened and normalized.")

# Κλιμάκωση χαρακτηριστικών
print("Scaling features...")
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print("Features scaled.")

# Εκπαίδευση SVM
print("Training the SVM model...")
svm = SVM(C=0.1)
svm.fit(x_train, y_train)
print("SVM model trained.")

# Αξιολόγηση στο test set
print("Evaluating the model on test set...")
y_pred_test = svm.predict(x_test)
accuracy_test = np.mean(y_pred_test == y_test)
print(f"Test Set Accuracy: {accuracy_test * 100:.2f}%")


# Αξιολόγηση στο train set
print("Evaluating the model on train set...")
y_pred_train = svm.predict(x_train)
accuracy_train = np.mean(y_pred_train == y_train)
print(f"Train Set Accuracy: {accuracy_train * 100:.2f}%")


print(f"Algorithm completed in {time.time() - start_time:.2f} seconds.")

# Εμφάνιση του confusion matrix για το test set
cm = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:")
print(cm)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-1, 1]).plot()
plt.title("Confusion Matrix - Test Set")
plt.show()

# Εμφάνιση παραδειγμάτων σωστής και εσφαλμένης κατηγοριοποίησης
print("Showing examples of correct and incorrect predictions...")
correct_indices = np.where(y_pred_test == y_test)[0]
incorrect_indices = np.where(y_pred_test != y_test)[0]

plt.figure(figsize=(3.5, 2.5))

# Εμφάνιση 5 σωστών προβλέψεων
for i, idx in enumerate(correct_indices[:2]):
    plt.subplot(2, 2, i + 1)
    plt.imshow(x_test_cop[idx])
    plt.axis('off')
    plt.title(f"True: {y_test[idx]}\nPred: {y_pred_test[idx]}", fontsize=10)

# Εμφάνιση 5 εσφαλμένων προβλέψεων
for i, idx in enumerate(incorrect_indices[:2]):
    plt.subplot(2, 2, i + 3)
    plt.imshow(x_test_cop[idx])
    plt.axis('off')
    plt.title(f"True: {y_test[idx]}\nPred: {y_pred_test[idx]}", fontsize=10)

plt.tight_layout()
plt.show()
