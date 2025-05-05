#Μια επανάληψη για κατηγοριοποιηση svm 10 κλασεων σε υποσύνολο της cifar
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import copy
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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

def create_balanced_subset(x, y, num_samples_per_class):
    classes = np.unique(y)
    x_subset, y_subset = [], []

    for cls in classes:
        # Επιλέγει τυχαία num_samples_per_class δείγματα από την κάθε κλάση
        indices = np.where(y == cls)[0]
        selected_indices = np.random.choice(indices, size=num_samples_per_class, replace=False)
        x_subset.append(x[selected_indices])
        y_subset.append(y[selected_indices])

    # Συγχωνεύει τις λίστες για να πάρει τα τελικά δεδομένα
    x_subset = np.concatenate(x_subset, axis=0)
    y_subset = np.concatenate(y_subset, axis=0)

    return x_subset, y_subset

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

x_train_cop = copy.deepcopy(x_train)
x_test_cop = copy.deepcopy(x_test)

x_train_sub, y_train_sub = create_balanced_subset(x_train, y_train, 2000)
x_test_sub, y_test_sub = create_balanced_subset(x_test, y_test, 300)

x_train_cop_sub = copy.deepcopy(x_train_sub)
x_test_cop_sub = copy.deepcopy(x_test_sub)

x_train = np.concatenate(x_train)
y_train = np.array(y_train)

x_train_sub = x_train_sub.reshape(x_train_sub.shape[0], -1) / 255.0
x_test_sub = x_test_sub.reshape(x_test_sub.shape[0], -1) / 255.0
print("Images flattened and normalized.")

start_time = time.time()

# Step 2: Train the SVM model using libsvm (SVC from sklearn)
print("Training the SVM model...")
scaler = StandardScaler()
x_train_sub = scaler.fit_transform(x_train_sub)
print("Training data scaled.")
x_test_sub = scaler.transform(x_test_sub)
print("Test data scaled.")

# SVM Classifier με πολυωνυμικό kernel
svm_classifier = SVC(kernel='poly', degree=2, C=10)  # Πολυωνυμικός kernel με βαθμό 2 και C=10

# Εκπαίδευση
print("Training SVM classifier...")
svm_classifier.fit(x_train_sub, y_train_sub.ravel())  # y_train.ravel() για να γίνει μονοδιάστατος πίνακας

# Πρόβλεψη
print("Predicting test set...")
y_pred = svm_classifier.predict(x_test_sub)

# Αξιολόγηση
print("Accuracy on test set:", accuracy_score(y_test_sub, y_pred))

# Training accuracy
train_accuracy = svm_classifier.score(x_train_sub, y_train_sub.ravel())
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

print("Classification report:\n", classification_report(y_test_sub, y_pred))

print(f"Algorithm completed in {time.time() - start_time:.2f} seconds.")

# Υπολογισμός Confusion Matrix
cm = confusion_matrix(y_test_sub, y_pred)
print("Confusion Matrix:")
print(cm)

# Εμφάνιση Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_train_sub)).plot()
plt.title("Confusion Matrix - Test Set")
plt.show()

# Εμφάνιση παραδειγμάτων σωστής και εσφαλμένης κατηγοριοποίησης
print("Showing examples of correct and incorrect predictions...")
correct_indices = np.where(y_pred == y_test_sub)[0]
incorrect_indices = np.where(y_pred != y_test_sub)[0]

plt.figure(figsize=(3.5, 2.5))

# Εμφάνιση 2 σωστών προβλέψεων
for i, idx in enumerate(correct_indices[:4]):
    plt.subplot(2, 2, i + 1)
    plt.imshow(x_test_cop_sub[idx])  # Επιστροφή στη μορφή εικόνας
    plt.axis('off')
    plt.title(f"True: {y_test_sub[idx]}\nPred: {y_pred[idx]}", fontsize=10)

plt.tight_layout()
plt.show()


plt.figure(figsize=(3.5, 2.5))

# Εμφάνιση 2 εσφαλμένων προβλέψεων
for i, idx in enumerate(incorrect_indices[:4]):
    plt.subplot(2, 2, i + 1)
    plt.imshow(x_test_cop_sub[idx])  # Επιστροφή στη μορφή εικόνας
    plt.axis('off')
    plt.title(f"True: {y_test_sub[idx]}\nPred: {y_pred[idx]}", fontsize=10)

plt.tight_layout()
plt.show()