#SVM για διαφορετικες παραμετρους σε υποσυνολο της Cifar 10 για 10 κλάσεις
import numpy as np
import pickle
import pandas as pd
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


x_train_sub = x_train_sub.reshape(x_train_sub.shape[0], -1) / 255.0
x_test_sub = x_test_sub.reshape(x_test_sub.shape[0], -1) / 255.0
print("Images flattened and normalized.")

# Διαφορετικές παράμετροι για το SVM
kernels = ['linear', 'poly', 'rbf']
C_values = [0.1, 1, 10, 100]
degrees = [2, 3, 5]  # Για polynomial kernel
gammas = [0.1, 0.5, 1]  # Για RBF kernel

# Πίνακας για αποθήκευση αποτελεσμάτων
results = []

# Preprocessing δεδομένων
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_sub)
x_test_scaled = scaler.transform(x_test_sub)

# Εκτέλεση του SVM για διαφορετικές παραμέτρους
for kernel in kernels:
    for C in C_values:
        if kernel == 'poly':  # Δοκιμάζει διαφορετικούς βαθμούς για polynomial
            for degree in degrees:
                print(f"Training SVM with kernel={kernel}, C={C}, degree={degree}")
                start_time = time.time()

                # Δημιουργία και εκπαίδευση του μοντέλου
                svm_classifier = SVC(kernel=kernel, C=C, degree=degree)
                svm_classifier.fit(x_train_scaled, y_train_sub.ravel())

                # Αξιολόγηση
                train_accuracy = svm_classifier.score(x_train_scaled, y_train_sub.ravel())
                y_pred = svm_classifier.predict(x_test_scaled)
                test_accuracy = accuracy_score(y_test_sub, y_pred)

                elapsed_time = time.time() - start_time
                print(
                    f"Kernel={kernel}, C={C}, degree={degree}, Train Accuracy={train_accuracy:.4f}, Test Accuracy={test_accuracy:.4f}, Time={elapsed_time:.2f}s")

                # Αποθήκευση των αποτελεσμάτων
                results.append({
                    'Kernel': kernel,
                    'C': C,
                    'Degree/Gamma': degree,
                    'Train Accuracy': train_accuracy,
                    'Test Accuracy': test_accuracy,
                    'Time (s)': elapsed_time
                })
        elif kernel == 'rbf':  # Δοκιμάζει διαφορετικές τιμές gamma για RBF
            for gamma in gammas:
                print(f"Training SVM with kernel={kernel}, C={C}, gamma={gamma}")
                start_time = time.time()

                # Δημιουργία και εκπαίδευση του μοντέλου
                svm_classifier = SVC(kernel=kernel, C=C, gamma=gamma)
                svm_classifier.fit(x_train_scaled, y_train_sub.ravel())

                # Αξιολόγηση
                train_accuracy = svm_classifier.score(x_train_scaled, y_train_sub.ravel())
                y_pred = svm_classifier.predict(x_test_scaled)
                test_accuracy = accuracy_score(y_test_sub, y_pred)

                elapsed_time = time.time() - start_time
                print(
                    f"Kernel={kernel}, C={C}, gamma={gamma}, Train Accuracy={train_accuracy:.4f}, Test Accuracy={test_accuracy:.4f}, Time={elapsed_time:.2f}s")

                # Αποθήκευση των αποτελεσμάτων
                results.append({
                    'Kernel': kernel,
                    'C': C,
                    'Degree/Gamma': gamma,
                    'Train Accuracy': train_accuracy,
                    'Test Accuracy': test_accuracy,
                    'Time (s)': elapsed_time
                })
        else:  # Για linear kernel δεν έχει επιπλέον παράμετρο
            print(f"Training SVM with kernel={kernel}, C={C}")
            start_time = time.time()

            # Δημιουργία και εκπαίδευση του μοντέλου
            svm_classifier = SVC(kernel=kernel, C=C)
            svm_classifier.fit(x_train_scaled, y_train_sub.ravel())

            # Αξιολόγηση
            train_accuracy = svm_classifier.score(x_train_scaled, y_train_sub.ravel())
            y_pred = svm_classifier.predict(x_test_scaled)
            test_accuracy = accuracy_score(y_test_sub, y_pred)

            elapsed_time = time.time() - start_time
            print(
                f"Kernel={kernel}, C={C}, Train Accuracy={train_accuracy:.4f}, Test Accuracy={test_accuracy:.4f}, Time={elapsed_time:.2f}s")

            # Αποθήκευση των αποτελεσμάτων
            results.append({
                'Kernel': kernel,
                'C': C,
                'Degree/Gamma': 'N/A',
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy,
                'Time (s)': elapsed_time
            })

# Μετατροπή σε DataFrame και εκτύπωση
results_df = pd.DataFrame(results)
print("\nResults Summary:")
print(results_df)
