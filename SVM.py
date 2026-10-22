import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import comb, factorial
from scipy.signal import resample
from numpy.polynomial.chebyshev import chebvander
from pathlib import Path
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

##---- Constants and functions ----##

# Constants:
gestures = ['click', 'pinch', 'wave', 'swipe']

# Functions:
def chebyshev_feature_extraction(X, max_order=20):
    """
    Extract discrete Chebyshev moments from spectrograms in X.
    Each sample has 800 frequency bins and a variable number of time frames.
    Returns an array of shape (number_of_samples, (max_order + 1)**2).
    """
    def chebyshev_polynomials(length):
        T = np.zeros((max_order + 1, length), dtype=float)

        for n in range(max_order + 1):
            for x in range(length):
                value = sum(
                    (-1) ** (n - k)
                    * comb(length - 1 - k, n - k)
                    * comb(n + k, n)
                    * comb(x, k)
                    for k in range(n + 1)
                    if length - 1 - k >= n - k
                )
                T[n, x] = value / (length ** n)

        return T

    def squared_norms(length):
        return np.array([
            factorial(length + n)
            / (
                (2 * n + 1)
                * factorial(length - n - 1)
                * factorial(n) ** 2
                * length ** (2 * n)
            )
            for n in range(max_order + 1)
        ])

    features = []

    for sample in X:
        sample = np.asarray(sample, dtype=float)

        if sample.ndim != 2 or sample.shape[0] != 800:
            raise ValueError(
                f"Each sample must have shape (800, time_frames); got {sample.shape}"
            )

        # Convert dB magnitude (20*log10(|STFT|)) back to linear magnitude.
        sample = 10 ** (sample / 20)

        freq_bins, time_frames = sample.shape
        Tf = chebyshev_polynomials(freq_bins)
        Tt = chebyshev_polynomials(time_frames)
        norm_f = squared_norms(freq_bins)
        norm_t = squared_norms(time_frames)

        moments = Tf @ sample @ Tt.T
        moments /= norm_f[:, None] * norm_t[None, :]

        features.append(moments.ravel())

    return np.asarray(features)

##---- Execution Pipeline ----##:

# Folder containing Person A, Person B etc:

DATA_DIR =  Path(__file__).parent

# Store the data and the corresponding information:

X=[]
y=[]
person=[]

for per in sorted(DATA_DIR.glob("Person*")):
    for gesture in gestures:

        gesture_dir = per/gesture

        for file in sorted(gesture_dir.glob("*.csv")):

            #Load spectogram:
            sample = np.loadtxt(file, delimiter=',')

            #print(per.name, gesture, file.name, sample.shape)

            # Store it:
            X.append(sample)

            # Store its gesture label
            y.append(gesture)

            # Store the name of the person who performed it:
            person.append(per.name)

# Check for corruption in the image:

for i, sample in enumerate(X):

    if np.isnan(sample).any():
        print("NaN found in sample", i)

    if np.isinf(sample).any():
        print("Inf found in sample", i)

    if sample.shape[0] != 800:
        print("Unexpected shape:", sample.shape)

# Train test validation split

X_trainval, X_test, y_trainval, y_test = train_test_split(X,y,test_size=0.15,random_state=42,stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_trainval,y_trainval,test_size=0.1767,random_state=42,stratify=y_trainval)

X_train = chebyshev_feature_extraction(X_train, max_order=20)
X_val   = chebyshev_feature_extraction(X_val, max_order=20)
X_test  = chebyshev_feature_extraction(X_test, max_order=20)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)

# 2. Standardise features: 
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 3. Hyperparameter search using the validation set
C_values = [0.1, 1, 10, 100]
gamma_values = ["scale", 0.001, 0.01]

best_val_accuracy = -1
best_params = None

for C in C_values:
    for gamma in gamma_values:
        model = SVC(kernel="rbf", C=C, gamma=gamma)
        model.fit(X_train_scaled, y_train)

        val_pred = model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, val_pred)

        print(f"C={C}, gamma={gamma}, "f"validation accuracy={val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_params = {"C": C, "gamma": gamma}

print("\nBest parameters:", best_params)
print(f"Best validation accuracy: {best_val_accuracy:.4f}")

# 4. Retrain using training + validation data
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

final_scaler = StandardScaler()
X_trainval_scaled = final_scaler.fit_transform(X_trainval)
X_test_final = final_scaler.transform(X_test)

final_model = SVC(kernel="rbf",C=best_params["C"],gamma=best_params["gamma"])
final_model.fit(X_trainval_scaled, y_trainval)

# 5. Final test evaluation (perform this after tuning is complete)
y_test_pred = final_model.predict(X_test_final)

print("\nFinal test accuracy:",f"{accuracy_score(y_test, y_test_pred):.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_test_pred, zero_division=0))

# 6. Plot the test confusion matrix
cm = confusion_matrix(y_test, y_test_pred, labels=gestures)

plt.figure(figsize=(7, 5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=gestures,yticklabels=gestures)
plt.xlabel("Predicted gesture")
plt.ylabel("True gesture")
plt.title("SVM RBF — Test Confusion Matrix")
plt.tight_layout()
plt.show()




