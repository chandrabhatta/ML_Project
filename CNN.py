import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix,ConfusionMatrixDisplay
from math import comb, factorial
from scipy.signal import resample
from numpy.polynomial.chebyshev import chebvander
from tensorflow.keras import layers, models

##---- Constants and functions ----##

# Constants:
gestures = ['click', 'pinch', 'wave', 'swipe']

# Functions:

# Chebyshev Feature extraction:

# Functions:

# Chebyshev Feature extraction:
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

# CNN 

def CNN(X_train_input, X_val_input, 
        y_train, y_val, filter1=32, filter2=64, 
        filter_size=3, max_pool=2,dense=64, dropout=0.3):

    # Clear keras's associated internal state for each model
    # Ensure the same random initial selection of parameters:

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    # Add channel dimension 

    if X_train_input.ndim != 2:
        raise ValueError(f"Expected X_train_input to have 2 dimensions; "f"got {X_train_input.shape}")

    n_features = X_train_input.shape[1]
    side = int(np.sqrt(n_features))

    if side * side != n_features:
        raise ValueError(f"Feature count ({n_features}) must be a perfect square ""to reshape into a 2D Chebyshev moment matrix.")

    if X_val_input.ndim != 2 or X_val_input.shape[1] != n_features:
        raise ValueError("Training and validation inputs must have the same "
            "number of features.")

    X_train_input = X_train_input.reshape(-1, side, side, 1)
    X_val_input = X_val_input.reshape(-1, side, side, 1)


    # Encode the output labels as numerical data:

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    # Train the Model:

    model = models.Sequential([
        layers.Input(shape=(side,side, 1)), 
        layers.Conv2D(filter1, (filter_size,filter_size), activation='relu',padding='same'),
        layers.MaxPooling2D((max_pool,max_pool)),
        layers.Conv2D(filter2, (filter_size,filter_size), activation='relu',padding='same'),
        layers.MaxPooling2D((max_pool,max_pool)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(dense, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(4,activation='softmax')])

    # Compile the model:

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    # Display the model summary:

    model.summary()

    # Train the network

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5, restore_best_weights = True)

    history = model.fit(X_train_input, y_train_encoded, validation_data=(X_val_input,y_val_encoded), epochs=30, batch_size=16, callbacks=[early_stopping], verbose=0)

    # Predict on validation data, not test data
    y_val_prob = model.predict(X_val_input, verbose=0)
    y_val_pred = np.argmax(y_val_prob, axis=1)


    return model, history, y_val_encoded, y_val_pred, label_encoder.classes_


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

# Do the train-validation-test split:

X_trainval, X_test, y_trainval, y_test = train_test_split(X,y,test_size=0.15,random_state=42,stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_trainval,y_trainval,test_size=0.1767,random_state=42,stratify=y_trainval)


# Plot the spectogram:

'''
def plot_spectrograms_and_analyze_lengths(X, X_standardized, X_cnn, y, gestures):

    # Plot the unstandardized spectrograms
    for gesture in gestures:
        # Find the first sample belonging to this gesture
        index = y.index(gesture)

        plt.figure(figsize=(7, 5))
        plt.imshow(
            X[index],
            aspect='auto',
            origin='lower',
            cmap='jet'
        )

        plt.title(f'{gesture.capitalize()} gesture - Unstandardized')
        plt.xlabel('Time')
        plt.ylabel('Doppler')
        plt.colorbar(label='Magnitude')
        plt.show()

    # Plot the standardized spectrograms
    for gesture in gestures:
        # Find the first sample belonging to this gesture
        index = y.index(gesture)

        plt.figure(figsize=(7, 5))
        plt.imshow(
            X_standardized[index],
            aspect='auto',
            origin='lower',
            cmap='jet'
        )

        plt.title(f'{gesture.capitalize()} gesture - Standardized')
        plt.xlabel('Time')
        plt.ylabel('Doppler')
        plt.colorbar(label='Standardized value')
        plt.show()

    # Analyze time lengths of the original spectrograms
    time_lengths = [sample.shape[1] for sample in X]

    print("Original spectrogram time lengths:")
    print("Minimum:", min(time_lengths))
    print("Maximum:", max(time_lengths))
    print("Mean:", np.mean(time_lengths))
    print("Median:", np.median(time_lengths))

    # Analyze time lengths of the CNN spectrograms
    time_lengths_cnn = [sample.shape[1] for sample in X_cnn]

    print("\nCNN spectrogram time lengths:")
    print("Minimum:", min(time_lengths_cnn))
    print("Maximum:", max(time_lengths_cnn))
    print("Mean:", np.mean(time_lengths_cnn))
    print("Median:", np.median(time_lengths_cnn))


'''

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


# Model 1: CNN

# Baseline configuration
baseline = {"filter1": 32,"filter2": 64,"filter_size": 3,"max_pool": 2,"dense": 64,"dropout": 0.3}

# Small set of configurations to compare:

configs = [{"name": "Baseline",**baseline},{"name":"Samller CNN", "filter1":16, "filter2": 32,"filter_size": 3,"max_pool": 2,"dense": 32,"dropout": 0.3},
           {"name": "Larger kernel","filter1": 32,"filter2": 64,"filter_size": 5,"max_pool": 2,"dense": 64,"dropout": 0.3},{"name": "More dropout","filter1": 32,
        "filter2": 64,"filter_size": 3,"max_pool": 2,"dense": 64,"dropout": 0.5}]

results = []

for config in configs:
    # Extract the experiment name and CNN hyperparameters
    name = config["name"]
    params = {k: v for k, v in config.items() if k != "name"}

    print(f"\n{'=' * 50}")
    print(f"Experiment: {name}")
    print(f"Parameters: {params}")
    print(f"{'=' * 50}")

    # # Train the CNN
    model, history, y_val_true, y_val_pred, class_names = CNN(X_train_scaled,X_val_scaled,y_train,y_val,**params)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_val_true, y_val_pred)
    precision = precision_score(y_val_true,y_val_pred,average="macro",zero_division=0)
    recall = recall_score(y_val_true,y_val_pred,average="macro",zero_division=0)
    macro_f1 = f1_score(y_val_true,y_val_pred,average="macro",zero_division=0)


    # Print the metrics
    print(f"\nValidation accuracy:  {accuracy:.4f}")
    print(f"Macro precision:      {precision:.4f}")
    print(f"Macro recall:         {recall:.4f}")
    print(f"Macro F1-score:       {macro_f1:.4f}")
    print(f"Epochs trained:       {len(history.history['loss'])}")

    # Detailed classification report
    print("\nClassification report:")
    print(classification_report(y_val_true,y_val_pred,labels=np.arange(len(class_names)),target_names=class_names,zero_division=0))

    # Plot confusion matrix
    cm = confusion_matrix(y_val_true,y_val_pred,labels=np.arange(len(class_names)))

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=class_names,yticklabels=class_names)

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.show()

    # Store results for comparison
    results.append({"configuration": name,**params,"accuracy": accuracy,"macro_precision": precision,
                "macro_recall": recall,"macro_f1": macro_f1,"epochs_trained": len(history.history["loss"])})

    # Release the current model before training the next one
    del model, history




