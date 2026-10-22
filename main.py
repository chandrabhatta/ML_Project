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
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

##---- Constants and functions ----##

# Constants:
gestures = ['click', 'pinch', 'wave', 'swipe']

# Functions:

# Function for standardizing the data:

def standardize_data(X):
    X_standardized =[]

    for sample in X:
        mean = np.mean(sample)
        std = np.std(sample)

        if std !=0:
            sample_standardized= (sample-mean)/std
            X_standardized.append(sample_standardized)
        else:
            print("Skipping a sample with 0 standard deviation")

    return X_standardized

# Function for resizing the image for CNN:

def resize_input(X):

    T_target = 448
    X_padded = []

    for sample in X:
        # Current number of time frames:
            T = sample.shape[1]

            if T > T_target:
                raise ValueError(f"Sample has {T} time frames, exceeding {T_target}")
        
            padding_width = T_target - T
        
            # Pad only along the time axis:
        
            padded_sample = np.pad(sample, pad_width = ((0,0),(0, padding_width)), mode='constant', constant_values =0)
        
            X_padded.append(padded_sample)

    X_cnn = np.array(X_padded)
    
    return X_cnn

# Function for enveloping used for SVM:

def extract_envelopes(sample, target_length):

    n_freq, T = sample.shape
    center = n_freq//2

    positive_env = []
    negative_env = []

    for t in range(T):

        column = sample[:,t]

        # Positive Doppler half

        pos = column[center:]

        neg = column[:center]

        pos_idx = np.argmax(pos)
        neg_idx = np.argmax(neg)

        positive_env.append(pos_idx)
        negative_env.append(neg_idx)

    positive_env = np.array(positive_env)
    negative_env = np.array(negative_env)

    # Normalize duration
    positive_env = resample(positive_env, target_length)
    negative_env = resample(negative_env, target_length)

    return np.concatenate([positive_env, negative_env])

# Function for performing CNN:

def CNN(X_train_input, X_val_input, 
        y_train, y_val, filter1=32, filter2=64, 
        filter_size=3, max_pool=2,dense=64, dropout=0.3):

    # Clear keras's associated internal state for each model
    # Ensure the same random initial selection of parameters:

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    # Add channel dimension 

    if X_train_input.ndim == 3:
        X_train_input = X_train_input[..., np.newaxis]
    if X_val_input.ndim == 3:
        X_val_input = X_val_input[..., np.newaxis]


    # Encode the output labels as numerical data:

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    # Train the Model:

    model = models.Sequential([
        layers.Input(shape=(800, 448, 1)), 
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


# epochs=100, batch_size=16-32
# Function for performing SVM:

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

# Standardization of the training, test and validation data:

X_train_standard = standardize_data(X_train)
X_val_standard = standardize_data(X_val)
X_test_standard = standardize_data(X_test)

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

# Pad the samples to have the same size (for CNN):

X_train_input = resize_input(X_train_standard)
X_val_input = resize_input(X_val_standard)
X_test_input = resize_input(X_test_standard)

# Convert to a regular Numpy array:

print("Padded dataset shape:", X_train_input.shape)
print("First sample shape:", X_train_input[0].shape)

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
    model, history, y_val_true, y_val_pred, class_names = CNN(X_train_input,X_val_input,y_train,y_val,**params)

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




