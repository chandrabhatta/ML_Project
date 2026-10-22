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

# Function for building CNN:

def build_cnn(input_shape=(200, 224, 1), n_classes=4, base=16, dropout=0.4, lr=1e-3):
    inp = layers.Input(shape=input_shape)
    x = inp
    for i, f in enumerate([base, base*2, base*4, base*4]):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
    # Keep some spatial info: pool to a small grid instead of 1x1
    x = layers.AdaptiveAveragePooling2D if False else layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = models.Model(inp, out)
    m.compile(tf.keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", ["accuracy"])
    return m

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5),
]

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

# Pad the samples to have the same size (for CNN):

X_train_input = resize_input(X_train_standard)
X_val_input = resize_input(X_val_standard)
X_test_input = resize_input(X_test_standard)

# Convert to a regular Numpy array:

print("Padded dataset shape:", X_train_input.shape)
print("First sample shape:", X_train_input[0].shape)


# Encode labels as integers
label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

# Add channel dimension
X_train_input = X_train_input[..., np.newaxis]
X_val_input = X_val_input[..., np.newaxis]
X_test_input = X_test_input[..., np.newaxis]

# Build CNN using the actual input dimensions
model = build_cnn(input_shape=X_train_input.shape[1:],n_classes=len(label_encoder.classes_),base=16,dropout=0.4,lr=1e-3)

model.summary()

# Train using validation data for model selection
history = model.fit(X_train_input,y_train_encoded,validation_data=(X_val_input, y_val_encoded),epochs=60,batch_size=16,callbacks=callbacks,verbose=1)

# Predict on the independent test set
y_test_prob = model.predict(X_test_input, verbose=0)
y_test_pred = np.argmax(y_test_prob, axis=1)

# Calculate final evaluation metrics
test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
test_precision = precision_score(y_test_encoded, y_test_pred, average="macro", zero_division=0)
test_recall = recall_score(y_test_encoded, y_test_pred, average="macro", zero_division=0)
test_f1 = f1_score(y_test_encoded, y_test_pred, average="macro", zero_division=0)

print("\\n========== FINAL TEST RESULTS ==========")
print(f"Test accuracy:       {test_accuracy:.4f}")
print(f"Test macro precision:{test_precision:.4f}")
print(f"Test macro recall:   {test_recall:.4f}")
print(f"Test macro F1-score: {test_f1:.4f}")

# Detailed classification report
print("\\nClassification report:")
print(classification_report(y_test_encoded,y_test_pred,labels=np.arange(len(label_encoder.classes_)),target_names=label_encoder.classes_,zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test_encoded,y_test_pred,labels=np.arange(len(label_encoder.classes_)))

plt.figure(figsize=(7, 5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=label_encoder.classes_,yticklabels=label_encoder.classes_)

plt.title("CNN - Test Set Confusion Matrix")
plt.xlabel("Predicted gesture")
plt.ylabel("True gesture")
plt.tight_layout()
plt.show()
