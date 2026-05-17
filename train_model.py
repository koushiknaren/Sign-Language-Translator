import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Constants
DATA_PATH = os.path.join('Feature_Extraction')
actions = np.array(['hello', 'thanks', 'iloveyou'])
number_sequences = 30
sequence_length = 30

def load_data():
    """Load the dataset from numpy files."""
    classes = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []

    for action in actions:
        for sequence in range(number_sequences):
            window = []
            for frame_num in range(sequence_length):
                file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                if os.path.exists(file_path):
                    res = np.load(file_path)
                    window.append(res)
                else:
                    print(f"Warning: Missing file {file_path}")
            
            if len(window) == sequence_length:
                sequences.append(window)
                labels.append(classes[action])
                
    return np.array(sequences), np.array(labels)

def train_model():
    """Train the LSTM model."""
    X, labels = load_data()
    
    if len(X) == 0:
        print("Error: No data found. Please run collect_data.py first.")
        return

    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    print("Building model...")
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

    print("Training model...")
    model.fit(X_train, y_train, epochs=100)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1).tolist()
    y_pred_classes = np.argmax(y_pred, axis=1).tolist()

    print("\nConfusion Matrix:")
    print(multilabel_confusion_matrix(y_true, y_pred_classes))
    print("\nAccuracy Score:")
    print(accuracy_score(y_true, y_pred_classes))

    print("Saving model...")
    model.save('lstm_model.h5')
    print("Model saved to lstm_model.h5")

if __name__ == '__main__':
    train_model()
