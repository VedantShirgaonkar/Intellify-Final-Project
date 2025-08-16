import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- 1. Load the Processed Data ---
print("Loading processed data...")
sequences = np.load('sequences.npy', allow_pickle=True)
labels = np.load('labels.npy')
print("Data loaded successfully.")

# --- 2. Prepare Data for the Model ---
# Determine the maximum sequence length for padding
# Using the 95th percentile is a good practice to handle outliers
sequence_lengths = [len(seq) for seq in sequences]
MAX_LENGTH = int(np.percentile(sequence_lengths, 95))
print(f"Padding sequences to a max length of: {MAX_LENGTH}")

# Pad sequences and one-hot encode labels
X = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post', dtype='float32')
y = to_categorical(labels)

# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data prepared: {len(X_train)} training samples, {len(X_test)} validation samples.")

# --- 3. Build the GRU Model ---
NUM_CLASSES = y.shape[1]
print(f"Building model for {NUM_CLASSES} classes.")

model = Sequential([
    # Input shape: (sequence_length, num_features)
    GRU(64, return_sequences=True, activation='relu', input_shape=(MAX_LENGTH, X.shape[2])),
    Dropout(0.3),
    GRU(128, return_sequences=True, activation='relu'),
    Dropout(0.3),
    GRU(64, return_sequences=False, activation='relu'),
    
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation='softmax') # The final output layer
])

# --- 4. Compile the Model ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 5. Train the Model ---
# Define callbacks to save the best model and stop training early if it stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('wlasl_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')

print("\n--- Starting Model Training ---")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=500, # Train for many epochs; EarlyStopping will find the optimal number
    callbacks=[early_stopping, model_checkpoint]
)
print("--- Model Training Complete ---")
print("The best model has been saved as 'wlasl_model.keras'.")