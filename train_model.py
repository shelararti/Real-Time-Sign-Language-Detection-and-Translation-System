import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# ===============================
# 1️⃣ Load the dataset
# ===============================
data_df = pd.read_csv('sign_data.csv')

# ===============================
# 2️⃣ Separate features and labels
# ===============================
X = data_df.drop(columns=['Sign']).values  # Features (Distance_0 ... Distance_209)
y = data_df['Sign'].values  # Labels (A, B, C, ...)

# ===============================
# 3️⃣ Encode labels to integers
# ===============================
label_mapping = {label: index for index, label in enumerate(np.unique(y))}
y_encoded = np.array([label_mapping[label] for label in y])

# ===============================
# 4️⃣ Normalize the feature data
# ===============================
X = X / np.max(X)  # simple normalization to range [0,1]

# ===============================
# 5️⃣ Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ===============================
# 6️⃣ Define the neural network model
# ===============================
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),       # Input layer
    layers.Dense(256, activation='relu'),          # Hidden layer 1
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),          # Hidden layer 2
    layers.Dropout(0.3),
    layers.Dense(len(label_mapping), activation='softmax')  # Output layer
])

# ===============================
# 7️⃣ Compile the model
# ===============================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ===============================
# 8️⃣ Train the model
# ===============================
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ===============================
# 9️⃣ Evaluate the model
# ===============================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'\n✅ Test Accuracy: {accuracy * 100:.2f}%')

# ===============================
# 🔟 Confusion matrix
# ===============================
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_mapping.keys(),
            yticklabels=label_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_mapping.keys()))

# ===============================
# 11️⃣ Save the model and label map
# ===============================
model.save('model.keras')
np.save('label_mapping.npy', label_mapping)
print("\n💾 Model saved as 'model.keras' and label mapping saved as 'label_mapping.npy'")
