import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image

# Path configuration
cur_path = os.getcwd()
dataset_path = os.path.join(cur_path, 'dataset/Train')

# Classes and data collection
classes = 4
data = []
labels = []
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Load dataset images
for i in range(classes):
    path = os.path.join(dataset_path, str(i))
    if not os.path.exists(path):
        print(f"Path {path} does not exist")
        continue

    images = os.listdir(path)
    for image_name in images:
        try:
            image = Image.open(os.path.join(path, image_name))
            image = image.resize((30, 30))  # Resize image to 30x30
            image = image.convert('L')  # Convert to grayscale (L mode)
            image = np.array(image)
            data.append(image)
            labels.append(i)
            print(f"{image_name} Loaded")
        except Exception as e:
            print(f"Error loading image {image_name}: {e}")

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=4)

# One-hot encoding for labels
y_train = to_categorical(y_train, 4)
y_test = to_categorical(y_test, 4)

# Build the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(30, 30, 1)))  # 1 channel for grayscale
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(4, activation='softmax'))  # 4 output classes

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))

# Save the trained model
model.save("my_model_new.h5")
print("Model saved as my_model_new.h5")

# Plot training history (Accuracy)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.close()
