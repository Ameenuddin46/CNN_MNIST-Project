#1st step: Load the MNIST Dataset
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Print the shape of the datasets
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

#2nd step
# Visualize the first image in the training dataset
plt.imshow(X_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

#3rd step : Preprocess the Data
# Normalize the images to values between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape the data to match the input shape of the model
X_train = X_train.reshape(-1, 28, 28, 1)  # (batch_size, height, width, channels)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Print the shapes of the processed data
print(f"Processed training data shape: {X_train.shape}")
print(f"Processed test data shape: {X_test.shape}")

#4th step: Build the Neural Network Model
# Build the CNN model
model = tf.keras.models.Sequential([
    # Convolutional layer with 32 filters, 3x3 kernel, ReLU activation
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Max-pooling layer to reduce spatial dimensions
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Second convolutional layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Third convolutional layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    # Flatten the 3D output to 1D for the fully connected layer
    tf.keras.layers.Flatten(),
    
    # Fully connected (dense) layer with 64 units
    tf.keras.layers.Dense(64, activation='relu'),
    
    # Output layer with 10 units (one for each digit)
    tf.keras.layers.Dense(10, activation='softmax')
])

# Print the model summary to see the architecture
model.summary()

#Step 6: Compile and Train the Model
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")


#Step 7: Visualize the Results
import matplotlib.pyplot as plt
import numpy as np

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Predict some test images
predictions = model.predict(X_test[:5])

# Display the first 5 test images and their predicted labels
for i in range(5):
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {np.argmax(y_test[i])}")
    plt.show()

    #Step 8 : save the model
    # Save the trained model to a file
model.save('mnist_model.h5')

print("Model saved as mnist_model.h5")

