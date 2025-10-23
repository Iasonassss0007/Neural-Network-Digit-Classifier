# main.py
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("TensorFlow version:", tf.__version__)

# 1. Load and Prepare the MNIST Dataset
print("Loading MNIST dataset...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# **Reshape data for the CNN**
# A CNN expects a 4D tensor: (num_images, height, width, color_channels)
# For MNIST, the color channel is 1 because it's grayscale.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(f"Data reshaped for CNN: {x_train.shape}")

# 2. Build the Convolutional Neural Network (CNN) Model
model = tf.keras.models.Sequential([
  # Convolutional Layer 1: Looks for 32 different 3x3 features.
  Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  
  # Pooling Layer 1: Shrinks the image by half, keeping the strongest features.
  MaxPooling2D((2, 2)),
  
  # Convolutional Layer 2: Looks for 64 more features in the output of the first layer.
  Conv2D(64, (3, 3), activation='relu'),
  
  # Pooling Layer 2: Shrinks the image again.
  MaxPooling2D((2, 2)),
  
  # Flatten the 2D feature maps into a 1D line for the final layers.
  Flatten(),
  
  # Dense Layer: A standard layer of 128 neurons.
  Dense(128, activation='relu'),
  
  # Dropout Layer: Randomly turns off 50% of neurons during training to prevent overfitting.
  Dropout(0.5),
  
  # Output Layer: 10 neurons, one for each digit.
  Dense(10)
])

# 3. Compile the Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print("CNN model compiled.")
model.summary() # Print a summary of our new model architecture

# 4. Train the Model
print("Starting CNN model training...")
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
print("Training finished.")

# 5. Evaluate the Model's Performance
print("\nEvaluating model on the test set...")
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print("\n--- TEST RESULTS ---")
print(f"Test Accuracy: {test_acc:.4f}") # Display with 4 decimal places
print("--------------------")

# 6. Save the Trained Model
if test_acc > 0.98: # Set a higher bar for our better model
    model.save('digit_recognizer_model.keras')
    print("\nHigh-accuracy CNN model saved as digit_recognizer_model.keras")
else:
    print("\nModel accuracy is not high enough. Not saving.")