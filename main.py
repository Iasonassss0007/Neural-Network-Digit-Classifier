import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("TensorFlow version:", tf.__version__)

print("Loading MNIST dataset...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(f"Data reshaped for CNN: {x_train.shape}")

model = tf.keras.models.Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  MaxPooling2D((2, 2)),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),
  Flatten(),
  Dense(128, activation='relu'),
  Dropout(0.5),
  Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print("CNN model compiled.")
model.summary()

print("Starting CNN model training...")
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
print("Training finished.")

print("\nEvaluating model on the test set...")
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print("\n--- TEST RESULTS ---")
print(f"Test Accuracy: {test_acc:.4f}")
print("--------------------")

if test_acc > 0.98:
    model.save('digit_recognizer_model.keras')
    print("\nHigh-accuracy CNN model saved as digit_recognizer_model.keras")
else:
    print("\nModel accuracy is not high enough. Not saving.")