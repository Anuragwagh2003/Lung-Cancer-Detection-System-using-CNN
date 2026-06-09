import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

# Define the path where your trained model is saved and the test data
basepath = r"C:\Users\luiza\Downloads\lung cancer 100%-20240810T091253Z-001\lung cancer 100_"
model_path = basepath + '/lung_model.h5'

# Load the pre-trained model
model = load_model(model_path)
print("Model loaded successfully.")

# Set up ImageDataGenerator for preprocessing test images
test_datagen = ImageDataGenerator(rescale=1./255)

# Create a test data generator to load the test images
test_set = test_datagen.flow_from_directory(
    basepath + '/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)  # Ensure predictions are in the same order as the test images

# Get true labels (ground truth)
y_true = test_set.classes

# Make predictions using the trained model
y_pred_probs = model.predict(test_set, verbose=1)

# Convert probabilities to class labels
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Print the accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")

# Optionally, print classification report and confusion matrix for detailed analysis
from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(test_set.class_indices.keys())))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

