def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.models import Model, load_model
    from keras.layers import Input, Dense, Dropout, Flatten, Concatenate, BatchNormalization, GlobalAveragePooling2D
    from keras.layers import Conv2D, MaxPooling2D
    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications import ResNet50
    from keras.optimizers import SGD
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    basepath = "C:\Users\luiza\Downloads\lung cancer 100%-20240810T091253Z-001"
    
    # Define input shape
    input_img = Input(shape=(64, 64, 3))

    # --- ResNet50 branch --- (more aggressive fine-tuning)
    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=input_img, pooling=None)
    for layer in resnet.layers:
        layer.trainable = True  # Allow the ResNet50 layers to be trainable
    resnet_out = GlobalAveragePooling2D()(resnet.output)  # More advanced pooling strategy for feature extraction

    # --- Custom CNN branch --- (additional complexity and batch normalization)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # Adjusted convolutional filter size
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)  # Batch normalization to stabilize training
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    # --- Fusion --- (adding more complexity)
    merged = Concatenate()([resnet_out, x])
    dense = Dense(512, activation='relu')(merged)  # Increased dense layer size for more learning capacity
    dropout = Dropout(0.5)(dense)
    output = Dense(3, activation='softmax')(dropout)  # Three classes: no cancer, small cancer, large cancer

    model = Model(inputs=input_img, outputs=output)

    # Compile model
    model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9),  # Adjusted learning rate and added momentum
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Data preparation (advanced augmentation strategies)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=30,  # Added rotation to augment data further
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2]  # Adjust brightness to help highlight subtle details
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory(
        basepath + '/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

    test_set = test_datagen.flow_from_directory(
        basepath + '/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)  # Important for prediction matching

    steps_per_epoch = int(np.ceil(training_set.samples / 32))
    val_steps = int(np.ceil(test_set.samples / 32))

    # Train model
    history = model.fit(
        training_set,
        steps_per_epoch=steps_per_epoch,
        epochs=300,
        validation_data=test_set,
        validation_steps=val_steps
    )

    # Save model
    model.save(basepath + '/lung_model.h5')

    # Evaluate model on the test and training sets
    scores_test = model.evaluate(test_set, verbose=1)
    scores_train = model.evaluate(training_set, verbose=1)
    B = "Testing Accuracy: %.2f%%" % (scores_test[1] * 100)
    C = "Training Accuracy: %.2f%%" % (scores_train[1] * 100)

    # --- Prediction Evaluation ---
    test_set.reset()
    y_pred_probs = model.predict(test_set, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_set.classes

    acc = accuracy_score(y_true, y_pred)
    correct_preds = np.sum(y_pred == y_true)
    total_preds = len(y_true)

    D = f"Correct Predictions: {correct_preds}"
    E = f"Total Predictions: {total_preds}"
    F = f"Overall Accuracy Score: {acc * 100:.2f}%"

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=list(test_set.class_indices.keys())))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    msg = f"{B}\n{C}\n{D}\n{E}\n{F}"
    print(msg)

    # Plot accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(basepath + "/accuracy_advanced_fusion.png", bbox_inches='tight')
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(basepath + "/loss_advanced_fusion.png", bbox_inches='tight')
    plt.show()

    return msg
