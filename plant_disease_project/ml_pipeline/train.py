import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os
import matplotlib.pyplot as plt
from dataset import load_dataset

# Parameters
IMG_SIZE = 224
EPOCHS_TL = 5  # Transfer learning epochs (frozen base)
EPOCHS_FT = 15 # Fine-tuning epochs (unfrozen base)
LEARNING_RATE_TL = 1e-3
LEARNING_RATE_FT = 1e-5
MODEL_DIR = "models"
MODEL_NAME = "plant_disease_efficientnetb0.h5"

def build_model(num_classes):
    """
    Builds the EfficientNetB0 model for transfer learning.
    """
    # Load pre-trained EfficientNetB0 without top classification layer
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom head for classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x) # Regularization
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model for initial transfer learning
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_TL)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model, base_model

def plot_history(history, title="Training History"):
    """Plots training and validation accuracy/loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()

def train():
    # Load data
    train_ds, val_ds, test_ds, num_classes, class_names = load_dataset()
    
    # Save class names for Android (labels.txt)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "labels.txt"), "w") as f:
        f.write("\n".join(class_names))
    print(f"Saved {num_classes} class names to {MODEL_DIR}/labels.txt")

    # Build initial model with frozen base
    model, base_model = build_model(num_classes)
    model.summary()
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    print("\n--- Phase 1: Transfer Learning (Training Top Layers Only) ---")
    history_tl = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_TL,
        callbacks=[early_stopping]
    )
    plot_history(history_tl, "Transfer Learning History")
    
    print("\n--- Phase 2: Fine-Tuning (Unfreezing Top Layers of Base Model) ---")
    # Unfreeze top layers of the base model
    base_model.trainable = True
    
    # Unfreeze all layers starting from "block6a_expand_conv" (for example)
    # Alternatively, fine-tune the whole base model, but usually better to leave early layers frozen
    fine_tune_at = 200 # EfficientNetB0 has ~237 layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # Recompile with a lower learning rate
    optimizer_ft = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FT)
    model.compile(optimizer=optimizer_ft, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Compute total epochs
    total_epochs = EPOCHS_TL + EPOCHS_FT
    
    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=history_tl.epoch[-1],
        callbacks=[early_stopping]
    )
    plot_history(history_ft, "Fine Tuning History")
    
    # Save Model
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Evaluate on test set
    print("\nEvaluating on Test Set...")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    train()
