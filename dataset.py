import tensorflow as tf
import tensorflow_datasets as tfds
import os

# Set parameters
IMG_SIZE = 224 # EfficientNetB0 Default
BATCH_SIZE = 32

def preprocess_image(image, label):
    """
    Resizes and formats the image for EfficientNetB0.
    Note: EfficientNetB0 in tf.keras.applications expects inputs in range [0, 255].
    It has a preprocessing layer built-in.
    """
    image = tf.image.resize(image, (IMG_SIZE,IMG_SIZE))
    image = tf.cast(image, tf.float32)
    return image, label

def augment_image(image, label):
    """
    Basic data augmentation: random flip and rotation.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label

def load_dataset(data_dir=None):
    """
    Loads the PlantVillage dataset using tfds.
    Returns train, validation, and test datasets along with class info.
    """
    print("Loading PlantVillage dataset...")
    
    # PlantVillage dataset has 'train' split which we'll divide into train/val/test
    # 80% train, 10% validation, 10% test
    splits = ['train[:80%]', 'train[80%:90%]', 'train[90%:]']
    
    df_list, ds_info = tfds.load(
        'plant_village', 
        split=splits,
        with_info=True,
        as_supervised=True, # Returns (image, label) tuples
        data_dir=data_dir
    )
    
    train_ds = df_list[0]
    val_ds = df_list[1]
    test_ds = df_list[2]
    
    num_classes = ds_info.features['label'].num_classes
    class_names = ds_info.features['label'].names
    
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")

    # Prepare datasets for training
    # AUTOTUNE = tf.data.AUTOTUNE
    
    # Train DS
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=1000)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    # Val DS
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    # Test DS
    test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, num_classes, class_names

if __name__ == "__main__":
    train_ds, val_ds, test_ds, num_classes, class_names = load_dataset()
    print("Dataset setup complete.")
