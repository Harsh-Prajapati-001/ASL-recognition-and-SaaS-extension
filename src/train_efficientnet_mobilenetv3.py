import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0, MobileNetV3Small
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATADIR = "asl_alphabet_train/asl_alphabet_train"
CATEGORIES = sorted(os.listdir(DATADIR)) if os.path.exists(DATADIR) else [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "space", "del", "nothing",
]
IMG_SIZE = 64
LIMIT = 500
BATCH_SIZE = 32


def load_data():
    """Load ASL dataset"""
    print("Loading data...")
    data = []
    labels = []

    for idx, category in enumerate(CATEGORIES):
        folder_path = os.path.join(DATADIR, category)
        if not os.path.exists(folder_path):
            print(f"⚠️  Category folder not found: {folder_path}")
            continue

        image_files = os.listdir(folder_path)[:LIMIT]

        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(idx)

    print(f"Loaded {len(data)} images for {len(set(labels))} categories")

    X = np.array(data) / 255.0
    y = to_categorical(labels, num_classes=len(CATEGORIES))

    return X, y, CATEGORIES


def build_efficientnet_model(num_classes):
    """Build EfficientNetB0 model (faster than full EfficientNet)"""
    print("Building EfficientNetB0 model...")

    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def build_mobilenetv3_model(num_classes):
    """Build MobileNetV3Small model (fastest)"""
    print("Building MobileNetV3Small model...")

    base_model = MobileNetV3Small(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def train_model(model, model_name, X_train, X_val, y_train, y_val):
    """Train model with data augmentation"""
    print(f"Training {model_name}...")

    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=0.00001),
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=15,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


def evaluate_model(model, X_val, y_val):
    """Evaluate model on validation set"""
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    X, y, categories = load_data()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    models_to_train = [
        ("EfficientNetB0", build_efficientnet_model),
        ("MobileNetV3Small", build_mobilenetv3_model),
    ]

    best_model = None
    best_accuracy = 0
    best_name = ""

    for model_name, builder in models_to_train:
        print(f"\n{'=' * 60}")
        print(f"Training {model_name}")
        print(f"{'=' * 60}")

        model = builder(len(categories))
        print(f"\nModel summary:")
        model.summary()

        model, history = train_model(model, model_name, X_train, X_val, y_train, y_val)

        accuracy = evaluate_model(model, X_val, y_val)

        output_path = f"asl_model_{model_name.lower()}.h5"
        model.save(output_path)
        print(f"✅ Model saved: {output_path}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = model_name

    print(f"\n{'=' * 60}")
    print(f"Best Model: {best_name} with accuracy {best_accuracy:.4f}")
    print(f"{'=' * 60}")

    best_model.save("asl_model_best.h5")
    print(f"✅ Best model saved as asl_model_best.h5")
