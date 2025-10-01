# -*- coding: utf-8 -*-
"""
CIFAR-10 multi-model benchmarking (CPU-only) with identical preprocessing,
optional cleaning ENABLED (blur + black), and CodeCarbon CO₂ tracking.

Models:
- simple_cnn
- mobilenet_v2
- resnet50
- efficientnet_b0
- vgg16
- vgg19
- unet

Outputs:
- ./results/results.csv   # accuracy, params, secs, emissions per phase + totals
- ./codecarbon_logs/      # CodeCarbon logs (disabled file save by default)
"""

import os
# ======= FORCE CPU ONLY (set before TF import) =======
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import random
import pathlib
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict

# Optional installs (uncomment if needed)
# !pip install --upgrade pip
# !pip install tensorflow opencv-python git+https://github.com/SarraBoughriou/codecarbon.git

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import cv2
from codecarbon import EmissionsTracker

# extra safety: hide GPUs from TF runtime
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

# ===== reproducibility =====
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ===== paths & codecarbon config =====
OUTDIR = pathlib.Path("./results"); OUTDIR.mkdir(parents=True, exist_ok=True)
LOGDIR = pathlib.Path("./codecarbon_logs"); LOGDIR.mkdir(parents=True, exist_ok=True)

CODECARBON_KWARGS = dict(
    measure_power_secs=1,
    save_to_file=False,       # capture return value instead of writing CSV
    log_level="warning",
    output_dir=str(LOGDIR),
)

# ===== unified preprocessing hyperparams =====
IMG_SIZE = 224               # identical input size for all models
BATCH_SIZE = 256
EPOCHS = 80
PATIENCE_ES = 12
PATIENCE_RLR = 6

# ---------- optional cleaning (ENABLED) ----------
def is_blurry_rgb(image: np.ndarray, threshold: float = 100.0) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def is_black(image: np.ndarray, threshold: float = 10.0) -> bool:
    return float(np.mean(image)) < threshold

def clean_images(images: np.ndarray,
                 labels: np.ndarray,
                 do_blur_check: bool = False,
                 do_black_check: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    mask = np.ones(len(images), dtype=bool)
    stats = {"removed_blurry": 0, "removed_black": 0}
    if do_blur_check:
        for i in range(len(images)):
            if mask[i] and is_blurry_rgb(images[i]):
                mask[i] = False
                stats["removed_blurry"] += 1
    if do_black_check:
        for i in range(len(images)):
            if mask[i] and is_black(images[i]):
                mask[i] = False
                stats["removed_black"] += 1
    return images[mask], labels[mask], stats

# ---------- data loading & unified preprocessing ----------
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    classes = np.unique(y_train)
    nclasses = int(len(classes))
    return (x_train, y_train), (x_test, y_test), nclasses

def build_preprocess_and_aug():
    """Identical preprocessing for ALL models."""
    return keras.Sequential(
        [
            layers.Resizing(IMG_SIZE, IMG_SIZE),
            layers.Rescaling(1./255),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ],
        name="preprocess_aug",
    )

def make_datasets(x_train, y_train, x_test, y_test, num_classes: int):
    preprocess_aug = build_preprocess_and_aug()
    y_train_oh = to_categorical(y_train, num_classes)
    y_test_oh  = to_categorical(y_test,  num_classes)

    def tf_train_gen():
        for img, lbl in zip(x_train, y_train_oh):
            yield img, lbl

    def tf_test_gen():
        for img, lbl in zip(x_test, y_test_oh):
            yield img, lbl

    train_ds = tf.data.Dataset.from_generator(
        tf_train_gen,
        output_signature=(
            tf.TensorSpec(shape=(32, 32, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
        ),
    ).shuffle(10000, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_generator(
        tf_test_gen,
        output_signature=(
            tf.TensorSpec(shape=(32, 32, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
        ),
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds = train_ds.map(lambda x, y: (preprocess_aug(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    test_ds  = test_ds.map(lambda x, y: (preprocess_aug(x, training=False), y),
                           num_parallel_calls=tf.data.AUTOTUNE)
    return train_ds, test_ds

# ---------- model zoo (all consume SAME preprocessed tensors) ----------
def model_simple_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for filters in [32, 64, 64]:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="simple_cnn")

def model_mobilenet_v2(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=10):
    base = keras.applications.MobileNetV2(
        include_top=False, input_shape=input_shape, weights="imagenet", pooling="avg"
    )
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)  # identical scaling already applied
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="mobilenet_v2")

def model_resnet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=10):
    base = keras.applications.ResNet50(
        include_top=False, input_shape=input_shape, weights="imagenet", pooling="avg"
    )
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="resnet50")

def model_efficientnet_b0(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=10):
    base = keras.applications.EfficientNetB0(
        include_top=False, input_shape=input_shape, weights="imagenet", pooling="avg"
    )
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="efficientnet_b0")

def model_vgg16(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=10):
    base = keras.applications.VGG16(
        include_top=False, input_shape=input_shape, weights="imagenet", pooling="avg"
    )
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="vgg16")

def model_vgg19(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=10):
    base = keras.applications.VGG19(
        include_top=False, input_shape=input_shape, weights="imagenet", pooling="avg"
    )
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="vgg19")

# ---- U-Net backbone for classification (encoder-decoder + GAP) ----
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def model_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, 32); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64);     p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128);    p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256);    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck
    bn = conv_block(p4, 512)

    # Decoder
    u6 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(bn)
    u6 = layers.Concatenate()([u6, c4])
    c6 = conv_block(u6, 256)

    u7 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = conv_block(u7, 128)

    u8 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = conv_block(u8, 64)

    u9 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = conv_block(u9, 32)

    # Classification head
    x = layers.GlobalAveragePooling2D()(c9)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="unet")

# register everything to run
EXPERIMENTS = [
    ("simple_cnn",          model_simple_cnn),
   # ("mobilenet_v2",        model_mobilenet_v2),
   # ("resnet50",            model_resnet50),
   # ("efficientnet_b0",     model_efficientnet_b0),
   # ("vgg16",               model_vgg16),
   #("vgg19",               model_vgg19),
   # ("unet",                model_unet),
]

# ---------- reporting helpers ----------
@dataclass
class PhaseReport:
    secs: float = 0.0
    emissions_kg: float = 0.0

@dataclass
class ModelReport:
    model: str
    params: int
    test_acc: float
    preprocess_secs: float
    train_secs: float
    eval_secs: float
    preprocess_emissions_kg: float
    train_emissions_kg: float
    eval_emissions_kg: float

    def as_dict(self) -> Dict:
        return {
            "model": self.model,
            "params": self.params,
            "test_acc": round(self.test_acc, 4),
            "preprocess_secs": round(self.preprocess_secs, 2),
            "train_secs": round(self.train_secs, 2),
            "eval_secs": round(self.eval_secs, 2),
            "preprocess_emissions_kg": self.preprocess_emissions_kg,
            "train_emissions_kg": self.train_emissions_kg,
            "eval_emissions_kg": self.eval_emissions_kg,
            "total_secs": round(self.preprocess_secs + self.train_secs + self.eval_secs, 2),
            "total_emissions_kg": self.preprocess_emissions_kg + self.train_emissions_kg + self.eval_emissions_kg,
        }

def run_phase_with_tracker(phase_name: str, fn: Callable[[], any]) -> Tuple[any, PhaseReport]:
    tracker = EmissionsTracker(project_name=phase_name, **CODECARBON_KWARGS)
    tracker.start()
    t0 = time.time()
    try:
        result = fn()
    finally:
        emissions = tracker.stop() or 0.0
    secs = time.time() - t0
    print(f"[{phase_name}] time={secs:.2f}s, emissions={emissions:.6f} kg")
    return result, PhaseReport(secs=secs, emissions_kg=emissions)

# ---------- main ----------
def main():
    # shared preprocess phase (CLEANING ENABLED)
    def preprocess_fn():
        (x_train, y_train), (x_test, y_test), nclasses = load_cifar10()
        print("Train:", x_train.shape, y_train.shape, " Test:", x_test.shape, y_test.shape)

        # ✅ Enable cleaning here
        DO_BLUR = True
        DO_BLACK = True
        if DO_BLUR or DO_BLACK:
            x_train2, y_train2, stats = clean_images(x_train, y_train, DO_BLUR, DO_BLACK)
            print("Removed from TRAIN:", stats, " ->", x_train2.shape)
            x_train, y_train = x_train2, y_train2

        return x_train, y_train, x_test, y_test, nclasses

    (x_train, y_train, x_test, y_test, nclasses), prep_rep = run_phase_with_tracker("preprocess", preprocess_fn)

    results: List[ModelReport] = []

    for model_name, builder in EXPERIMENTS:
        print("\n==============================")
        print(f"Running model: {model_name} (CPU-only, unified preprocessing, cleaned data)")
        print("==============================")

        train_ds, test_ds = make_datasets(x_train, y_train, x_test, y_test, nclasses)

        model = builder(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=nclasses)
        lr = 1e-3 if model_name in ("simple_cnn", "unet") else 3e-4
        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        params = model.count_params()
        model.summary(line_length=120)

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=PATIENCE_ES,
                                          restore_best_weights=True, mode="max"),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                              patience=PATIENCE_RLR, min_lr=1e-5),
        ]

        def train_fn():
            return model.fit(
                train_ds,
                validation_data=test_ds,
                epochs=EPOCHS,
                verbose=1,
                callbacks=callbacks,
            )

        _, train_rep = run_phase_with_tracker(f"train_{model_name}", train_fn)

        def eval_fn():
            loss, acc = model.evaluate(test_ds, verbose=0)
            print(f"[{model_name}] Test Accuracy: {acc:.4f}")
            return acc

        acc, eval_rep = run_phase_with_tracker(f"eval_{model_name}", eval_fn)

        results.append(
            ModelReport(
                model=model_name,
                params=params,
                test_acc=float(acc),
                preprocess_secs=prep_rep.secs,
                train_secs=train_rep.secs,
                eval_secs=eval_rep.secs,
                preprocess_emissions_kg=prep_rep.emissions_kg,
                train_emissions_kg=train_rep.emissions_kg,
                eval_emissions_kg=eval_rep.emissions_kg,
            )
        )

    import pandas as pd
    df = pd.DataFrame([r.as_dict() for r in results])
    out_csv = OUTDIR / "results.csv"
    df.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv.resolve())
    print(df)

if __name__ == "__main__":
    main()
