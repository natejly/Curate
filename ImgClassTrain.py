from ImgClassData import ImgClassData
import tensorflow as tf
import os
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import preprocess_input
parser = ImgClassData("/Users/natejly/Desktop/sorted_digits")
img_dims = parser.IMSIZE
train_dir = os.path.join(parser.filepath, "train")
val_dir = os.path.join(parser.filepath, "val")
test_dir = os.path.join(parser.filepath, "test")
def get_classes_from_train(train_dir):
    return len([d.name for d in os.scandir(train_dir) if d.is_dir()])

def bucket_dims(img_dims):
    if img_dims[0] < 128:
        dim = max(32, max(img_dims[0], img_dims[1]))
        return (dim, dim)
    if img_dims[0] < 256:
        return (128, 128)
    elif img_dims[0] < 512:
        return (256, 256)
    else:
        return (512, 512)

# -----------------------------
# CONFIGURATION
# -----------------------------
IMG_SIZE = bucket_dims(img_dims)
BATCH_SIZE = 64

NUM_CLASSES = get_classes_from_train(train_dir)
print("Detected classes:", NUM_CLASSES)
print("Using image size:", IMG_SIZE)

BASE_MODEL_NAME = "EfficientNetB0"
INITIAL_LEARNING_RATE = 1e-3  # Higher LR for frozen backbone
FINE_TUNE_LEARNING_RATE = 1e-5  # Lower LR for fine-tuning

# -----------------------------
# FUNCTION: Load Base Model
# -----------------------------
def get_base_model(model_name, input_shape, weights="imagenet"):
    """
    Returns a pretrained model (EfficientNet variants) without the top layer.
    """
    ModelClass = getattr(tf.keras.applications, model_name)
    base_model = ModelClass(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape + (3,)
    )
    base_model.trainable = False  # Start with frozen backbone
    return base_model

def build_model(base_model, num_classes):
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)         # backbone
    x = layers.GlobalAveragePooling2D()(x)         # flatten features
    x = layers.Dropout(0.2)(x)                     # regularization
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)

# -----------------------------
# DATA PREPROCESSING
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)   # scale to [-1,1]
    return image, label

# Data augmentation for training (optional but recommended)
def augment(image, label):
    return image, label

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
).map(preprocess, num_parallel_calls=AUTOTUNE).map(augment, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
).map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
).map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# -----------------------------
# BUILD AND COMPILE MODEL
# -----------------------------
base_model = get_base_model(BASE_MODEL_NAME, IMG_SIZE)
model = build_model(base_model, NUM_CLASSES)

print(f"Total parameters: {model.count_params():,}")
print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
# -----------------------------
# STAGE 1: Train with frozen backbone
# -----------------------------
base_model.trainable = False
print("\n=== STAGE 1: Training with frozen backbone ===")

model.compile(
    optimizer=tf.keras.optimizers.Adam(INITIAL_LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_stage1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )
]

history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,   # effectively "infinite"
    callbacks=callbacks_stage1,
    verbose=1
)

# -----------------------------
# STAGE 2: Fine-tune with unfrozen backbone
# -----------------------------
