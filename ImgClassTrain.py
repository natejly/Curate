import tensorflow as tf
import os
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import preprocess_input
from ImgClassData import ImgClassData


class ImgClassTrainer:
    def __init__(self, dataset_path: str,
                 base_model_name: str = "EfficientNetB0",
                 batch_size: int = 64,
                 initial_learning_rate: float = 1e-3,
                 fine_tune_learning_rate: float = 1e-5):
        # Data source
        self.parser = ImgClassData(dataset_path, debug=True)
        self.filepath = self.parser.filepath
        self.img_dims = self.parser.IMSIZE
        self.train_dir = os.path.join(self.filepath, "train")
        self.val_dir = os.path.join(self.filepath, "val")
        self.test_dir = os.path.join(self.filepath, "test")

        # Config
        self.base_model_name = base_model_name
        self.batch_size = batch_size
        self.initial_learning_rate = initial_learning_rate
        self.fine_tune_learning_rate = fine_tune_learning_rate

        # Derived
        self.IMG_SIZE = self.bucket_dims(self.img_dims)
        self.NUM_CLASSES = self.get_classes_from_train(self.train_dir)
        print("Detected classes:", self.NUM_CLASSES)
        print("Using image size:", self.IMG_SIZE)

        # Placeholders
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.base_model = None
        self.model = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    @staticmethod
    def get_classes_from_train(train_dir: str) -> int:
        return len([d.name for d in os.scandir(train_dir) if d.is_dir()])

    @staticmethod
    def bucket_dims(img_dims):
        #TODO: Fix bucketing 
        
        if img_dims[0] < 128:
            dim = max(32, max(img_dims[0], img_dims[1]))
            return (dim, dim)
        if img_dims[0] < 256:
            return (128, 128)
        elif img_dims[0] < 512:
            return (256, 256)
        else:
            return (512, 512)

    def get_base_model(self):
        ModelClass = getattr(tf.keras.applications, self.base_model_name)
        base_model = ModelClass(
            include_top=False,
            weights="imagenet",
            input_shape=self.IMG_SIZE + (3,)
        )
        base_model.trainable = False  # Start with frozen backbone
        return base_model

    def build_model(self, base_model):
        inputs = tf.keras.Input(shape=self.IMG_SIZE + (3,))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.NUM_CLASSES, activation="softmax")(x)
        return models.Model(inputs, outputs)

    @staticmethod
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = preprocess_input(image)  # scale to [-1,1]
        return image, label

    @staticmethod
    def augment(image, label):
        return image, label

    def build_datasets(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_dir, image_size=self.IMG_SIZE, batch_size=self.batch_size, shuffle=True
        ).map(self.preprocess, num_parallel_calls=self.AUTOTUNE).map(self.augment, num_parallel_calls=self.AUTOTUNE).prefetch(self.AUTOTUNE)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.val_dir, image_size=self.IMG_SIZE, batch_size=self.batch_size, shuffle=False
        ).map(self.preprocess, num_parallel_calls=self.AUTOTUNE).prefetch(self.AUTOTUNE)

        self.test_ds = tf.keras.utils.image_dataset_from_directory(
            self.test_dir, image_size=self.IMG_SIZE, batch_size=self.batch_size, shuffle=False
        ).map(self.preprocess, num_parallel_calls=self.AUTOTUNE).prefetch(self.AUTOTUNE)

    def build(self):
        self.base_model = self.get_base_model()
        self.model = self.build_model(self.base_model)
        print(f"Total parameters: {self.model.count_params():,}")
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        print(f"Trainable parameters: {trainable_params:,}")

    def compile_stage1(self):
        self.base_model.trainable = False
        print("\n=== STAGE 1: Training with frozen backbone ===")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.initial_learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train_stage1(self, epochs: int = 10):
        callbacks_stage1 = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            )
        ]
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks_stage1,
            verbose=1
        )
        return history

    def evaluate_test(self):
        """Evaluate the trained model on the test set and return metrics."""
        if self.test_ds is None:
            self.build_datasets()
        if self.model is None:
            self.build()
            self.compile_stage1()
        print("\n=== EVALUATE: Test set ===")
        results = self.model.evaluate(self.test_ds, verbose=1)
        metrics = dict(zip(self.model.metrics_names, [float(r) for r in results]))
        print(f"Test results: {metrics}")
        return metrics

    def run(self):
        self.build_datasets()
        self.build()
        self.compile_stage1()
        return self.train_stage1(epochs=5)
    
    def evaluate(self):
        return self.evaluate_test()

if __name__ == "__main__":
    trainer = ImgClassTrainer("/Users/natejly/Desktop/PetImages")
    trainer.run()
    trainer.evaluate()
