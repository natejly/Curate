import tensorflow as tf
import os
from datetime import datetime
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import preprocess_input
from ImgClassData import ImgClassData
from TrainingLog import TrainingLog


class ImgClassTrainer:
    def __init__(self, dataset_path: str,
                 base_model_name: str = "EfficientNetB0",
                 batch_size: int = 64,
                 initial_learning_rate: float = 1e-3,
                 fine_tune_learning_rate: float = 1e-5,
                 initial_epochs: int = 10,
                 fine_tune_epochs: int = 10,
                 dual_stage: bool = True,
                 custom_img_size: tuple = None,
                 unfreeze_percent: float = 0.5):
        # Data source
        self.parser = ImgClassData(dataset_path, debug=False)
        self.file_tree = self.parser.json_tree
        self.filepath = self.parser.filepath
        self.img_dims = self.parser.IMSIZE
        self.train_dir = self.parser.train_dir
        self.val_dir = self.parser.val_dir
        self.test_dir = self.parser.test_dir

        # Config
        self.base_model_name = base_model_name
        self.batch_size = batch_size
        self.initial_learning_rate = initial_learning_rate
        self.fine_tune_learning_rate = fine_tune_learning_rate
        self.initial_epochs = initial_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.dual_stage = dual_stage
        self.custom_img_size = custom_img_size
        self.unfreeze_percent = unfreeze_percent

        # Derived
        # Use custom image size if provided, otherwise use bucketed dims
        if custom_img_size:
            # Ensure IMG_SIZE is always a tuple for concatenation operations
            self.IMG_SIZE = tuple(custom_img_size) if isinstance(custom_img_size, (list, tuple)) else custom_img_size
        else:
            self.IMG_SIZE = self.bucket_dims(self.img_dims)
        self.NUM_CLASSES = len(self.parser.classes)
        # print("Detected classes:", self.parser.classes)
        # print("Using image size:", self.IMG_SIZE)

        # Placeholders
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.base_model = None
        self.model = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.metrics = None
        self.history_stage1 = None
        self.history_stage2 = None
        
        # Training log integration
        self.training_log = TrainingLog()
        
        
    def edit_config(self, base_model_name: str, 
                    batch_size: int, 
                    initial_learning_rate: float, 
                    fine_tune_learning_rate: float,
                    initial_epochs: int,
                    fine_tune_epochs: int,
                    dual_stage: bool = None,
                    custom_img_size: tuple = None,
                    unfreeze_percent: float = None):
        
        self.base_model_name = base_model_name
        self.batch_size = batch_size
        self.initial_learning_rate = initial_learning_rate
        self.fine_tune_learning_rate = fine_tune_learning_rate
        self.initial_epochs = initial_epochs
        self.fine_tune_epochs = fine_tune_epochs
        if dual_stage is not None:
            self.dual_stage = dual_stage
        if custom_img_size is not None:
            self.custom_img_size = custom_img_size
            # Ensure IMG_SIZE is always a tuple for concatenation operations
            self.IMG_SIZE = tuple(custom_img_size) if isinstance(custom_img_size, (list, tuple)) else custom_img_size
        if unfreeze_percent is not None:
            self.unfreeze_percent = unfreeze_percent

    @staticmethod
    def get_classes_from_train(train_dir: str) -> list:
        return ([d.name for d in os.scandir(train_dir) if d.is_dir()])

    @staticmethod
    def bucket_dims(img_dims):
        #TODO: Fix bucketing 
        
        if img_dims[0] < 128:
            dim = max(64, max(img_dims[0], img_dims[1]))
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
        """Stage 1: Compile model with frozen backbone for initial training."""
        self.base_model.trainable = False
        print("\n=== STAGE 1: Training with FROZEN backbone ===")
        print(f"Frozen base model layers: {len(self.base_model.layers)}")
        
        self.model.compile(
            optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.initial_learning_rate,
            weight_decay=1e-4   # typical value, tune for your task
        ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Print trainable parameters after freezing
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        print(f"Trainable parameters (Stage 1): {trainable_params:,}")

    def train_stage1(self):
        """Stage 1: Train classifier head with frozen EfficientNet backbone."""
        callbacks_stage1 = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(self.initial_epochs*.1),
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=int(self.initial_epochs*.05),
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'model1_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras',
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            )
        ]
        
        print(f"Training Stage 1 for {self.initial_epochs} epochs...")
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.initial_epochs,
            callbacks=callbacks_stage1,
            verbose=1
        )
        self.history_stage1 = history.history
        print("Stage 1 training completed!")
        
        return history

    def compile_stage2(self):
        """Stage 2: Compile model with unfrozen top layers for fine-tuning."""
        # Unfreeze the base model
        self.base_model.trainable = True
        
        # Calculate how many layers to freeze based on unfreeze_percent
        total_layers = len(self.base_model.layers)
        layers_to_unfreeze = int(total_layers * self.unfreeze_percent)
        fine_tune_at = total_layers - layers_to_unfreeze
        
        # Freeze all layers up to fine_tune_at
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        print(f"\n=== STAGE 2: Fine-tuning with UNFROZEN top layers ===")
        print(f"Fine-tuning from layer {fine_tune_at} onwards")
        print(f"Frozen layers: {fine_tune_at}")
        print(f"Unfrozen layers: {len(self.base_model.layers) - fine_tune_at}")
        
        # Compile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.fine_tune_learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Print trainable parameters after unfreezing
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        print(f"Trainable parameters (Stage 2): {trainable_params:,}")

    def train_stage2(self):
        """Stage 2: Fine-tune the model with unfrozen top layers."""
        callbacks_stage2 = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=2,
                min_lr=1e-8,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'model2_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras',
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            )
        ]
        
        print(f"Training Stage 2 for {self.fine_tune_epochs} epochs...")
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.fine_tune_epochs,
            callbacks=callbacks_stage2,
            verbose=1
        )
        self.history_stage2 = history.history
        print("Stage 2 fine-tuning completed!")
        
        return history

    def evaluate(self):
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
        self.metrics = metrics
    def getHistory1(self):
        return self.history_stage1
    
    def getHistory2(self):
        return self.history_stage2
    
    def getMetrics(self):
        return self.metrics
    
    def getParams(self):
        params = {
            "base_model_name": self.base_model_name,
            "batch_size": self.batch_size,
            "initial_learning_rate": self.initial_learning_rate,
            "initial_epochs": self.initial_epochs,
            "dual_stage": self.dual_stage,
            "img_size": self.IMG_SIZE,
            "custom_img_size": self.custom_img_size,
            "unfreeze_percent": self.unfreeze_percent,
        }
        
        # Only include fine-tuning parameters if dual_stage is True
        if self.dual_stage:
            params.update({
                "fine_tune_learning_rate": self.fine_tune_learning_rate,
                "fine_tune_epochs": self.fine_tune_epochs,
            })
        
        return params
    
    def run_stage1_only(self):
        """Run only Stage 1 training (frozen backbone)."""
        print("Running Stage 1 only...")
        self.build_datasets()
        self.build()
        self.compile_stage1()
        self.train_stage1()
        self.evaluate()
        
        # Log training results
        self.log_training_results()
        
    def run_full_training(self):
        """Run complete two-stage training process."""
        print("Running full two-stage training...")
        
        # Stage 1: Frozen backbone
        self.build_datasets()
        self.build()
        self.compile_stage1()
        self.train_stage1()
        
        # Stage 2: Fine-tuning
        self.compile_stage2()
        self.train_stage2()
        
        # Final evaluation
        self.evaluate()
        
        # Log training results
        self.log_training_results()
        
        print("\n=== TRAINING COMPLETE ===")
        print("Stage 1 (frozen) + Stage 2 (fine-tuning) completed successfully!")
    
    def run(self):
        """Default run method - executes training based on dual_stage flag."""
        if self.dual_stage:
            self.run_full_training()
        else:
            self.run_stage1_only()
    
    def set_single_stage_training(self):
        """Configure trainer for single-stage training only."""
        self.dual_stage = False
        print("Trainer configured for single-stage training (frozen backbone only)")
    
    def set_dual_stage_training(self):
        """Configure trainer for dual-stage training (frozen + fine-tuning)."""
        self.dual_stage = True
        print("Trainer configured for dual-stage training (frozen + fine-tuning)")
    
    def get_training_mode(self):
        """Get current training mode."""
        return "Dual-stage (frozen + fine-tuning)" if self.dual_stage else "Single-stage (frozen only)"
    
    def log_training_results(self):
        """Log training results based on dual_stage flag."""
        if self.dual_stage and self.history_stage1 and self.history_stage2:
            # Use addTwoStageEntry for dual-stage training
            self.training_log.addTwoStageEntry(
                params=self.getParams(),
                stage1_logs=self.history_stage1,
                stage2_logs=self.history_stage2,
                test_metrics=self.metrics
            )
            print("Two-stage training results logged.")
        elif self.history_stage1:
            # Use addEntry for single-stage training
            self.training_log.addEntry(
                params=self.getParams(),
                logs=self.history_stage1,
                test=self.metrics
            )
            print("Single-stage training results logged.")
        else:
            print("No training history to log.")
    
    def save_training_log(self, filename=None):
        """Save training log to file."""
        if filename is None:
            mode = "dual" if self.dual_stage else "single"
            filename = f"training_log_{self.base_model_name}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            f.write(self.training_log.json(pretty=True))
        print(f"Training log saved to {filename}")
        return filename
    



