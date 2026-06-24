import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
from cnnClassifier.config.configuration import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config = PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model =tf.keras.applications.MobileNetV2(
            input_shape = self.config.params_image_size,
            weights = self.config.params_weights,
            include_top = self.config.params_include_top,
            classes = self.config.params_classes,

        ) 

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model:tf.keras.Model):
        model.save(path)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[: -freeze_till ]:
                layer.trainable = False
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./127.5, offset=-1),
            tf.keras.layers.RandomFlip("horizontal"),
              ], name="data_augmentation")
        
        #Build the model structure
        inputs = tf.keras.Input(shape=model.input_shape[1:])
        x = data_augmentation(inputs)
        x = model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128,activation="relu",kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation='softmax'
        )(x)
        
        full_model = tf.keras.models.Model(inputs=inputs,
                                           outputs=prediction)
        
        full_model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        full_model.summary()
        return full_model
    
    def update_base_model(self):
        self.full_model =self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate)
        
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    def fine_tune_model(self, freeze_till=20, fine_tune_lr=1e-5):
        """Phase 2: Unfreeze the top layers and re-compile."""
        # Ensure we are using the existing full_model
        # We re-call the prepare function with freeze_all=False and a specific freeze_till
        self.full_model = self._prepare_full_model(
            model=self.model, 
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=freeze_till,
            learning_rate=fine_tune_lr
        )
        print(f"Model re-compiled for fine-tuning, unfreezing last {freeze_till} layers.")
        return self.full_model   

