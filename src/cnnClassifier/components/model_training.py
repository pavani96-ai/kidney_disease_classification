import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

tf.keras.backend.clear_session()
from pathlib import Path
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger

class Training:
    def __init__(self, config=ConfigurationManager):
        self.config = config

    def get_base_model(self):
        self.model =tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            validation_split=0.20)

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation='bilinear',
            class_mode='categorical'
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory= self.config.training_data,
            subset = "validation",
            shuffle=False,
            **dataflow_kwargs
                            )
        
        if self.config.params_is_augmentation:
            logger.info(">> Applying Data Augmentation...")
            
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            )

        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory =self.config.training_data,
            subset='training',
            shuffle=True,
            **dataflow_kwargs
        )
        print(self.train_generator.class_indices)
        print("Training class counts:", np.bincount(self.train_generator.classes))
        print("Validation class counts:", np.bincount(self.valid_generator.classes))

    @staticmethod
    def save_model(path:Path, model: tf.keras.Model):
            model.save(path)

    
    def train(self, callbacks=list, resume=False):
        tf.keras.backend.clear_session()
        if resume and os.path.exists(self.config.trained_model_path):
            print(f"--- Loading existing model from {self.config.trained_model_path} ---")
            self.model = tf.keras.models.load_model(self.config.trained_model_path)
        else:
            print("--- Starting training from base model ---")
            self.get_base_model()
        self.get_base_model()
        optimizer = tf.keras.optimizers.SGD(learning_rate=float(self.config.params_learning_rate))
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy', # Matches your class_mode='categorical'
            metrics=[
            "accuracy", 
            tf.keras.metrics.F1Score(average='macro', name='f1_score')]
        )
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.trained_model_path, # Path where the best model will be saved
            monitor='val_loss',
            save_best_only=True,                    # Only save if validation loss improves
            verbose=1
        )
        
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=8, # Stop after 3 epochs of no improvement
        restore_best_weights=True
         )
        
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_f1_score', 
            factor=0.2, 
            patience=3,
            verbose=1,
            mode='max'
        )
        
        csv_logger = tf.keras.callbacks.CSVLogger('training_log.csv', append=True)
        my_callbacks = [checkpoint, early_stopping, lr_scheduler, csv_logger]
        
        labels = self.train_generator.classes
        class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
        )
        class_weight_dict = dict(enumerate(class_weights))

        # --- PHASE 1: Train the Head ---
        print("--- Starting Phase 1: Training the Head ---")
        self.model.fit(
             self.train_generator,
             class_weight=class_weight_dict,
             epochs=self.config.params_epochs,
             validation_data=self.valid_generator,
             callbacks=my_callbacks
             )
        
        # --- PHASE 2: Fine-Tuning ---
        print("--- Starting Phase 2: Fine-Tuning ---")
        # 1. Unfreeze the base layers
        # Access the underlying base model (MobileNet) which is the first layer
        # if your model is a Sequential or functional model
        base_model = self.model.layers[1] 
        base_model.trainable = True
        
        # 2. Freeze everything except the top 20 layers
        for layer in base_model._layers[:-20]:
            layer.trainable = False
            
        # 3. Re-compile with a lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Lower LR for fine-tuning
            loss='categorical_crossentropy',
            metrics=[
            "accuracy", 
            tf.keras.metrics.F1Score(average='macro', name='f1_score')]
        )
        fine_tune_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, # Stop if no improvement for 5 epochs
        restore_best_weights=True
         )
        # 4. Continue training (Fine-tuning)
        self.model.fit(
            self.train_generator,
            class_weight=class_weight_dict,
            epochs=self.config.params_epochs, # Remaining epochs
            validation_data=self.valid_generator,
            callbacks=[checkpoint, fine_tune_early_stopping, lr_scheduler, csv_logger]
        )
        
        # 5. Save the final model
        self.save_model(path=self.config.trained_model_path, model=self.model)

        

