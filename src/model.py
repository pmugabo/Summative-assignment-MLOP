from typing import Tuple, Optional, Union, List, Dict, Any
import os

import tensorflow as tf
from tensorflow.keras import (
    layers, 
    models, 
    applications, 
    regularizers, 
    optimizers, 
    callbacks, 
    Model,
    Sequential,
    utils
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau, 
    TensorBoard
)

def create_model(
    input_shape: Tuple[int, int, int] = (128, 17, 1),
    num_classes: int = 10,
    dropout_rate: float = 0.5,
    l2_reg: float = 0.001,
    use_batch_norm: bool = True
) -> tf.keras.Model:
    """Create a CNN model for audio classification using mel-spectrograms."""
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # First conv block
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x) if use_batch_norm else x
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)

    # Convolutional blocks
    filters = 64
    for _ in range(4):
        x = layers.Conv2D(filters, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x) if use_batch_norm else x
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)
        filters = min(512, filters * 2)

    # Global pooling and dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)
    
def create_model_with_transfer_learning(
    input_shape: Tuple[int, int, int] = (128, 87, 3),
    num_classes: int = 10,
    dropout_rate: float = 0.3,
    l2_reg: float = 0.001,
    base_model_trainable: bool = False
) -> tf.keras.Model:
    """Create a model using transfer learning with a pre-trained base model."""
    # Use MobileNetV2 as base model
    base_model = applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze the base model
    base_model.trainable = base_model_trainable
    
    # Create new model on top
    inputs = layers.Input(shape=input_shape)
    x = applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)

def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 0.001,
    loss: str = 'sparse_categorical_crossentropy',
    metrics: Optional[List[str]] = None
) -> tf.keras.Model:
    """Compile the model with appropriate loss and optimizer."""
    if metrics is None:
        metrics = ['accuracy']
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    return model

def get_callbacks(
    checkpoint_path: str,
    patience: int = 10,
    reduce_lr_patience: int = 5
) -> List[tf.keras.callbacks.Callback]:
    """Get callbacks for model training."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    return [
        ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]