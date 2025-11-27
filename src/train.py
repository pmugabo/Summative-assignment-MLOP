import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from .model import create_model, compile_model, create_model_with_transfer_learning
from .data_loader import UrbanSound8KLoader
from sklearn.preprocessing import LabelEncoder
import joblib
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_data_augmentation():
    """Create data augmentation pipeline"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomBrightness(0.2),
    ], name="data_augmentation")

def get_callbacks(model_name, patience=10, monitor='val_accuracy'):
    """Create training callbacks"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join('models', f'best_{model_name}.h5'),
            save_best_only=True,
            monitor=monitor,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )
    ]

def preprocess_data(X, y, num_classes=10):
    """Preprocess input data and labels"""
    # Ensure proper shape
    X = np.array(X)
    if X.ndim == 3:
        X = np.expand_dims(X, -1)
    
    # Convert labels to one-hot if needed
    if len(y.shape) == 1 or y.shape[1] != num_classes:
        y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    
    return X, y

def train_model(
    model_type='cnn',
    batch_size=32,
    epochs=50,
    use_augmentation=True,
    learning_rate=0.001,
    dropout_rate=0.3,
    l2_reg=0.001,
    use_transfer_learning=False
):
    """Train the audio classification model."""
    # Setup paths and logging
    data_path = os.path.join('data', 'raw', 'UrbanSound8K')
    model_name = 'transfer_model' if use_transfer_learning else 'cnn_model'
    
    # Data loading
    data_loader = UrbanSound8KLoader(data_path=data_path)
    train_folds = [1, 2, 3]
    val_folds = [4]
    test_folds = [10]

    logger.info(f"Training on folds: {train_folds}, Validating on: {val_folds}, Testing on: {test_folds}")

    # Load and preprocess data
    logger.info("Loading training data...")
    X_train, y_train = data_loader.load_dataset(folds=train_folds)
    X_val, y_val = data_loader.load_dataset(folds=val_folds)
    X_test, y_test = data_loader.load_dataset(folds=test_folds)

    # Preprocess data
    X_train, y_train = preprocess_data(X_train, y_train)
    X_val, y_val = preprocess_data(X_val, y_val)
    X_test, y_test = preprocess_data(X_test, y_test)

    logger.info(f"Train shapes - X: {X_train.shape}, y: {y_train.shape}")
    logger.info(f"Validation shapes - X: {X_val.shape}, y: {y_val.shape}")
    logger.info(f"Test shapes - X: {X_test.shape if len(X_test) > 0 else 'None'}")

    # Model creation
    logger.info("Creating model...")
    input_shape = X_train[0].shape
    
    if use_transfer_learning:
        model = create_model_with_transfer_learning(
            input_shape=input_shape,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg
        )
    else:
        model = create_model(
            input_shape=input_shape,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg
        )

    # Compile model
    model = compile_model(
        model,
        learning_rate=learning_rate,
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # Callbacks
    callbacks = get_callbacks(model_name)

    # Training
    if use_augmentation:
        logger.info("Using data augmentation")
        data_augmentation = setup_data_augmentation()
        
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=lambda x: data_augmentation(x, training=True)
        )
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True
        )

        history = model.fit(
            train_generator,
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    else:
        logger.info("Training without data augmentation")
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

    # Evaluation
    if len(X_test) > 0:
        logger.info("Evaluating on test set...")
        test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

    # Save final model
    final_model_path = f'models/final_{model_name}.h5'
    model.save(final_model_path)
    joblib.dump(history.history, f'models/training_history_{model_name}.pkl')
    
    logger.info(f"Model saved to {final_model_path}")
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train audio classification model')
    parser.add_argument('--model', type=str, default='cnn', 
                      choices=['cnn', 'transfer'],
                      help='Model type to train (cnn or transfer)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--no_augmentation', action='store_false', dest='use_augmentation',
                      help='Disable data augmentation')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate for optimizer')
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout rate')
    parser.add_argument('--l2_reg', type=float, default=0.001,
                      help='L2 regularization factor')
    parser.add_argument('--patience', type=int, default=10,
                      help='Patience for early stopping')

    args = parser.parse_args()

    train_model(
        model_type=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_augmentation=args.use_augmentation,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout,
        l2_reg=args.l2_reg
    )