import os
import sys
from src.model import create_model, compile_model, get_callbacks
from src.train import train_model

from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
class_weights = dict(enumerate(class_weights))

# Use in model.fit
history = model.fit(
    train_generator,
    validation_data=val_generator,
    class_weight=class_weights,  # Add this line
    epochs=epochs,
    callbacks=callbacks
)

if __name__ == "__main__":
    import argparse
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train audio classification model')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate for the optimizer')
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout rate for regularization')
    parser.add_argument('--l2_reg', type=float, default=0.001,
                      help='L2 regularization factor')
    parser.add_argument('--no_augmentation', action='store_true',
                      help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Train the model
    train_model(
        model_type='cnn',
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout,
        l2_reg=args.l2_reg,
        use_augmentation=not args.no_augmentation
    )