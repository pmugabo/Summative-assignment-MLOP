import os
import json
import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from .model import create_model, compile_model
from .data_loader import UrbanSound8KLoader
from .train import train_model

def run_hparam_tuning():
    # Define hyperparameters to tune
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([128, 256, 512]))
    HP_DROPOUT = hp.HParam('dropout_rate', hp.RealInterval(0.1, 0.5))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.0001, 0.01))
    
    METRIC_ACCURACY = 'accuracy'

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_LEARNING_RATE],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )

    def train_test_model(hparams):
        model = create_model(
            num_units=hparams[HP_NUM_UNITS],
            dropout_rate=hparams[HP_DROPOUT]
        )
        model = compile_model(
            model,
            learning_rate=hparams[HP_LEARNING_RATE]
        )
        
        # Train the model
        history = train_model(model, use_augmentation=True)
        
        # Get the validation accuracy
        _, val_acc = model.evaluate(X_val, y_val)
        return val_acc

    session_num = 0
    for num_units in HP_NUM_UNITS.domain.values:
        for dropout_rate in np.linspace(HP_DROPOUT.domain.min_value,
                                      HP_DROPOUT.domain.max_value, 3):
            for learning_rate in np.linspace(HP_LEARNING_RATE.domain.min_value,
                                           HP_LEARNING_RATE.domain.max_value, 3):
                hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_DROPOUT: float(dropout_rate),
                    HP_LEARNING_RATE: float(learning_rate)
                }
                
                run_name = f"run-{session_num}"
                print(f"--- Starting trial: {run_name}")
                print({h.name: hparams[h] for h in hparams})
                
                # Log the hyperparameters
                with tf.summary.create_writer(f"logs/hparam_tuning/{run_name}").as_default():
                    hp.hparams(hparams)
                    accuracy = train_test_model(hparams)
                    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
                
                session_num += 1