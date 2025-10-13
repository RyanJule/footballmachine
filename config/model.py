LSTM_CONFIG = {
    'sequence_length': 20,
    'player_features': 670,
    'roster_size': 64,
    'hidden_units': [256, 128, 64],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'validation_split': 0.2
}

HYPERPARAMETER_GRID = {
    'learning_rate': [0.001, 0.01, 0.1],
    'dropout_rate': [0.2, 0.3, 0.4, 0.5],
    'hidden_units': [[128, 64], [256, 128, 64], [512, 256, 128]],
    'batch_size': [16, 32, 64]
}

PREDICTION_TARGETS = {
    'play_type': ['run', 'pass', 'kick', 'punt'],
    'players_involved': ['qb', 'rb', 'wr', 'te', 'k', 'dst'],
    'yardage': 'continuous',
    'scoring': ['no_score', 'touchdown', 'field_goal', 'safety']
}

