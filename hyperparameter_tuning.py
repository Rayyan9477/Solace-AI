import optuna
import torch
import torch.nn as nn
from model import LSTMNet

def objective(trial):
    hidden_size = trial.suggest_int('hidden_size', 8, 128)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    num_epochs = trial.suggest_int('num_epochs', 100, 1000)

    model = LSTMNet(input_size, hidden_size, output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # Train your model here and return the validation loss
    input_size = 100  # Example input size, replace with actual value
    output_size = 10  # Example output size, replace with actual value
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMNet(input_size, hidden_size, output_size).to(device)
    validation_loss = 0.0  # Replace with actual validation loss calculation
    return validation_loss
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)