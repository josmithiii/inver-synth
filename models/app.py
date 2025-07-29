import logging
import os
from pathlib import Path
from pickle import load
from typing import Callable, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from generators.parameters import ParameterSet, ParamValue
from models.common.data_generator import SoundDataGenerator

"""Dotenv Config"""
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)


"""Data Utils"""


def train_val_split(
    x_train: np.ndarray, y_train: np.ndarray, split: float = 0.2,
) -> tuple:

    slice: int = int(x_train.shape[0] * split)

    x_val: np.ndarray = x_train[-slice:]
    y_val: np.ndarray = y_train[-slice:]

    x_train = x_train[:-slice]
    y_train = y_train[:-slice]

    return (x_val, y_val, x_train, y_train)


"""Model Utils"""


def mean_percentile_rank(y_true, y_pred, k=5):
    """
    @paper
    The first evaluation measure is the Mean Percentile Rank
    (MPR) which is computed per synthesizer parameter.
    """
    # TODO


def top_k_mean_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    @ paper
    The top-k mean accuracy is obtained by computing the top-k
    accuracy for each test example and then taking the mean across
    all examples. In the same manner as done in the MPR analysis,
    we compute the top-k mean accuracy per synthesizer
    parameter for 𝑘 = 1, ... ,5.
    """
    # TODO: per parameter?
    batch_size = y_true.size(0)
    
    # Get the indices of the true values (argmax of one-hot encoded)
    y_true_indices = torch.argmax(y_true, dim=-1)
    
    # Get top-k predictions
    _, top_k_indices = torch.topk(y_pred, k, dim=-1)
    
    # Check if true indices are in top-k predictions
    correct = torch.zeros_like(y_true_indices, dtype=torch.bool)
    for i in range(k):
        correct |= (top_k_indices[:, i] == y_true_indices)
    
    return torch.mean(correct.float())


def setup_model_training(model: nn.Module, device: torch.device):
    """
    Setup model for training with optimizer and loss function
    """
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Move model to device
    model = model.to(device)
    
    # Optimizer - Adam optimizer as per paper
    optimizer = optim.Adam(model.parameters())
    
    # Loss function - Binary Cross Entropy as per paper
    criterion = nn.BCELoss()
    
    return model, optimizer, criterion


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    running_top_k_acc = 0.0
    running_mae = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_top_k_acc += top_k_mean_accuracy(targets, outputs).item()
        running_mae += torch.mean(torch.abs(outputs - targets)).item()
    
    return running_loss / num_batches, running_top_k_acc / num_batches, running_mae / num_batches


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Validate model for one epoch"""
    model.eval()
    running_loss = 0.0
    running_top_k_acc = 0.0
    running_mae = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            running_top_k_acc += top_k_mean_accuracy(targets, outputs).item()
            running_mae += torch.mean(torch.abs(outputs - targets)).item()
    
    return running_loss / num_batches, running_top_k_acc / num_batches, running_mae / num_batches


def compare(target, prediction, params, precision=1, print_output=False):
    if print_output and len(prediction) < 10:
        print(prediction)
        print("Pred: {}".format(np.round(prediction, decimals=2)))
        print("PRnd: {}".format(np.round(prediction)))
        print("Act : {}".format(target))
        print("+" * 5)

    pred: List[ParamValue] = params.decode(prediction)
    act: List[ParamValue] = params.decode(target)
    pred_index: List[int] = [np.array(p.encoding).argmax() for p in pred]
    act_index: List[int] = [np.array(p.encoding).argmax() for p in act]
    width = 8
    names = "Parameter: "
    act_s = "Actual:    "
    pred_s = "Predicted: "
    pred_i = "Pred. Indx:"
    act_i = "Act. Index:"
    diff_i = "Index Diff:"
    for p in act:
        names += p.name.rjust(width)[:width]
        act_s += f"{p.value:>8.2f}"
    for p in pred:
        pred_s += f"{p.value:>8.2f}"
    for p in pred_index:
        pred_i += f"{p:>8}"
    for p in act_index:
        act_i += f"{p:>8}"
    for i in range(len(act_index)):
        diff = pred_index[i] - act_index[i]
        diff_i += f"{diff:>8}"
    exact = 0.0
    close = 0.0
    n_params = len(pred_index)
    for i in range(n_params):
        if pred_index[i] == act_index[i]:
            exact = exact + 1.0
        if abs(pred_index[i] - act_index[i]) <= precision:
            close = close + 1.0
    exact_ratio = exact / n_params
    close_ratio = close / n_params
    if print_output:
        print(names)
        print(act_s)
        print(pred_s)
        print(act_i)
        print(pred_i)
        print(diff_i)
        print("-" * 30)
    return exact_ratio, close_ratio


def evaluate(
    prediction: np.ndarray, x: np.ndarray, y: np.ndarray, params: ParameterSet,
):

    print("Prediction Shape: {}".format(prediction.shape))

    num: int = x.shape[0]
    correct: int = 0
    correct_r: float = 0.0
    close_r: float = 0.0
    for i in range(num):
        should_print = i < 5
        exact, close = compare(
            target=y[i],
            prediction=prediction[i],
            params=params,
            print_output=should_print,
        )
        if exact == 1.0:
            correct = correct + 1
        correct_r += exact
        close_r += close
    summary = params.explain()
    print(
        "{} Parameters with {} levels (fixed: {})".format(
            summary["n_variable"], summary["levels"], summary["n_fixed"]
        )
    )
    print(
        "Got {} out of {} ({:.1f}% perfect); Exact params: {:.1f}%, Close params: {:.1f}%".format(
            correct,
            num,
            correct / num * 100,
            correct_r / num * 100,
            close_r / num * 100,
        )
    )


def data_format_audio(audio: np.ndarray, data_format: str) -> np.ndarray:
    # `(None, n_channel, n_freq, n_time)` if `'channels_first'`,
    # `(None, n_freq, n_time, n_channel)` if `'channels_last'`,

    if data_format == "channels_last":
        audio = audio[np.newaxis, :, np.newaxis]
    else:
        audio = audio[np.newaxis, np.newaxis, :]

    return audio


"""
Wrap up the whole training process in a standard function. Gets a callback
to actually make the model, to keep it as flexible as possible.
# Params:
# - dataset_name (dataset name)
# - model_name: (C1..C6,e2e)
# - model_callback: function taking name,inputs,outputs,data_format and returning a PyTorch model
# - epochs: int
# - dataset_dir: place to find input data
# - output_dir: place to put outputs
# - parameters_file (override parameters filename)
# - dataset_file (override dataset filename)
# - data_format (channels_first or channels_last)
# - run_name: to save this run as
# - batch_size: batch size for training
"""


def train_model(
    # Main options
    dataset_name: str,
    model_name: str,
    epochs: int,
    model_callback: Callable[[str, int, int, str], nn.Module],
    dataset_dir: str,
    output_dir: str,  # Directory names
    dataset_file: str = None,
    parameters_file: str = None,
    run_name: str = None,
    data_format: str = "channels_last",
    save_best: bool = True,
    resume: bool = False,
    checkpoint: bool = True,
    model_type: str = "E2E",
    batch_size: int = 64,
):

    if not dataset_file:
        dataset_file = (
            os.getcwd() + "/" + dataset_dir + "/" + dataset_name + "_data.hdf5"
        )
    if not parameters_file:
        parameters_file = (
            os.getcwd() + "/" + dataset_dir + "/" + dataset_name + "_params.pckl"
        )
    if not run_name:
        run_name = dataset_name + "_" + model_name

    model_file = f"{output_dir}/{run_name}.pth"
    best_model_file = f"{output_dir}/{run_name}_best.pth"
    checkpoint_model_file = f"{output_dir}/{run_name}_checkpoint.pth"
    history_file = f"{output_dir}/{run_name}.csv"
    history_graph_file = f"{output_dir}/{run_name}.pdf"

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_gpu_avail = torch.cuda.is_available()

    print("+" * 30)
    print(f"++ {run_name}")
    print(
        f"Running model: {model_name} on dataset {dataset_file} (parameters {parameters_file}) for {epochs} epochs"
    )
    print(f"Saving model in {output_dir} as {model_file}")
    print(f"Saving history as {history_file}")
    print(f"Device: {device}, CUDA available: {cuda_gpu_avail}")
    print("+" * 30)

    os.makedirs(output_dir, exist_ok=True)

    # Get training and validation datasets
    training_dataset = SoundDataGenerator(
        data_file=dataset_file, batch_size=batch_size, shuffle=True, first=0.8
    )
    validation_dataset = SoundDataGenerator(
        data_file=dataset_file, batch_size=batch_size, shuffle=False, last=0.2
    )
    
    # Create data loaders
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    n_samples = training_dataset.get_audio_length()
    print(f"get_audio_length: {n_samples}")
    n_outputs = training_dataset.get_label_size()

    # Initialize model
    model: nn.Module = None
    initial_epoch = 0
    history_data = []
    
    if resume and os.path.exists(checkpoint_model_file):
        history_df = pd.read_csv(history_file)
        initial_epoch = max(history_df.iloc[:, 0]) + 1 if not history_df.empty else 0
        print(f"Resuming from model file: {checkpoint_model_file} after epoch {initial_epoch}")
        
        model = model_callback(
            model_name=model_name,
            inputs=n_samples,
            outputs=n_outputs,
            data_format=data_format,
        )
        model.load_state_dict(torch.load(checkpoint_model_file, map_location=device))
        history_data = history_df.to_dict('records')
    else:
        model = model_callback(
            model_name=model_name,
            inputs=n_samples,
            outputs=n_outputs,
            data_format=data_format,
        )
        # Create empty history file
        pd.DataFrame().to_csv(history_file, index=False)

    # Setup training
    model, optimizer, criterion = setup_model_training(model, device)

    # Training loop
    best_val_loss = float('inf')
    
    try:
        for epoch in range(initial_epoch, epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Shuffle training data
            training_dataset.shuffle()
            
            # Train
            train_loss, train_top_k_acc, train_mae = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            
            # Validate
            val_loss, val_top_k_acc, val_mae = validate_epoch(
                model, val_loader, criterion, device
            )
            
            print(f"Train Loss: {train_loss:.4f}, Train Top-K Acc: {train_top_k_acc:.4f}, Train MAE: {train_mae:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Top-K Acc: {val_top_k_acc:.4f}, Val MAE: {val_mae:.4f}")
            
            # Save history
            epoch_data = {
                'epoch': epoch,
                'loss': train_loss,
                'top_k_mean_accuracy': train_top_k_acc,
                'mean_absolute_error': train_mae,
                'val_loss': val_loss,
                'val_top_k_mean_accuracy': val_top_k_acc,
                'val_mean_absolute_error': val_mae,
            }
            history_data.append(epoch_data)
            
            # Save checkpoint
            if checkpoint:
                torch.save(model.state_dict(), checkpoint_model_file)
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_file)
                print(f"New best model saved with val_loss: {val_loss:.4f}")
            
            # Save history to CSV
            pd.DataFrame(history_data).to_csv(history_file, index=False)
            
    except Exception as e:
        print(f"Something went wrong during training: {e}")

    # Save final model
    torch.save(model.state_dict(), model_file)

    # Save history plot
    if history_data:
        try:
            hist_df = pd.DataFrame(history_data)
            try:
                fig = hist_df.plot(subplots=True, figsize=(8, 25))
                fig[0].get_figure().savefig(history_graph_file)
            except Exception as e:
                print("Couldn't create history graph")
                print(e)
        except Exception as e:
            print("Couldn't save history")
            print(e)

    # Evaluate prediction on random sample from validation set
    with open(parameters_file, "rb") as f:
        parameters: ParameterSet = load(f)

    # Get a sample batch for evaluation
    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            prediction = model(X)
            
            # Convert back to numpy for evaluation
            X_np = X.cpu().numpy()
            y_np = y.cpu().numpy()
            prediction_np = prediction.cpu().numpy()
            
            evaluate(prediction_np, X_np, y_np, parameters)
            break  # Only evaluate first batch
