#
#-------------------------------------------------------------------------------
##            Department of Computer Science 
###      Master's Degree in Artificial Intelligence
###   Preoject: Classification of wheel images for Industrial process Control



#   Supervisor: Prof. Vittorio Murino /Prof. Cigdem Beyan
#   Author:  Mahdi Soltani 
#   Matricola: VR539135    
#   September 2026
#-------------------------------------------------------------------------------


import os
import sys
import random
import copy
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import numpy as np
import math
import pandas as pd
import torchvision.transforms as T
import torch
import torchvision.datasets as datasets # Import datasets module
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split


Dataset_dir = r'E:\PROJECTS_IA\Laurea Magistrale___UNIVR\DeepLearning\Project_DL\Dataset\Display_Wheel_Photo_Total'

# Select exactly one evaluation strategy.
SPLIT_METHOD = 'HOLDOUT'  # RESUBSTITUTION, HOLDOUT, HOLDOUT_AVERAGE, K_FOLD, LEAVE_ONE_OUT
TEST_SIZE = 0.20          # Used only by HOLDOUT; change to any value in (0, 1)
N_SPLITS = 5              # Used only by K_FOLD
N_REPETITIONS = 5         # Used only by HOLDOUT_AVERAGE
RANDOM_STATE = 42

# Select regularization before running the experiment.
REGULARIZATION_METHOD = 'ALL'  # NONE, EARLY_STOPPING, L2, DROPOUT, ALL
L2_LAMBDA = 1e-4
DROPOUT_RATE = 0.5

class_names = [d for d in os.listdir(Dataset_dir)
               if os.path.isdir(os.path.join(Dataset_dir, d))]
num_classes = len(class_names)#### Fix the number of classes dynamically based on the dataset


class TeeOutput:
    """Write script output to both the console and a text file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()
class DigitMLP(nn.Module):
    """Fully-connected baseline classifier operating on flattened 3x64x64 images."""
    def __init__(self, num_classes=num_classes, hidden_size=256, dropout_rate=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 64 * 64, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))
class DigitCNN(nn.Module):
    """CNN classifier with the same encoder and latent size as the AE/VAE."""
    def __init__(self, num_classes=num_classes, latent_dim=128, dropout_rate=0.0):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_latent = nn.Linear(64 * 8 * 8, latent_dim)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        x = self.encoder_conv(x).flatten(1)
        z = self.fc_latent(x)
        return self.classifier(self.dropout(z))
class DigitRNN(nn.Module):
    """Treats each image row as a timestep (seq_len=64, features=3*64) fed to a RNN."""
    def __init__(self, num_classes=num_classes, hidden_size=256, num_layers=1):
        super(DigitRNN, self).__init__()
        self.input_size = 3 * 64
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, 3, 64, 64) -> (batch, seq_len=64 rows, features=3*64)
        batch_size = x.size(0)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, 64, self.input_size)
        _, h_n = self.rnn(x)
        out = h_n[-1] # Last layer's final hidden state
        return self.fc(out)   
class DigitLSTM(nn.Module):
    """Same row-as-timestep sequence framing as DigitRNN, using an LSTM cell."""
    def __init__(self, num_classes=num_classes, hidden_size=256, num_layers=1):
        super(DigitLSTM, self).__init__()
        self.input_size = 3 * 64
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, 64, self.input_size)
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1] # Last layer's final hidden state
        return self.fc(out)       
class ViTPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from the ViT Transformer formulation."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])
class ViTScaledDotProductAttention(nn.Module):
    """Computes softmax(QK^T / sqrt(d_k)) V, the core attention operation."""
    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, value), attention
class ViTMultiHeadAttention(nn.Module):
    """Splits Q/K/V into multiple heads, applies scaled dot-product attention, and merges results."""
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError('d_model must be divisible by num_heads')

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.attention = ViTScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.query(query).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.key(key).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(value).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        x, attention = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.head_dim
        )
        return self.output(self.dropout(x)), attention
class ViTFeedForward(nn.Module):
    """Position-wise two-layer MLP with ReLU, applied identically to every token."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(d_model, d_ff)
        self.output = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.output(self.dropout(F.relu(self.input(x))))
class ViTLayerNorm(nn.Module):
    """Layer normalization with learnable scale/shift, implemented explicitly (no nn.LayerNorm)."""
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.shift = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / (std + self.eps) + self.shift
class ViTSublayerConnection(nn.Module):
    """Pre-norm residual wrapper: x + dropout(sublayer(norm(x)))."""
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = ViTLayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        sublayer_output = sublayer(self.norm(x))
        if isinstance(sublayer_output, tuple):
            sublayer_output = sublayer_output[0]
        return x + self.dropout(sublayer_output)
class ViTEncoderLayer(nn.Module):
    """One Transformer encoder block: self-attention sublayer followed by a feed-forward sublayer."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = ViTMultiHeadAttention(num_heads, d_model, dropout)
        self.feed_forward = ViTFeedForward(d_model, d_ff, dropout)
        self.sublayers = nn.ModuleList([
            ViTSublayerConnection(d_model, dropout),
            ViTSublayerConnection(d_model, dropout),
        ])

    def forward(self, x, mask=None):
        x = self.sublayers[0](
            x, lambda normalized: self.self_attention(
                normalized, normalized, normalized, mask
            )
        )
        return self.sublayers[1](x, self.feed_forward)
class ViTEncoder(nn.Module):
    """Stack of ViTEncoderLayer blocks followed by a final layer norm."""
    def __init__(self, d_model, num_heads, d_ff, depth, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ViTEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(depth)
        ])
        self.norm = ViTLayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
class DigitViT(nn.Module):
    """Vision Transformer using the explicit ViT encoder components."""
    def __init__(self, num_classes=num_classes, img_size=64, patch_size=8,
                 embed_dim=128, depth=4, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError('img_size must be divisible by patch_size')

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = ViTPositionalEncoding(
            embed_dim, dropout, max_len=self.num_patches + 1
        )
        self.encoder = ViTEncoder(
            d_model=embed_dim,
            num_heads=num_heads,
            d_ff=embed_dim * mlp_ratio,
            depth=depth,
            dropout=dropout,
        )
        self.head = nn.Linear(embed_dim, num_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)
        for parameter in self.parameters():
            if parameter.dim() > 1 and parameter is not self.cls_token:
                nn.init.xavier_uniform_(parameter)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.position(x)
        x = self.encoder(x)
        return self.head(x[:, 0])        
class DigitAEClassifier(nn.Module):
    """Autoencoder whose encoder bottleneck feeds a classification head; decoder kept for reconstruction use."""
    def __init__(self, num_classes=num_classes, latent_dim=128):
        super(DigitAEClassifier, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16 -> 8
        )
        self.fc_latent = nn.Linear(64 * 8 * 8, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 64 * 8 * 8)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.ReLU(),           # 8 -> 16
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), nn.ReLU(),           # 16 -> 32
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2), nn.Sigmoid(),         # 32 -> 64
        )

        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        return self.fc_latent(x)

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 64, 8, 8)
        return self.decoder_conv(x)

    def forward(self, x):
        z = self.encode(x)
        return self.classifier(z)    
class DigitVAEClassifier(nn.Module):
    """Variational Autoencoder whose latent mean feeds a classification head; decoder kept for reconstruction use."""
    def __init__(self, num_classes=num_classes, latent_dim=128):
        super(DigitVAEClassifier, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16 -> 8
        )
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 64 * 8 * 8)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.ReLU(),           # 8 -> 16
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), nn.ReLU(),           # 16 -> 32
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2), nn.Sigmoid(),         # 32 -> 64
        )

        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 64, 8, 8)
        return self.decoder_conv(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        # Use reparameterized sample only in training so eval predictions are deterministic (mu)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.classifier(z)  
class DigitGANDiscriminatorClassifier(nn.Module):
    """DCGAN-style discriminator backbone with a classification head."""
    def __init__(self, num_classes=num_classes):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.main(x).view(x.size(0), -1)
        return self.classifier(x)
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device,
                early_stop_patience=None, early_stop_min_delta=1e-4, restore_best=True,
                regularization_method='NONE'):
    """Run the train/eval loop for num_epochs, tracking metrics and optional early stopping."""
    # To store metrics for plotting later
    history = {
        'train_loss': [], 'train_acc': [], 'train_precision': [],
        'train_recall': [], 'train_f1': [], 'train_balanced_acc': [],
        'test_loss': [], 'test_acc': [], 'test_precision': [],
        'test_recall': [], 'test_f1': [], 'test_balanced_acc': [],
    }
    best_train_loss = float('inf')
    best_epoch = 1
    no_improve = 0
    best_state = None

    for epoch in range(num_epochs):
        # Set the model to training mode (Enables Dropout/Batch Norm)
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_labels = []
        train_predictions = []

        for images, labels in train_loader:
            # Move data to the same device as the model (CPU or GPU)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()   # Clear out old gradients
            loss.backward()         # Compute how much each weight contributed to the loss
            optimizer.step()        # Adjust the weights to reduce the loss

            # loss.item() is the average loss for the current batch
            # images.size(0) is the number of images in the current batch
            # This multiplication returns the total loss for the current batch
            running_loss += loss.item() * images.size(0)

            # get value (discarded) and index for the highest score
            _, predicted = torch.max(outputs.data, 1)
            train_labels.extend(labels.detach().cpu().numpy())
            train_predictions.extend(predicted.detach().cpu().numpy())

            # number of images classified so far in this epoch
            total_train += labels.size(0)

            # count the elements classified correctly (Boolean masking)
            correct_train += (predicted == labels).sum().item()

        # Calculate final metrics for the Training Epoch
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train
        train_precision = precision_score(train_labels, train_predictions, average='macro', zero_division=0)
        train_recall = recall_score(train_labels, train_predictions, average='macro', zero_division=0)
        train_f1 = f1_score(train_labels, train_predictions, average='macro', zero_division=0)
        train_balanced_acc = balanced_accuracy_score(train_labels, train_predictions)

        # Evaluation on the test set
        model.eval() # Set the model to evaluation mode (Freezes Dropout/Batch Norm)
        test_running_loss = 0.0
        correct_test = 0
        total_test = 0
        test_labels = []
        test_predictions = []

        # Disable gradient calculation for evaluation (Saves memory and time)
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                # Weighted loss for the test batch
                test_running_loss += loss.item() * images.size(0)

                # Accuracy tracking for the test batch
                _, predicted = torch.max(outputs.data, 1)
                test_labels.extend(labels.detach().cpu().numpy())
                test_predictions.extend(predicted.detach().cpu().numpy())
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        # Calculate final metrics for the Testing Epoch
        test_loss = test_running_loss / len(test_loader.dataset)
        test_accuracy = 100 * correct_test / total_test
        test_precision = precision_score(test_labels, test_predictions, average='macro', zero_division=0)
        test_recall = recall_score(test_labels, test_predictions, average='macro', zero_division=0)
        test_f1 = f1_score(test_labels, test_predictions, average='macro', zero_division=0)
        test_balanced_acc = balanced_accuracy_score(test_labels, test_predictions)

        # Save metrics to history dictionary for later plotting
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_f1'].append(train_f1)
        history['train_balanced_acc'].append(train_balanced_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_accuracy)
        history['test_precision'].append(test_precision)
        history['test_recall'].append(test_recall)
        history['test_f1'].append(test_f1)
        history['test_balanced_acc'].append(test_balanced_acc)

        # Early Termination Regularization: monitor train loss in the same training run.
        if (regularization_method in {'EARLY_STOPPING', 'ALL'}
            and early_stop_patience is not None):
            delta = best_train_loss - train_loss
            improved = delta > early_stop_min_delta
            if improved:
                best_train_loss = train_loss
                best_epoch = epoch + 1
                no_improve = 0
                if restore_best:
                    best_state = copy.deepcopy(model.state_dict())
            else:
                no_improve += 1

            print(f"Resub monitor | epoch: {epoch+1:3d} | train_loss: {train_loss:.6f} | "
                  f"best_loss: {best_train_loss:.6f} | delta: {delta:.6f} | "
                  f"improved: {improved} | no_improve: {no_improve}")

        # Print the progress for the current epoch



            print(f'Epoch [{epoch+1}/{num_epochs}] | '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Train F1: {train_f1:.3f} | '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, '
              f'Test F1: {test_f1:.3f}')

        if (regularization_method in {'EARLY_STOPPING', 'ALL'}
            and early_stop_patience is not None
            and no_improve >= early_stop_patience):
            print(f"Early stop at epoch {epoch+1}. Best epoch: {best_epoch}")
            break

    if (regularization_method in {'EARLY_STOPPING', 'ALL'}
        and early_stop_patience is not None
        and restore_best and best_state is not None):
        model.load_state_dict(best_state)
        print(f"Restored model weights from best early-stopping epoch: {best_epoch}")
    history['best_epoch'] = best_epoch
    history['best_train_loss'] = best_train_loss
    history['best_test_acc_epoch'] = int(np.argmax(history['test_acc'])) + 1
    history['best_test_acc'] = max(history['test_acc'])
    history['best_test_f1_epoch'] = int(np.argmax(history['test_f1'])) + 1
    history['best_test_f1'] = max(history['test_f1'])
    history['best_test_balanced_acc'] = max(history['test_balanced_acc'])

    return history
def plot_loss_vs_epochs(history, title='Loss vs. Epochs'):
    """Plot train and test loss across epochs from the training history."""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], marker='o', label='Train Loss')
    plt.plot(epochs, history['test_loss'], marker='s', label='Test Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_metric_vs_epochs(histories, metric, title, ylabel):
    """Plot one test metric across epochs for every model."""
    plt.figure(figsize=(10, 6))
    for model_name, history in histories.items():
        epochs = range(1, len(history[f'test_{metric}']) + 1)
        plt.plot(epochs, history[f'test_{metric}'], marker='o', label=model_name)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
def print_final_classification_reports(trained_models, test_loader, class_names, device):
    """Print per-class precision, recall and F1 for each final model."""
    for model_name, model in trained_models.items():
        all_labels = []
        all_predictions = []
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images.to(device, non_blocking=True))
                all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.numpy())

        print(f'\n=== Per-class report: {model_name} ===')
        print(classification_report(
            all_labels,
            all_predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        ))       
def plot_confusion_matrix(model, test_loader, class_names, device):
    """Run inference over test_loader and display the resulting confusion matrix heatmap."""
    # 1. Collect all predictions and true labels
    all_labels = []
    all_predictions = []

    model.eval() # Set to evaluation mode (turns off Dropout)

    with torch.no_grad(): # Disable gradient calculation to save memory
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)

            # Get the index of the highest score (the prediction)
            _, predicted = torch.max(outputs.data, 1)

            # Move data back to CPU and convert to numpy for sklearn
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 2. Compute the mathematical confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # 3. Create the Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix: Model Errors at a Glance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
def wheel_class_sort_key(class_name):
    """Sort names like wheel_1, wheel_2, ..., wheel_n by numeric suffix."""
    prefix, sep, suffix = class_name.rpartition('_')
    if sep and suffix.isdigit():
        return (prefix, int(suffix))
    return (class_name, float('inf'))
def plot_sample_images(base_path, images_per_class):
    """Display a grid of random sample images, one row per class found under base_path."""
    # Get the list of subfolders (class names)
    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    classes.sort(key=wheel_class_sort_key) # Ensure wheel_1, wheel_2, ..., wheel_n order

    num_classes = len(classes)

    # Create a grid: Rows = Classes, Cols = images_per_class
    fig, axes = plt.subplots(num_classes, images_per_class, figsize=(images_per_class * 3, num_classes * 3))

    # If there is only one class, wrap axes in a list to keep the loop working
    if num_classes == 1:
        axes = [axes]

    for i, class_name in enumerate(classes):
        class_path = os.path.join(base_path, class_name)
        # Get all image files in this subfolder
        all_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Pick random samples
        sample_images = random.sample(all_images, min(images_per_class, len(all_images)))

        for j, img_name in enumerate(sample_images):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path)

            # Select the correct subplot
            ax = axes[i][j] if num_classes > 1 else axes[j]
            ax.imshow(img)
            ax.set_title(f"{class_name}\n{img_name}", fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    plt.show()
def make_loader(dataset, shuffle):
    """Build a DataLoader with the shared batch size, worker count, and pin_memory settings."""
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=shuffle,
        num_workers=loader_workers,
        pin_memory=pin_memory,
    )
def average_histories(fold_histories):
    """Average epoch metrics across folds, padding shorter early-stopped runs."""
    metric_keys = [
        'train_loss', 'train_acc', 'train_precision', 'train_recall',
        'train_f1', 'train_balanced_acc', 'test_loss', 'test_acc',
        'test_precision', 'test_recall', 'test_f1', 'test_balanced_acc',
    ]
    averaged = {}
    for key in metric_keys:
        max_length = max(len(history[key]) for history in fold_histories)
        values = []
        for history in fold_histories:
            series = list(history[key])
            series.extend([series[-1]] * (max_length - len(series)))
            values.append(series)
        averaged[key] = np.mean(values, axis=0).tolist()

    averaged['best_epoch'] = int(np.argmin(averaged['train_loss'])) + 1
    averaged['best_train_loss'] = min(averaged['train_loss'])
    averaged['best_test_acc_epoch'] = int(np.argmax(averaged['test_acc'])) + 1
    averaged['best_test_acc'] = max(averaged['test_acc'])
    averaged['best_test_f1_epoch'] = int(np.argmax(averaged['test_f1'])) + 1
    averaged['best_test_f1'] = max(averaged['test_f1'])
    averaged['best_test_balanced_acc'] = max(averaged['test_balanced_acc'])
    return averaged


original_stdout = sys.stdout
output_path = os.path.splitext(__file__)[0] + '.out'
output_file = open(output_path, 'w', encoding='utf-8', buffering=1)
sys.stdout = TeeOutput(original_stdout, output_file)
class_names = sorted(class_names, key=wheel_class_sort_key)
plot_sample_images(Dataset_dir, images_per_class=3)
# Count the number of images in each class
class_counts = {}
for class_name in class_names:
    class_path = os.path.join(Dataset_dir, class_name)
    num_images = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
    class_counts[class_name] = num_images
# Convert to a pandas Series for easier plotting
df_counts = pd.Series(class_counts).reindex(class_names)
# Plotting the histogram (bar chart)
plt.figure(figsize=(10, 6))
# sns.barplot(x=df_counts.index, y=df_counts.values, palette='viridis')
sns.barplot(x=df_counts.index, y=df_counts.values, hue=df_counts.index, palette='viridis', legend=False)
plt.title('Distribution of Images Across Classes')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# Define the pipeline
training_transforms = T.Compose([
    T.Resize((64, 64)),                     # Standardize size. Keeping resolution low speeds up training
    T.ToTensor(),                             # Convert to Tensor
    T.Normalize(mean=[0.485, 0.456, 0.406],   # Center the data
                std=[0.229, 0.224, 0.225])
])

test_transforms = T.Compose([
    T.Resize((64, 64)),                     # Standardize size. Keeping resolution low speeds up training
    T.ToTensor(),                             # Convert to Matrix (0.0 - 1.0)
    T.Normalize(mean=[0.485, 0.456, 0.406],   # Center the data
                std=[0.229, 0.224, 0.225])
])


# Create one dataset from Dataset_dir. Splitting is done with indices below so every
# evaluation method uses the same images and class mapping.
full_dataset = datasets.ImageFolder(root=Dataset_dir, transform=training_transforms)
print(f"Number of images in Dataset_dir: {len(full_dataset)}")
print(full_dataset[0][0].shape)

valid_split_methods = {
    'RESUBSTITUTION', 'HOLDOUT', 'HOLDOUT_AVERAGE',
    'K_FOLD', 'LEAVE_ONE_OUT',
}
if SPLIT_METHOD not in valid_split_methods:
    raise ValueError(
        "SPLIT_METHOD must be RESUBSTITUTION, HOLDOUT, HOLDOUT_AVERAGE, "
        "K_FOLD, or LEAVE_ONE_OUT"
    )
if not 0 < TEST_SIZE < 1:
    raise ValueError('TEST_SIZE must be between 0 and 1')
if N_REPETITIONS < 1:
    raise ValueError('N_REPETITIONS must be at least 1')

all_indices = np.arange(len(full_dataset))
all_labels = np.asarray(full_dataset.targets)

if SPLIT_METHOD == 'RESUBSTITUTION':
    split_indices = [(all_indices, all_indices)]
elif SPLIT_METHOD == 'HOLDOUT':
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=all_labels,
    )
    split_indices = [(train_indices, test_indices)]
elif SPLIT_METHOD == 'HOLDOUT_AVERAGE':
    split_indices = []
    for repetition in range(N_REPETITIONS):
        train_indices, test_indices = train_test_split(
            all_indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE + repetition,
            stratify=all_labels,
        )
        split_indices.append((train_indices, test_indices))
elif SPLIT_METHOD == 'K_FOLD':
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    split_indices = list(splitter.split(all_indices, all_labels))
else:
    split_indices = list(LeaveOneOut().split(all_indices))

print(f"Evaluation method: {SPLIT_METHOD}")
print(f"Number of train/test evaluations: {len(split_indices)}")

# Create a loader pair for every evaluation split.
loader_workers = 0 if os.name == 'nt' else 2
pin_memory = torch.cuda.is_available()


split_loaders = []
for train_indices, test_indices in split_indices:
    train_subset = Subset(full_dataset, train_indices.tolist())
    test_subset = Subset(full_dataset, test_indices.tolist())
    split_loaders.append((make_loader(train_subset, True), make_loader(test_subset, False)))

train_loader, test_loader = split_loaders[-1]

# Define Loss Function and Optimizers
criterion = nn.CrossEntropyLoss()
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Install a CUDA-enabled PyTorch build and a compatible NVIDIA driver. "
        f"Current PyTorch version: {torch.__version__}"
    )

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
valid_regularization_methods = {'NONE', 'EARLY_STOPPING', 'L2', 'DROPOUT', 'ALL'}
if REGULARIZATION_METHOD not in valid_regularization_methods:
    raise ValueError(
        "REGULARIZATION_METHOD must be NONE, EARLY_STOPPING, L2, DROPOUT, or ALL"
    )

l2_lambda = L2_LAMBDA if REGULARIZATION_METHOD in {'L2', 'ALL'} else 0.0
dropout_enabled = REGULARIZATION_METHOD in {'DROPOUT', 'ALL'}
early_stopping_enabled = REGULARIZATION_METHOD in {'EARLY_STOPPING', 'ALL'}
# Define max epochs; best epoch will be found inside training with Early termination regularization
num_epochs = 30  # Hard maximum; convergence can stop earlier
print(f"Regularization method: {REGULARIZATION_METHOD}")
print(f"Using Ridge/L2 regularization with weight_decay={l2_lambda}")
print(f"Using dropout regularization: {dropout_enabled}, rate={DROPOUT_RATE}")
early_stop_patience = 3  # Stop after this many epochs without training-loss improvement

# Model configs: each is built, trained (with early termination regularization) and evaluated once
model_configs = {
    'MLP': lambda: DigitMLP(
        num_classes=num_classes,
        dropout_rate=DROPOUT_RATE if dropout_enabled else 0.0,
    ),
    'CNN': lambda: DigitCNN(
        num_classes=num_classes,
        dropout_rate=DROPOUT_RATE if dropout_enabled else 0.0,
    ),
    'RNN': lambda: DigitRNN(num_classes=num_classes),
    'LSTM': lambda: DigitLSTM(num_classes=num_classes),
    'ViT': lambda: DigitViT(
        num_classes=num_classes,
        dropout=DROPOUT_RATE if dropout_enabled else 0.0,
    ),
    'AE': lambda: DigitAEClassifier(num_classes=num_classes),
    'VAE': lambda: DigitVAEClassifier(num_classes=num_classes),
    'DCGAN': lambda: DigitGANDiscriminatorClassifier(num_classes=num_classes),
}




histories = {}
trained_models = {}
evaluation_loaders = {}

for model_name, build_model in model_configs.items():
    print(f"\n=== {model_name} defined with {num_classes} output classes. ===")
    fold_histories = []
    final_model = None
    final_loader_pair = None

    for fold_number, (fold_train_loader, fold_test_loader) in enumerate(split_loaders, 1):
        print(f"--- Evaluation {fold_number}/{len(split_loaders)} ---")
        model = build_model()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_lambda)

        fold_history = train_model(
            model=model,
            train_loader=fold_train_loader,
            test_loader=fold_test_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            device=device,
            early_stop_patience=early_stop_patience if early_stopping_enabled else None,
            early_stop_min_delta=1e-4,
            restore_best=True,
            regularization_method=REGULARIZATION_METHOD,
        )
        fold_histories.append(fold_history)
        final_model = model
        final_loader_pair = (fold_train_loader, fold_test_loader)

    history = average_histories(fold_histories)
    histories[model_name] = history
    trained_models[model_name] = final_model
    evaluation_loaders[model_name] = final_loader_pair

    print(f"Average best test accuracy: {history['best_test_acc']:.2f}%")

# ====================================================================================================================
# Cross-Model Comparison (all 9 architectures side by side) for Wheel Project 
# ====================================================================================================================
comparison_records = []
for model_name, history in histories.items():
    parameter_count = sum(parameter.numel() for parameter in trained_models[model_name].parameters())
    comparison_records.append({
        'Model': model_name,
        'Parameters': parameter_count,
        'Best Test Acc Epoch': history['best_test_acc_epoch'],
        'Best Train Loss Epoch': history['best_epoch'],
        'Epochs Run': len(history['train_loss']),
        'Best Train Loss': history['best_train_loss'],
        'Final Train Acc (%)': history['train_acc'][-1],
        'Final Test Acc (%)': history['test_acc'][-1],
        'Best Test Acc (%)': max(history['test_acc']),
        'Best Test Loss': min(history['test_loss']),
        'Final Test Precision': history['test_precision'][-1],
        'Final Test Recall': history['test_recall'][-1],
        'Final Test F1': history['test_f1'][-1],
        'Best Test Precision': max(history['test_precision']),
        'Best Test Recall': max(history['test_recall']),
        'Best Test F1': history['best_test_f1'],
        'Best Test Balanced Acc': history['best_test_balanced_acc'],
    })

df_comparison = pd.DataFrame(comparison_records).sort_values(
    by='Best Test Acc (%)', ascending=False
).reset_index(drop=True)

print("\n=== Cross-Model Comparison Summary (sorted by Best Test Accuracy) ===")
print(df_comparison)

# 1) Bar chart: best test accuracy per model
plt.figure(figsize=(10, 6))
sns.barplot(data=df_comparison, x='Model', y='Best Test Acc (%)', hue='Model', palette='viridis', legend=False)
plt.title('Model Comparison: Best Test Accuracy')
plt.xlabel('Model')
plt.ylabel('Best Test Accuracy (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 2) Bar chart: best test loss per model
df_loss_comparison = df_comparison.sort_values(
    by='Best Test Loss', ascending=True
).reset_index(drop=True)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_loss_comparison, x='Model', y='Best Test Loss', hue='Model', palette='rocket', legend=False)
plt.title('Model Comparison: Best Test Loss')
plt.xlabel('Model')
plt.ylabel('Best Test Loss')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3) Test precision, recall, F1 and balanced accuracy by epoch
plot_metric_vs_epochs(histories, 'precision', 'Test Macro Precision vs. Epochs', 'Macro precision')
plot_metric_vs_epochs(histories, 'recall', 'Test Macro Recall vs. Epochs', 'Macro recall')
plot_metric_vs_epochs(histories, 'f1', 'Test Macro F1 vs. Epochs', 'Macro F1')
plot_metric_vs_epochs(histories, 'balanced_acc', 'Test Balanced Accuracy vs. Epochs', 'Balanced accuracy')

# 4) Overlaid test accuracy curves across all models
plt.figure(figsize=(10, 6))
for model_name, history in histories.items():
    epochs = range(1, len(history['test_acc']) + 1)
    plt.plot(epochs, history['test_acc'], marker='o', label=model_name)
plt.title('Test Accuracy vs. Epochs (All Models)')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.grid(alpha=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 5) Overlaid test loss curves across all models
plt.figure(figsize=(10, 6))
for model_name, history in histories.items():
    epochs = range(1, len(history['test_loss']) + 1)
    plt.plot(epochs, history['test_loss'], marker='s', label=model_name)
plt.title('Test Loss vs. Epochs (All Models)')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.grid(alpha=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 6) Final train vs test accuracy, grouped by model (highlights overfitting gap)
df_train_vs_test = df_comparison[['Model', 'Final Train Acc (%)', 'Final Test Acc (%)']].melt(
    id_vars='Model', var_name='Split', value_name='Accuracy (%)'
)
plt.figure(figsize=(10, 6))
sns.barplot(data=df_train_vs_test, x='Model', y='Accuracy (%)', hue='Split', palette='Set2')
plt.title('Model Comparison: Final Train vs. Test Accuracy (Overfitting Check)')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 7) Final test metric comparison (all scores normalized to 0-1)
metric_frame = df_comparison[['Model', 'Final Test Acc (%)', 'Final Test Precision',
                              'Final Test Recall', 'Final Test F1']].copy()
metric_frame = metric_frame.rename(columns={'Final Test Acc (%)': 'Final Test Accuracy'})
metric_frame['Final Test Accuracy'] = metric_frame['Final Test Accuracy'] / 100.0
df_metric_comparison = metric_frame.melt(
    id_vars='Model', var_name='Metric', value_name='Score'
)
plt.figure(figsize=(12, 6))
sns.barplot(data=df_metric_comparison, x='Model', y='Score', hue='Metric')
plt.title('Final Test Metrics by Model')
plt.xlabel('Model')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 8) Text ranking, best model first
print("\n=== Ranking by Best Test Accuracy ===")
for rank, row in df_comparison.iterrows():
        print(f"{rank + 1}. {row['Model']} | Best Test Acc: {row['Best Test Acc (%)']:.2f}% | "
            f"Best Test Loss: {row['Best Test Loss']:.4f} | "
            f"Best Test F1: {row['Best Test F1']:.3f} | "
            f"Best Balanced Acc: {row['Best Test Balanced Acc']:.3f} | "
            f"Best Test Acc Epoch: {row['Best Test Acc Epoch']} | "
            f"Best Train Loss Epoch: {row['Best Train Loss Epoch']}")

print("\n=== Ranking by Best Test F1 ===")
print(df_comparison.sort_values('Best Test F1', ascending=False)[
    ['Model', 'Best Test F1', 'Best Test Acc (%)', 'Best Test Balanced Acc', 'Parameters']
].to_string(index=False))

# ====================================================================================================================
# Per-model diagnostics (loss curves + confusion matrices), shown only after all training/comparison is done
# ====================================================================================================================
for model_name in model_configs:
    plot_loss_vs_epochs(histories[model_name], title=f'{model_name}: Loss vs. Epochs')
    plot_confusion_matrix(model=trained_models[model_name], test_loader=test_loader,
                           class_names=full_dataset.classes, device=device)

print_final_classification_reports(trained_models, test_loader, full_dataset.classes, device)

sys.stdout.flush()
output_file.close()
sys.stdout = original_stdout




