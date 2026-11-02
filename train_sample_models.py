"""
Train models on sample data so they can correctly classify demo samples.
This ensures the demo works out-of-the-box without the full ASVspoof dataset.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa

from src.config import Config
from src.data.dataset import extract_mel_spectrogram
from src.data.sample_generator import generate_sample_data, ensure_sample_data
from src.models import get_model, list_models


class SampleDataset(Dataset):
    """Simple dataset for generated sample WAV files."""
    
    def __init__(self, sample_dir, num_samples_per_class=50, train=True, seed=42):
        self.sample_dir = sample_dir
        self.samples = []
        
        # Collect bonafide samples (label 0)
        bonafide_dir = os.path.join(sample_dir, 'bonafide')
        bonafide_files = sorted([f for f in os.listdir(bonafide_dir) if f.endswith('.wav')])
        
        # Collect spoof samples (label 1)
        spoof_dir = os.path.join(sample_dir, 'spoof')
        spoof_files = sorted([f for f in os.listdir(spoof_dir) if f.endswith('.wav')])
        
        # Use fixed seed for reproducible train/val split
        rng = np.random.RandomState(seed)
        
        # Limit to num_samples_per_class
        bonafide_files = bonafide_files[:num_samples_per_class]
        spoof_files = spoof_files[:num_samples_per_class]
        
        # Split into train/val (80/20)
        n_bona = len(bonafide_files)
        n_spoof = len(spoof_files)
        
        bona_indices = rng.permutation(n_bona)
        spoof_indices = rng.permutation(n_spoof)
        
        if train:
            bona_idx = bona_indices[:int(0.8 * n_bona)]
            spoof_idx = spoof_indices[:int(0.8 * n_spoof)]
        else:
            bona_idx = bona_indices[int(0.8 * n_bona):]
            spoof_idx = spoof_indices[int(0.8 * n_spoof):]
        
        for i in bona_idx:
            self.samples.append((os.path.join(bonafide_dir, bonafide_files[i]), 0))
        
        for i in spoof_idx:
            self.samples.append((os.path.join(spoof_dir, spoof_files[i]), 1))
        
        # Shuffle
        rng.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        try:
            waveform, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
            
            # Pad or truncate
            if len(waveform) > Config.MAX_LENGTH:
                waveform = waveform[:Config.MAX_LENGTH]
            else:
                padding = Config.MAX_LENGTH - len(waveform)
                waveform = np.pad(waveform, (0, padding), mode='constant')
            
            # Extract mel spectrogram
            mel_spec = extract_mel_spectrogram(waveform)
            features = torch.tensor(mel_spec).float().unsqueeze(0)
            
            return features, label
        
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(1, Config.N_MELS, 250).float(), label


def train_sample_model(model_name='simple_cnn', num_epochs=100, num_samples=100):
    """
    Train a model on generated sample data.
    
    Args:
        model_name: Name of model architecture
        num_epochs: Number of training epochs
        num_samples: Number of samples per class to generate
    """
    print(f"Training {model_name} on sample data...")
    print(f"Generating {num_samples} samples per class...")
    
    # Generate training samples
    sample_dir = Config.SAMPLE_DATA_DIR
    generate_sample_data(output_dir=sample_dir, num_samples=num_samples)
    
    # Create datasets
    train_dataset = SampleDataset(sample_dir, num_samples_per_class=num_samples, train=True)
    val_dataset = SampleDataset(sample_dir, num_samples_per_class=num_samples, train=False)
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Create model
    model = get_model(model_name, num_classes=Config.NUM_CLASSES)
    device = torch.device(Config.DEVICE)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    # Save best model
    model_path = os.path.join(Config.MODEL_SAVE_PATH, f'best_{model_name}.pth')
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    torch.save(best_model_state, model_path)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {model_path}")
    
    return model_path, best_val_acc


def main():
    """Train all available models on sample data."""
    print("=" * 60)
    print("Training models on sample data for demo purposes")
    print("=" * 60)
    
    Config.ensure_directories()
    
    models_to_train = ['simple_cnn', 'resnet18', 'lightcnn']
    results = {}
    
    for model_name in models_to_train:
        try:
            print(f"\n{'='*60}")
            model_path, acc = train_sample_model(
                model_name=model_name,
                num_epochs=80,
                num_samples=100
            )
            results[model_name] = {'path': model_path, 'accuracy': acc}
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    for name, info in results.items():
        print(f"  {name}: {info['accuracy']:.2%} accuracy -> {info['path']}")
    
    return results


if __name__ == '__main__':
    main()
