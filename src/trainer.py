import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import warnings

from .config import Config
from .models import get_model
from .data import ASVspoofDataset, extract_mel_spectrogram, load_protocol
from .utils import calculate_eer, compute_metrics

warnings.filterwarnings('ignore')


class Trainer:
    """
    Training pipeline for deepfake voice detection models.
    
    Handles model training, validation, checkpointing, and evaluation.
    """
    
    def __init__(self, model_name='resnet18', config=None):
        self.config = config or Config
        self.model_name = model_name
        self.device = torch.device(self.config.DEVICE)
        
        # Initialize model
        self.model = get_model(model_name, num_classes=self.config.NUM_CLASSES)
        self.model = self.model.to(self.device)
        
        # Training history
        self.history = {
            'loss': [],
            'acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_eer': []
        }
        
        # Best model tracking
        self.best_eer = float('inf')
        self.best_acc = 0.0
    
    def setup_data(self, protocol_df, audio_path, batch_size=None, shuffle=True):
        """
        Setup data loader from protocol and audio path.
        
        Args:
            protocol_df: DataFrame with filename and label columns
            audio_path: Path to audio files
            batch_size: Batch size (default from config)
            shuffle: Whether to shuffle data
        
        Returns:
            DataLoader instance
        """
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        
        dataset = ASVspoofDataset(
            protocol_df=protocol_df,
            audio_base_path=audio_path,
            transform=extract_mel_spectrogram,
            max_length=self.config.MAX_LENGTH,
            sample_rate=self.config.SAMPLE_RATE
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return loader
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for features, labels in pbar:
            if features.size(0) == 0:
                continue
            
            # Ensure correct dimensions
            if features.dim() == 3:
                features = features.unsqueeze(1)
            
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc='Validating'):
                if features.size(0) == 0:
                    continue
                
                if features.dim() == 3:
                    features = features.unsqueeze(1)
                
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0
        
        # Calculate EER
        if len(all_labels) > 0:
            eer, _ = calculate_eer(np.array(all_labels), np.array(all_predictions))
        else:
            eer = 1.0
        
        return avg_loss, accuracy, eer
    
    def train(self, train_loader, val_loader, num_epochs=None, 
              learning_rate=None, save_dir=None):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (default from config)
            learning_rate: Learning rate (default from config)
            save_dir: Directory to save model checkpoints
        
        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        if learning_rate is None:
            learning_rate = self.config.LEARNING_RATE
        if save_dir is None:
            save_dir = self.config.MODEL_SAVE_PATH
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        print(f"Training {self.model_name} for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc, val_eer = self.validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Record history
            self.history['loss'].append(train_loss)
            self.history['acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_eer'].append(val_eer)
            
            # Print results
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val EER: {val_eer:.4f}')
            
            # Save best model (based on EER)
            if val_eer < self.best_eer:
                self.best_eer = val_eer
                model_path = os.path.join(save_dir, f'best_{self.model_name}.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f'Saved best model (EER: {val_eer:.4f})')
            
            # Also save best accuracy model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
        
        # Save final model
        final_path = os.path.join(save_dir, f'final_{self.model_name}.pth')
        torch.save(self.model.state_dict(), final_path)
        
        print(f"\nTraining complete!")
        print(f"Best EER: {self.best_eer:.4f}")
        print(f"Best Accuracy: {self.best_acc:.2f}%")
        
        return self.history
    
    def evaluate(self, test_loader):
        """
        Full evaluation on test set.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary with all evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc='Evaluating'):
                if features.size(0) == 0:
                    continue
                
                if features.dim() == 3:
                    features = features.unsqueeze(1)
                
                features = features.to(self.device)
                
                outputs = self.model(features)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                
                all_predictions.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_predictions)
        )
        
        return metrics
    
    def load_model(self, model_path):
        """Load model from checkpoint."""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded model from {model_path}")
            return True
        else:
            print(f"Model file not found: {model_path}")
            return False
    
    def predict(self, features):
        """
        Make prediction on single batch of features.
        
        Args:
            features: Input features tensor
        
        Returns:
            Tuple of (predicted_class, confidence, probabilities)
        """
        self.model.eval()
        
        with torch.no_grad():
            if features.dim() == 3:
                features = features.unsqueeze(1)
            
            features = features.to(self.device)
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
            
            return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]


class Predictor:
    """
    Simplified predictor for inference only.
    
    Used for GUI and single-file inference.
    """
    
    def __init__(self, model_name='resnet18', model_path=None):
        self.device = torch.device(Config.DEVICE)
        self.model_name = model_name
        
        self.model = get_model(model_name, num_classes=Config.NUM_CLASSES)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load model weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
    
    def predict_audio(self, audio_path):
        """
        Predict if an audio file is bonafide or spoof.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Use the same robust loading logic as server.py
            # Import the function from server module
            import numpy as np
            
            # Check if server module has load_audio_robust
            try:
                from src.server import load_audio_robust
                waveform, sr = load_audio_robust(audio_path, Config.SAMPLE_RATE)
            except (ImportError, AttributeError):
                # Fallback: inline the robust loading logic
                import soundfile as sf
                import os
                import librosa
                
                # Try soundfile first
                waveform = None
                sr = Config.SAMPLE_RATE
                try:
                    data, sr_read = sf.read(audio_path, dtype='float32')
                    if len(data.shape) > 1:
                        data = data.mean(axis=1)
                    if sr_read != Config.SAMPLE_RATE:
                        data = librosa.resample(data, orig_sr=sr_read, target_sr=Config.SAMPLE_RATE)
                    waveform = data
                except Exception:
                    pass
                
                # Fallback to raw PCM
                if waveform is None:
                    with open(audio_path, 'rb') as f:
                        raw_data = f.read()
                    
                    # Try different offsets
                    for skip in [0, 44, 100, 128, 256, 512]:
                        if skip >= len(raw_data) - 1000:
                            continue
                        potential = raw_data[skip:]
                        byte_count = len(potential) // 2 * 2
                        if byte_count < 1000:
                            continue
                        samples = np.frombuffer(potential[:byte_count], dtype=np.int16)
                        samples_float = samples.astype(np.float32) / 32768.0
                        rms = np.sqrt(np.mean(samples_float**2))
                        max_amp = np.max(np.abs(samples_float))
                        
                        if 0.001 < rms < 1.0 and 0.01 < max_amp < 1.5:
                            # Test with 16kHz (most common)
                            if len(samples_float) > 16000:
                                duration = len(samples_float) / 16000
                                if 10 < duration < 3600:
                                    # Resample if needed
                                    if 16000 != Config.SAMPLE_RATE:
                                        samples_float = librosa.resample(
                                            samples_float, 
                                            orig_sr=16000, 
                                            target_sr=Config.SAMPLE_RATE
                                        )
                                    waveform = samples_float
                                    break
                
                if waveform is None or len(waveform) == 0:
                    raise RuntimeError("Could not decode audio file with any method")
            
            # Pad or truncate
            if len(waveform) > Config.MAX_LENGTH:
                waveform = waveform[:Config.MAX_LENGTH]
            else:
                padding = Config.MAX_LENGTH - len(waveform)
                waveform = np.pad(waveform, (0, padding), mode='constant')
            
            # Extract features
            mel_spec = extract_mel_spectrogram(waveform)
            features = torch.tensor(mel_spec).float().unsqueeze(0).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                features = features.to(self.device)
                outputs = self.model(features)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = probabilities.max(1)
                
                spoof_prob = probabilities[0, 1].item()
                bonafide_prob = probabilities[0, 0].item()
            
            result = {
                'prediction': 'spoof' if predicted.item() == 1 else 'bonafide',
                'prediction_label': 'Synthetic/Fake' if predicted.item() == 1 else 'Real/Authentic',
                'confidence': confidence.item(),
                'bonafide_probability': bonafide_prob,
                'spoof_probability': spoof_prob,
                'success': True,
                'audio_path': audio_path,
                'duration': len(waveform) / Config.SAMPLE_RATE
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'audio_path': audio_path
            }
