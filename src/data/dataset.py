import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset

from ..config import Config


def extract_mel_spectrogram(waveform, sr=Config.SAMPLE_RATE, n_mels=Config.N_MELS,
                            n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH):
    """
    Extract log-mel spectrogram from audio waveform.
    
    The mel spectrogram is a standard feature representation for speech
    that approximates human auditory perception. For deepfake detection,
    subtle artifacts in the spectral domain are key discriminative features.
    
    Args:
        waveform: Input audio waveform (numpy array)
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length between frames
    
    Returns:
        Log-mel spectrogram (n_mels x time_frames)
    """
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram


def extract_f0(waveform, sr=Config.SAMPLE_RATE):
    """
    Extract fundamental frequency (F0) contour using PYIN algorithm.
    F0 characteristics are important prosodic features that can reveal
    synthetic speech artifacts.
    
    Args:
        waveform: Input audio waveform
        sr: Sample rate
    
    Returns:
        f0: Fundamental frequency contour
        voiced_flag: Boolean array indicating voiced frames
    """
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            waveform,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        return f0, voiced_flag
    except Exception:
        return None, None


def extract_mfcc(waveform, sr=Config.SAMPLE_RATE, n_mfcc=20):
    """
    Extract MFCC (Mel-Frequency Cepstral Coefficients).
    MFCCs capture the spectral envelope of speech and are widely used
    in speaker verification and anti-spoofing systems.
    
    Args:
        waveform: Input audio waveform
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
    
    Returns:
        MFCC features (n_mfcc x time_frames)
    """
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def load_protocol(protocol_file_path):
    """
    Load ASVspoof protocol file.
    
    Protocol format (space-separated):
    speaker_id filename - - label
    
    Args:
        protocol_file_path: Path to protocol file
    
    Returns:
        DataFrame with 'filename' and 'label' columns
    """
    try:
        if not os.path.exists(protocol_file_path):
            return None
        
        protocol = pd.read_csv(
            protocol_file_path,
            sep=' ',
            header=None,
            names=['speaker', 'filename', 'unknown1', 'unknown2', 'label']
        )
        return protocol[['filename', 'label']]
    except Exception:
        return None


class ASVspoofDataset(Dataset):
    """
    Dataset class for ASVspoof 2019 Logical Access (LA) task.
    
    Supports loading from actual dataset or falling back to sample data
    when the full dataset is not available.
    """
    
    def __init__(self, protocol_df, audio_base_path, transform=None, 
                 max_length=Config.MAX_LENGTH, sample_rate=Config.SAMPLE_RATE):
        self.protocol_df = protocol_df
        self.audio_base_path = audio_base_path
        self.transform = transform
        self.max_length = max_length
        self.sample_rate = sample_rate
    
    def __len__(self):
        return len(self.protocol_df) if self.protocol_df is not None else 0
    
    def __getitem__(self, idx):
        if self.protocol_df is None:
            return torch.zeros(1, Config.N_MELS, 250).float(), 0
        
        row = self.protocol_df.iloc[idx]
        filename = row['filename']
        label = row['label']
        
        # Build audio path
        flac_dir = os.path.join(self.audio_base_path, 'flac')
        audio_path = os.path.join(flac_dir, f"{filename}.flac")
        
        # Check if file exists
        if not os.path.exists(audio_path):
            return torch.zeros(1, Config.N_MELS, 250).float(), 0
        
        try:
            # Load audio
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Pad or truncate to fixed length
            if len(waveform) > self.max_length:
                waveform = waveform[:self.max_length]
            else:
                padding = self.max_length - len(waveform)
                waveform = np.pad(waveform, (0, padding), mode='constant')
            
            # Extract features
            if self.transform:
                features = self.transform(waveform)
                if len(features.shape) == 2:
                    features = features[np.newaxis, :, :]  # Add channel dimension
                features = torch.tensor(features).float()
            else:
                features = torch.tensor(waveform).float()
            
            # Encode label: bonafide->0, spoof->1
            label_num = 0 if label == 'bonafide' else 1
            
            return features, label_num
        
        except Exception:
            dummy_features = torch.zeros(1, Config.N_MELS, 250).float()
            return dummy_features, 0


class SingleAudioDataset(Dataset):
    """
    Dataset for single audio file inference.
    """
    
    def __init__(self, audio_path, transform=None, max_length=Config.MAX_LENGTH,
                 sample_rate=Config.SAMPLE_RATE):
        self.audio_path = audio_path
        self.transform = transform
        self.max_length = max_length
        self.sample_rate = sample_rate
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        try:
            waveform, sr = librosa.load(self.audio_path, sr=self.sample_rate)
            
            if len(waveform) > self.max_length:
                waveform = waveform[:self.max_length]
            else:
                padding = self.max_length - len(waveform)
                waveform = np.pad(waveform, (0, padding), mode='constant')
            
            if self.transform:
                features = self.transform(waveform)
                if len(features.shape) == 2:
                    features = features[np.newaxis, :, :]
                features = torch.tensor(features).float()
            else:
                features = torch.tensor(waveform).float()
            
            return features, 0
        
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {e}")
