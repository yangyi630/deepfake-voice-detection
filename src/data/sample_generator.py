import os
import numpy as np
import soundfile as sf
from ..config import Config


def _generate_bonafide(duration, sr):
    """Generate realistic bonafide speech with natural characteristics."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Natural pitch variation (larger range, more dynamic)
    f0_base = 110 + 40 * np.sin(2 * np.pi * 1.5 * t)  # Large pitch modulation
    f0 = f0_base + 10 * np.sin(2 * np.pi * 5 * t)     # Fast micro-prosody
    
    waveform = np.zeros_like(t)
    
    # Natural harmonic series with formant resonances
    # Real speech has stronger lower harmonics, weaker higher ones
    for harmonic in range(1, 10):
        amplitude = 1.0 / (harmonic ** 1.5)
        # Add slight frequency jitter (natural imperfection)
        freq_jitter = 1.0 + 0.005 * np.random.randn(len(t))
        waveform += amplitude * np.sin(2 * np.pi * harmonic * f0 * freq_jitter * t)
    
    # Add natural breathiness/noise (more noise than synthetic)
    noise = 0.03 * np.random.randn(len(t))
    # Shape noise with formant envelope
    for formant_freq, bw in [(500, 80), (1500, 150), (2500, 200)]:
        formant_env = np.exp(-((np.arange(0, sr//2) - formant_freq)**2) / (2 * bw**2))
        # Simple noise shaping approximation
        noise *= 1.0
    
    # Apply time-varying amplitude envelope (natural speech rhythm)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t)
    envelope += 0.2 * np.sin(2 * np.pi * 7 * t)
    envelope = np.clip(envelope, 0, 1.5)
    waveform *= envelope
    
    # Add slight tremolo
    tremolo = 1.0 + 0.02 * np.sin(2 * np.pi * 6 * t)
    waveform *= tremolo
    
    waveform += noise
    
    return waveform


def _generate_spoof(duration, sr):
    """Generate synthetic/spoof speech with AI artifact characteristics."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Synthetic pitch - very stable, less natural variation
    f0 = 140 + 5 * np.sin(2 * np.pi * 0.8 * t)  # Very stable pitch
    
    waveform = np.zeros_like(t)
    
    # Perfect harmonic series (unnaturally perfect)
    # Synthetic speech often has flatter harmonic spectrum
    for harmonic in range(1, 15):
        amplitude = 1.0 / (harmonic ** 0.8)  # Flatter spectrum than real
        waveform += amplitude * np.sin(2 * np.pi * harmonic * f0 * t)
    
    # Add high-frequency artifacts (typical of TTS/voice conversion)
    artifact_freq1 = 3500 + 200 * np.sin(2 * np.pi * 8 * t)
    waveform += 0.04 * np.sin(2 * np.pi * artifact_freq1 * t)
    
    # Add metallic ringing artifact
    artifact_freq2 = 5000 + 100 * np.sin(2 * np.pi * 15 * t)
    waveform += 0.02 * np.sin(2 * np.pi * artifact_freq2 * t)
    
    # Phase artifacts (buzziness typical of parametric TTS)
    phase_distortion = 0.02 * np.cumsum(np.random.randn(len(t)) * 0.01)
    waveform += 0.02 * np.sin(2 * np.pi * 300 * t + phase_distortion)
    
    # Robotic/mechanical amplitude modulation
    am_mod = 1.0 + 0.03 * np.sin(2 * np.pi * 20 * t)
    waveform *= am_mod
    
    # Less natural noise (very low noise floor, unnaturally clean)
    waveform += 0.005 * np.random.randn(len(t))
    
    # Smoother amplitude envelope (less dynamic than real speech)
    envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 1.0 * t)
    envelope = np.clip(envelope, 0, 1.2)
    waveform *= envelope
    
    return waveform


def generate_sample_audio(duration=4.0, sr=Config.SAMPLE_RATE, voice_type='bonafide'):
    """
    Generate synthetic sample audio for demonstration purposes.
    
    This creates realistic-sounding speech-like signals that simulate
    either bonafide (real) or spoof (synthetic) speech characteristics.
    
    Args:
        duration: Audio duration in seconds
        sr: Sample rate
        voice_type: 'bonafide' or 'spoof'
    
    Returns:
        waveform: Generated audio waveform
    """
    if voice_type == 'bonafide':
        waveform = _generate_bonafide(duration, sr)
    else:
        waveform = _generate_spoof(duration, sr)
    
    # Normalize
    max_amp = np.max(np.abs(waveform))
    if max_amp > 0:
        waveform = waveform / max_amp * 0.7
    
    # Apply amplitude envelope (simulate speech onset/offset)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    envelope = np.ones_like(t)
    attack_samples = int(0.05 * sr)
    release_samples = int(0.05 * sr)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-release_samples:] = np.linspace(1, 0, release_samples)
    waveform *= envelope
    
    return waveform.astype(np.float32)


def generate_sample_data(output_dir=None, num_samples=10):
    """
    Generate sample audio files for demonstration.
    
    Creates a small dataset with bonafide and spoof samples
    that can be used to demonstrate the system without the full ASVspoof dataset.
    
    Args:
        output_dir: Output directory for sample files
        num_samples: Number of samples per class
    
    Returns:
        Dictionary with paths to generated files
    """
    if output_dir is None:
        output_dir = Config.SAMPLE_DATA_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    bonafide_dir = os.path.join(output_dir, 'bonafide')
    spoof_dir = os.path.join(output_dir, 'spoof')
    os.makedirs(bonafide_dir, exist_ok=True)
    os.makedirs(spoof_dir, exist_ok=True)
    
    generated_files = {
        'bonafide': [],
        'spoof': []
    }
    
    # Generate bonafide samples
    for i in range(num_samples):
        waveform = generate_sample_audio(duration=3.0, voice_type='bonafide')
        file_path = os.path.join(bonafide_dir, f'bonafide_{i:04d}.wav')
        sf.write(file_path, waveform, Config.SAMPLE_RATE)
        generated_files['bonafide'].append(file_path)
    
    # Generate spoof samples
    for i in range(num_samples):
        waveform = generate_sample_audio(duration=3.0, voice_type='spoof')
        file_path = os.path.join(spoof_dir, f'spoof_{i:04d}.wav')
        sf.write(file_path, waveform, Config.SAMPLE_RATE)
        generated_files['spoof'].append(file_path)
    
    # Create protocol file
    protocol_path = os.path.join(output_dir, 'protocol.txt')
    with open(protocol_path, 'w') as f:
        for i in range(num_samples):
            f.write(f"spk{i:04d} bonafide_{i:04d} - - bonafide\n")
        for i in range(num_samples):
            f.write(f"spk{i:04d} spoof_{i:04d} - - spoof\n")
    
    return generated_files, protocol_path


def ensure_sample_data():
    """Ensure sample data exists, generate if not present."""
    sample_dir = Config.SAMPLE_DATA_DIR
    protocol_file = os.path.join(sample_dir, 'protocol.txt')
    
    if not os.path.exists(protocol_file):
        generate_sample_data(sample_dir, num_samples=10)
    
    return sample_dir, protocol_file
