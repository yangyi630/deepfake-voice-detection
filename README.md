# DeepFake Voice Detection System

A comprehensive deep learning system for detecting AI-generated synthetic speech.

## Quick Start

**Method 1 (Recommended):** Double-click `start.bat`

**Method 2:** Run in terminal:
```bash
python app.py
```

## Project Structure

```
.
├── app.py                      # Main entry point (click to run)
├── start.bat                   # Windows launcher
├── requirements.txt            # Python dependencies
├── 使用说明.md                 # Chinese user guide
├── sample_data/               # Demo audio samples (auto-generated)
│   ├── bonafide/             # Real speech samples
│   ├── spoof/                # Synthetic speech samples
│   └── protocol.txt
└── src/                       # Source code
    ├── config.py             # Configuration
    ├── trainer.py            # Training & inference
    ├── models/               # Deep learning models
    │   ├── cnn.py           # SimpleCNN, ResNet18, LightCNN
    │   ├── conformer.py     # Conformer (Transformer + CNN)
    │   └── baseline.py      # Model factory
    ├── data/                 # Data processing
    │   ├── dataset.py       # Dataset classes
    │   └── sample_generator.py  # Demo data generation
    ├── utils/                # Utilities
    │   ├── metrics.py       # Evaluation metrics (EER, etc.)
    │   ├── visualization.py # Plotting functions
    │   └── theoretical_analysis.py  # Theoretical content
    └── ui/                   # GUI
        └── main_window.py    # Tkinter GUI application
```

## Features

### 1. Voice Detection
- Single audio file detection
- Supports WAV, FLAC, MP3, OGG, M4A formats
- Real-time visualization: waveform, mel spectrogram, F0 contour
- Confidence scores and probability breakdown

### 2. Deep Learning Models
| Model | Parameters | Description |
|-------|------------|-------------|
| Simple CNN | ~101K | Baseline 3-layer CNN |
| ResNet-18 | ~11.3M | Deep residual network |
| LightCNN | ~2.3M | Max-Feature-Map activation |
| Conformer | ~3.0M | CNN + Transformer hybrid |

### 3. Theoretical Analysis
7 comprehensive chapters covering:
- Research background and significance
- AI voice synthesis technologies (TTS, VC)
- Detection methods and architectures
- Feature analysis (spectral, prosodic, phase)
- Datasets and benchmarks (ASVspoof challenges)
- Current challenges (generalization, adversarial attacks)
- Future research directions

## System Requirements

- Python 3.8+
- 4GB+ RAM
- Windows/Linux/Mac
- Optional: NVIDIA GPU (for faster inference)

## First Run

1. Double-click `start.bat`
2. System will automatically:
   - Check and install dependencies
   - Generate sample audio data
   - Start GUI application
3. Click "Use Sample" to test with demo data
4. Or click "Browse..." to select your own audio file

## Technical Details

### Detection Approach
The system analyzes subtle artifacts in synthetic speech:
- **Spectral features**: Mel spectrogram, MFCC
- **Prosodic features**: F0 contour, energy variation
- **Phase features**: Phase spectrum, group delay
- **Quality features**: SNR, harmonic-to-noise ratio

### Model Architectures
All models use mel spectrogram as input:
```
Audio → Mel Spectrogram (80 bands) → CNN/Transformer → [Real/Fake]
```

### Performance Metrics
- **EER** (Equal Error Rate): Primary metric
- **AUC**: Area Under ROC Curve
- **Accuracy, Precision, Recall, F1**

## Support

For issues or questions, check `使用说明.md` for detailed instructions.

---

**Note:** This is an educational system. For production deployment, 
models need to be trained on datasets like ASVspoof 2019 LA.
