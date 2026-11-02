import os


def _get_device():
    """Get available device (cuda or cpu). Lazy import torch."""
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


class Config:
    """Configuration class for deep fake voice detection system"""
    
    # Project paths
    # config.py is at src/config.py, so src directory is its parent
    # Project root is one level up from src
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SRC_DIR)  # Project root directory
    ROOT_DIR = BASE_DIR
    
    # Data paths
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    SAMPLE_DATA_DIR = os.path.join(ROOT_DIR, 'sample_data')
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'saved_models')
    RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
    
    # ASVspoof dataset paths (optional, for full training)
    ASVSPOOF_BASE_PATH = os.path.join(DATA_DIR, 'LA')
    PROTOCOLS_PATH = os.path.join(ASVSPOOF_BASE_PATH, 'ASVspoof2019_LA_cm_protocols')
    TRAIN_AUDIO_PATH = os.path.join(ASVSPOOF_BASE_PATH, 'ASVspoof2019_LA_train')
    DEV_AUDIO_PATH = os.path.join(ASVSPOOF_BASE_PATH, 'ASVspoof2019_LA_dev')
    EVAL_AUDIO_PATH = os.path.join(ASVSPOOF_BASE_PATH, 'ASVspoof2019_LA_eval')
    
    # Protocol files
    TRAIN_PROTOCOL_FILE = 'ASVspoof2019.LA.cm.train.trn.txt'
    DEV_PROTOCOL_FILE = 'ASVspoof2019.LA.cm.dev.trl.txt'
    EVAL_PROTOCOL_FILE = 'ASVspoof2019.LA.cm.eval.trl.txt'
    
    # Audio processing parameters
    SAMPLE_RATE = 16000
    N_MELS = 80
    N_FFT = 512
    HOP_LENGTH = 256
    MAX_LENGTH = 64000  # 4 seconds at 16kHz
    
    # Training parameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    DEVICE = _get_device()
    
    # Model parameters
    DEFAULT_MODEL = 'resnet18'
    NUM_CLASSES = 2
    
    # GUI parameters
    WINDOW_TITLE = "DeepFake Voice Detection System"
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all necessary directories exist"""
        dirs_to_create = [
            cls.DATA_DIR,
            cls.SAMPLE_DATA_DIR,
            cls.MODEL_SAVE_PATH,
            cls.RESULTS_DIR
        ]
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        return True
