"""
Flask backend server for DeepFake Voice Detection System.
Provides REST API endpoints and serves the web frontend.
"""

import os
import sys
import json
import base64
import io
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

from src.config import Config
from src.models import list_models
from src.trainer import Predictor
from src.data.sample_generator import ensure_sample_data, generate_sample_data
from src.utils.theoretical_analysis import TheoreticalAnalyzer

import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display


app = Flask(__name__, 
            static_folder='../web/static',
            template_folder='../web/templates')

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(Config.ROOT_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global predictors cache
_predictors = {}
_predictor_lock = threading.Lock()

# Theoretical analyzer
_analyzer = TheoreticalAnalyzer()

# Sample data status
_sample_data_ready = False


def get_predictor(model_name):
    """Get or create a predictor for the given model."""
    with _predictor_lock:
        if model_name not in _predictors:
            model_path = os.path.join(Config.MODEL_SAVE_PATH, f'best_{model_name}.pth')
            if not os.path.exists(model_path):
                model_path = None
            _predictors[model_name] = Predictor(model_name=model_name, model_path=model_path)
        return _predictors[model_name]


def ensure_system_ready():
    """Ensure system components are initialized and ready."""
    global _sample_data_ready
    if not _sample_data_ready:
        Config.ensure_directories()
        ensure_sample_data()
        
        # Auto-train a default model if none exists
        default_model = 'simple_cnn'
        model_path = os.path.join(Config.MODEL_SAVE_PATH, f'best_{default_model}.pth')
        if not os.path.exists(model_path):
            print(f"No model found for {default_model}, auto-training on sample data...")
            try:
                from train_sample_models import train_sample_model
                train_sample_model(model_name=default_model, num_epochs=50, num_samples=80)
                print(f"Auto-training complete: {default_model}")
            except Exception as e:
                print(f"Warning: Auto-training failed: {e}")
                print("System will still work but predictions will be random.")
        
        _sample_data_ready = True


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def detect_audio_format(audio_path):
    """
    Detect if file is a standard audio format or raw PCM.
    Returns: 'standard' or 'raw_pcm'
    """
    import numpy as np
    
    with open(audio_path, 'rb') as f:
        data = f.read()
    
    if len(data) < 100:
        return 'unknown'
    
    # Check for standard audio format signatures
    if data[:4] == b'RIFF' and b'WAVE' in data[:12]:
        return 'wav'
    if data[:3] == b'ID3':
        return 'mp3'
    if data[:4] == b'fLaC':
        return 'flac'
    if data[:4] == b'OggS':
        return 'ogg'
    if data[:2] in [b'\xff\xfb', b'\xff\xf3', b'\xff\xf2', b'\xff\xfa']:
        return 'mp3'
    
    # Check for MP3 sync word within first 1000 bytes
    for i in range(min(1000, len(data)-1)):
        if data[i] == 0xFF and (data[i+1] & 0xE0) == 0xE0:
            return 'mp3'
    
    # Check if it looks like raw PCM
    # Raw PCM files often have a small header followed by audio data
    # Test if interpreting as 16-bit PCM gives reasonable audio-like values
    for skip in [0, 44, 100, 128, 256, 512]:
        if skip + 1000 > len(data):
            continue
        test_data = data[skip:skip+2000]
        samples = np.frombuffer(test_data[:len(test_data)//2*2], dtype=np.int16)
        if len(samples) < 100:
            continue
        
        rms = np.sqrt(np.mean(samples.astype(np.float32)**2))
        max_val = np.max(np.abs(samples))
        
        # Audio-like signal: not all zeros, has reasonable dynamic range
        if rms > 10 and max_val > 100 and max_val < 32768:
            return 'raw_pcm'
    
    return 'unknown'


def try_raw_pcm_load(audio_path, target_sr):
    """
    Try to load audio file as raw PCM with various header sizes and sample rates.
    Returns (waveform, sample_rate) on success, None on failure.
    """
    import numpy as np
    
    with open(audio_path, 'rb') as f:
        raw_data = f.read()
    
    file_size = len(raw_data)
    if file_size < 1000:
        return None
    
    # Try different header sizes and sample rates
    for skip in [0, 44, 100, 128, 256, 512, 1024, 2048]:
        if skip >= file_size - 1000:
            break
            
        potential_data = raw_data[skip:]
        if len(potential_data) < 1000:
            continue
        
        # Try 16-bit PCM (most common)
        try:
            byte_count = len(potential_data) // 2 * 2
            samples = np.frombuffer(potential_data[:byte_count], dtype=np.int16)
            if len(samples) < 100:
                continue
            
            samples_float = samples.astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(samples_float**2))
            max_amp = np.max(np.abs(samples_float))
            
            # Check if it looks like valid audio
            if rms < 0.001 or max_amp < 0.01:
                continue
            if max_amp > 1.5:  # should be within [-1, 1] after normalization
                continue
            
            # Try different sample rates
            for sr_candidate in [16000, 44100, 48000, 22050, 8000, 32000]:
                duration = len(samples) / sr_candidate
                # Duration should be reasonable (1s to 60min)
                if 1 < duration < 3600:
                    # Resample to target rate if needed
                    result = samples_float
                    result_sr = sr_candidate
                    if sr_candidate != target_sr:
                        import librosa
                        result = librosa.resample(
                            result, 
                            orig_sr=sr_candidate, 
                            target_sr=target_sr
                        )
                        result_sr = target_sr
                    
                    print(f"Raw PCM detected: skip={skip}B, sr={sr_candidate}Hz, duration={duration:.1f}s")
                    return result, result_sr
        except Exception:
            continue
    
    # Also try 8-bit PCM
    for skip in [0, 44, 100, 128, 256]:
        if skip >= file_size - 1000:
            break
        potential_data = raw_data[skip:]
        try:
            samples = np.frombuffer(potential_data[:len(potential_data)//2*2], dtype=np.uint8)
            if len(samples) < 100:
                continue
            samples_float = (samples.astype(np.float32) - 128) / 128.0
            rms = np.sqrt(np.mean(samples_float**2))
            max_amp = np.max(np.abs(samples_float))
            if 0.01 < rms < 1.0 and max_amp > 0.1:
                for sr_candidate in [16000, 44100, 22050, 8000]:
                    duration = len(samples) / sr_candidate
                    if 1 < duration < 3600:
                        result = samples_float
                        if sr_candidate != target_sr:
                            import librosa
                            result = librosa.resample(result, orig_sr=sr_candidate, target_sr=target_sr)
                        print(f"8-bit PCM detected: skip={skip}B, sr={sr_candidate}Hz")
                        return result, target_sr
        except Exception:
            continue
    
    return None


def load_audio_robust(audio_path, target_sr=Config.SAMPLE_RATE):
    """
    Robust audio loading that tries multiple backends and validates results.
    Supports WAV, FLAC, MP3, OGG, and raw PCM formats.
    """
    import soundfile as sf
    import numpy as np
    file_size = os.path.getsize(audio_path)
    
    errors = []
    
    # First detect format to choose strategy
    fmt = detect_audio_format(audio_path)
    print(f"Detected audio format: {fmt}")
    
    # If it looks like raw PCM, try that first (faster for non-standard formats)
    if fmt == 'raw_pcm' or fmt == 'unknown':
        result = try_raw_pcm_load(audio_path, target_sr)
        if result is not None:
            return result
        errors.append("raw_pcm: tried but failed to decode")
    
    # Method 1: soundfile (native MP3/WAV/FLAC support)
    try:
        data, sr = sf.read(audio_path, dtype='float32')
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        
        duration = len(data) / sr if sr > 0 else 0
        
        if len(data) == 0:
            errors.append("soundfile: returned empty audio")
        elif duration < 0.1 and file_size > 10000:
            errors.append(f"soundfile: suspiciously short ({duration:.3f}s) for {file_size}B")
        else:
            if sr != target_sr:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            return data, sr
    except Exception as sf_err:
        errors.append(f"soundfile error: {str(sf_err)[:100]}")
    
    # Method 2: librosa
    try:
        import librosa
        data, sr = librosa.load(audio_path, sr=target_sr, res_type='kaiser_fast')
        
        duration = len(data) / sr if sr > 0 else 0
        if len(data) == 0:
            errors.append("librosa: returned empty audio")
        elif duration < 0.1 and file_size > 10000:
            errors.append(f"librosa: suspiciously short ({duration:.3f}s) for {file_size}B")
        else:
            return data, sr
    except Exception as librosa_err:
        errors.append(f"librosa error: {str(librosa_err)[:100]}")
    
    # Method 3: Try pydub if available
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(target_sr)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.sample_width == 2:
            samples /= 32768.0
        elif audio.sample_width == 4:
            samples /= 2147483648.0
        
        if len(samples) == 0:
            errors.append("pydub: returned empty audio")
        else:
            return samples, target_sr
    except ImportError:
        pass
    except Exception as pydub_err:
        errors.append(f"pydub error: {str(pydub_err)[:100]}")
    
    # If we haven't tried raw PCM yet (because format was detected as standard), try it as fallback
    if fmt in ['wav', 'mp3', 'flac', 'ogg']:
        result = try_raw_pcm_load(audio_path, target_sr)
        if result is not None:
            return result
        errors.append("raw_pcm (fallback): failed")
    
    # All methods failed
    error_msg = "Failed to load audio file. Tried: " + "; ".join(errors)
    error_msg += " | Formats: WAV, FLAC, MP3, OGG, raw PCM"
    raise RuntimeError(error_msg)


def generate_visualizations(audio_path):
    """Generate waveform, mel spectrogram, and F0 contour visualizations."""
    try:
        waveform, sr = load_audio_robust(audio_path, Config.SAMPLE_RATE)
        
        # Trim to max length
        if len(waveform) > Config.MAX_LENGTH:
            waveform = waveform[:Config.MAX_LENGTH]
        
        results = {}
        
        # Waveform
        fig, ax = plt.subplots(figsize=(10, 3))
        time = np.arange(len(waveform)) / sr
        ax.plot(time, waveform, linewidth=0.5, color='#2c3e50')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Waveform', fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        results['waveform'] = fig_to_base64(fig)
        
        # Mel spectrogram
        from src.data.dataset import extract_mel_spectrogram
        mel_spec = extract_mel_spectrogram(waveform)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(mel_spec, sr=sr, hop_length=Config.HOP_LENGTH,
                                       x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Log-Mel Spectrogram', fontweight='bold')
        fig.tight_layout()
        results['mel_spectrogram'] = fig_to_base64(fig)
        
        # F0 contour
        from src.data.dataset import extract_f0
        f0, voiced = extract_f0(waveform, sr)
        
        fig, ax = plt.subplots(figsize=(10, 3))
        if f0 is not None:
            time_axis = np.arange(len(f0)) * 0.01
            voiced_mask = f0 > 0
            if np.any(voiced_mask):
                ax.plot(time_axis[voiced_mask], f0[voiced_mask], 
                       color='#e74c3c', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('F0 (Hz)')
        ax.set_title('F0 (Pitch) Contour', fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        results['f0_contour'] = fig_to_base64(fig)
        
        return results
        
    except Exception as e:
        return {'error': str(e)}


# ==================== Routes ====================

@app.route('/')
def index():
    """Main detection page."""
    ensure_system_ready()
    return render_template('index.html')


@app.route('/theory')
def theory():
    """Theoretical analysis page."""
    return render_template('theory.html')


@app.route('/models')
def models_page():
    """Models introduction page."""
    return render_template('models.html')


@app.route('/api/status')
def api_status():
    """Get system status."""
    ensure_system_ready()
    models = list_models()
    return jsonify({
        'status': 'ready',
        'device': Config.DEVICE,
        'available_models': list(models.keys()),
        'sample_data_dir': Config.SAMPLE_DATA_DIR
    })


@app.route('/api/models')
def api_models():
    """Get list of available models with details."""
    models = list_models()
    return jsonify(models)


@app.route('/api/samples')
def api_samples():
    """Get list of sample audio files."""
    sample_dir = Config.SAMPLE_DATA_DIR
    result = {'bonafide': [], 'spoof': []}
    
    for category in ['bonafide', 'spoof']:
        cat_dir = os.path.join(sample_dir, category)
        if os.path.exists(cat_dir):
            files = sorted([f for f in os.listdir(cat_dir) if f.endswith(('.wav', '.flac'))])
            result[category] = [{'name': f, 'path': os.path.join(cat_dir, f)} for f in files]
    
    return jsonify(result)


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """Detect if audio is real or fake with timeout protection."""
    audio_path = None
    is_uploaded = False
    task_id = None
    
    try:
        model_name = request.form.get('model', 'simple_cnn')
        sample_type = request.form.get('sample_type', None)
        sample_name = request.form.get('sample_name', None)
        
        # Check file size first
        if 'audio' in request.files:
            file = request.files['audio']
            content_length = request.content_length or 0
            max_size = 100 * 1024 * 1024  # 100MB max
            if content_length > max_size:
                return jsonify({
                    'success': False, 
                    'error': f'File too large. Maximum size is 100MB. Your file: {content_length / (1024*1024):.1f}MB'
                }), 413
        
        # Check if file uploaded
        if 'audio' in request.files and request.files['audio'].filename:
            file = request.files['audio']
            original_name = file.filename
            
            # Use unique temp filename to avoid issues
            import uuid
            ext = os.path.splitext(original_name)[1].lower()
            if not ext:
                ext = '.wav'
            temp_filename = f"upload_{uuid.uuid4().hex}{ext}"
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            
            # Save file directly without loading into memory
            file.save(audio_path)
            is_uploaded = True
            
            # Get file size
            file_size = os.path.getsize(audio_path)
            print(f"Uploaded file: {original_name}, size: {file_size / (1024*1024):.2f}MB")
        
        # Or use sample
        elif sample_type and sample_name:
            sample_dir = os.path.join(Config.SAMPLE_DATA_DIR, sample_type)
            candidate = os.path.join(sample_dir, sample_name)
            if os.path.exists(candidate):
                audio_path = candidate
        
        if not audio_path or not os.path.exists(audio_path):
            return jsonify({'success': False, 'error': 'No audio file provided or file not found'}), 400
        
        # Verify audio can be loaded
        try:
            waveform, sr = load_audio_robust(audio_path, Config.SAMPLE_RATE)
            duration = len(waveform) / sr
            file_size = os.path.getsize(audio_path)
            print(f"Audio loaded successfully: {duration:.2f}s, {len(waveform)} samples, sr={sr}, file_size={file_size} bytes")
            
            if len(waveform) == 0:
                return jsonify({
                    'success': False, 
                    'error': 'Audio file is empty or corrupted. Please check the file.'
                }), 400
            # Check minimum duration (at least 0.5 seconds)
            if len(waveform) < Config.SAMPLE_RATE * 0.5:
                return jsonify({
                    'success': False, 
                    'error': f'Audio file too short ({duration:.2f}s). Minimum 0.5 seconds required.'
                }), 400
            
            # For very long audio (>30s), truncate to avoid memory issues
            max_duration_samples = Config.SAMPLE_RATE * 30
            if len(waveform) > max_duration_samples:
                print(f"Audio too long ({duration:.1f}s), truncating to 30s")
                waveform = waveform[:max_duration_samples]
                
        except Exception as load_err:
            error_msg = str(load_err)
            print(f"Failed to load audio: {error_msg}")
            return jsonify({
                'success': False, 
                'error': error_msg
            }), 400
        
        # Run detection with timeout protection
        result = None
        detection_error = None
        
        def run_detection():
            nonlocal result, detection_error
            try:
                predictor = get_predictor(model_name)
                result = predictor.predict_audio(audio_path)
            except Exception as e:
                detection_error = str(e)
        
        # Run detection in thread with timeout
        detection_thread = threading.Thread(target=run_detection)
        detection_thread.daemon = True
        detection_thread.start()
        detection_thread.join(timeout=120)  # 2 minute timeout for detection
        
        if detection_thread.is_alive():
            # Thread still running, timeout occurred
            return jsonify({
                'success': False, 
                'error': 'Detection timeout. The audio file is too long or the server is overloaded. '
                         'Please try with a shorter audio file (max 30 seconds).'
            }), 408
        
        if detection_error:
            return jsonify({'success': False, 'error': f'Detection error: {detection_error}'}), 500
        
        if result is None:
            return jsonify({'success': False, 'error': 'Detection failed silently. Please try again.'}), 500
        
        # Generate visualizations (optional, skip for large files)
        try:
            viz = generate_visualizations(audio_path)
            result['visualizations'] = viz
        except Exception as viz_err:
            print(f"Visualization generation warning: {viz_err}")
            result['visualizations'] = {}
        
        # Clean up uploaded file
        if is_uploaded and audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass
        
        return jsonify(result)
        
    except Exception as e:
        # Clean up on error
        if is_uploaded and audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass
        
        import traceback
        error_detail = traceback.format_exc()
        print(f"Detection API error:\n{error_detail}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500


@app.route('/api/theory')
def api_theory():
    """Get theoretical analysis content."""
    content = _analyzer.get_all_sections()
    return jsonify(content)


@app.route('/api/theory/<section_key>')
def api_theory_section(section_key):
    """Get specific theoretical section."""
    section = _analyzer.get_section(section_key)
    if section:
        return jsonify(section)
    return jsonify({'error': 'Section not found'}), 404


def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask server."""
    ensure_system_ready()
    print(f"\nServer running at http://{host}:{port}")
    print(f"Open your browser and visit http://localhost:{port}")
    print(f"Press Ctrl+C to stop the server\n")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    run_server()
