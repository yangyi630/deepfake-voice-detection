import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..config import Config
from ..trainer import Predictor
from ..data import extract_mel_spectrogram, extract_f0, generate_sample_data
from ..data.sample_generator import ensure_sample_data
from ..utils import plot_mel_spectrogram, plot_f0_contour, TheoreticalAnalyzer
from ..models import list_models


class DetectionApp:
    """
    Main GUI application for deepfake voice detection.
    
    Features:
    - Single audio file detection
    - Batch detection
    - Feature visualization
    - Theoretical analysis
    - Model comparison
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title(Config.WINDOW_TITLE)
        self.root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")
        self.root.minsize(900, 600)
        
        # Initialize components
        self.predictor = None
        self.current_audio_path = None
        self.current_result = None
        self.theoretical_analyzer = TheoreticalAnalyzer()
        
        # Setup UI
        self._setup_style()
        self._create_widgets()
        self._initialize_system()
    
    def _setup_style(self):
        """Setup ttk style."""
        style = ttk.Style()
        
        # Try modern theme
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        
        # Custom styles
        style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'))
        style.configure('Subtitle.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Status.TLabel', font=('Segoe UI', 9))
        style.configure('BigButton.TButton', font=('Segoe UI', 11, 'bold'), padding=10)
        style.configure('Card.TFrame', background='white', relief='flat')
        style.configure('Result.TLabel', font=('Segoe UI', 14, 'bold'))
    
    def _create_widgets(self):
        """Create all UI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title bar
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(title_frame, text="DeepFake Voice Detection System", 
                  style='Title.TLabel').pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar(value="Initializing...")
        ttk.Label(title_frame, textvariable=self.status_var, 
                  style='Status.TLabel').pack(side=tk.RIGHT)
        
        # Notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self._create_detection_tab()
        self._create_analysis_tab()
        self._create_theory_tab()
        self._create_models_tab()
    
    def _create_detection_tab(self):
        """Create the main detection tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="  Detection  ")
        
        # Left panel - controls
        left_frame = ttk.Frame(tab)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # File selection
        file_frame = ttk.LabelFrame(left_frame, text="Audio Input", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=40)
        file_entry.pack(fill=tk.X, pady=(0, 5))
        
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Browse...", 
                   command=self._browse_file).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Use Sample", 
                   command=self._load_sample).pack(side=tk.LEFT, padx=(5, 0))
        
        # Model selection
        model_frame = ttk.LabelFrame(left_frame, text="Model Selection", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.model_var = tk.StringVar(value='resnet18')
        models_info = list_models()
        model_names = list(models_info.keys())
        
        ttk.Label(model_frame, text="Select Model:").pack(anchor=tk.W)
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                   values=model_names, state='readonly', width=30)
        model_combo.pack(fill=tk.X, pady=(5, 5))
        model_combo.bind('<<ComboboxSelected>>', self._on_model_change)
        
        self.model_desc_var = tk.StringVar()
        ttk.Label(model_frame, textvariable=self.model_desc_var, 
                  wraplength=250, style='Status.TLabel').pack(anchor=tk.W)
        
        # Detect button
        self.detect_btn = ttk.Button(left_frame, text="Start Detection", 
                                      style='BigButton.TButton',
                                      command=self._start_detection)
        self.detect_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Result display
        result_frame = ttk.LabelFrame(left_frame, text="Detection Result", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_label = ttk.Label(result_frame, text="--", 
                                      style='Result.TLabel', anchor=tk.CENTER)
        self.result_label.pack(fill=tk.X, pady=(0, 10))
        
        self.confidence_var = tk.StringVar(value="Confidence: --")
        ttk.Label(result_frame, textvariable=self.confidence_var).pack(anchor=tk.W)
        
        self.bonafide_var = tk.StringVar(value="Real (Bonafide): --")
        ttk.Label(result_frame, textvariable=self.bonafide_var).pack(anchor=tk.W)
        
        self.spoof_var = tk.StringVar(value="Fake (Spoof): --")
        ttk.Label(result_frame, textvariable=self.spoof_var).pack(anchor=tk.W)
        
        self.duration_var = tk.StringVar(value="Duration: --")
        ttk.Label(result_frame, textvariable=self.duration_var).pack(anchor=tk.W)
        
        # Right panel - visualization
        right_frame = ttk.Frame(tab)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Visualization notebook
        viz_notebook = ttk.Notebook(right_frame)
        viz_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Waveform tab
        waveform_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(waveform_frame, text="Waveform")
        self.waveform_canvas_frame = waveform_frame
        
        # Mel spectrogram tab
        mel_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(mel_frame, text="Mel Spectrogram")
        self.mel_canvas_frame = mel_frame
        
        # F0 contour tab
        f0_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(f0_frame, text="F0 Contour")
        self.f0_canvas_frame = f0_frame
        
        # Initialize matplotlib figures
        self._init_plots()
    
    def _create_analysis_tab(self):
        """Create the feature analysis tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="  Feature Analysis  ")
        
        # Description
        desc_frame = ttk.LabelFrame(tab, text="Feature Analysis Overview", padding="10")
        desc_frame.pack(fill=tk.X, pady=(0, 10))
        
        desc_text = (
            "DeepFake voice detection relies on analyzing subtle artifacts in "
            "AI-synthesized speech. The key features include:\n\n"
            "1. Spectral Features: Mel spectrogram, MFCC - capture frequency-domain characteristics\n"
            "2. Prosodic Features: F0 contour, energy - capture rhythm and intonation patterns\n"
            "3. Phase Features: Phase spectrum, group delay - capture timing and phase information\n"
            "4. Quality Features: SNR, harmonic-to-noise ratio - capture speech quality differences"
        )
        ttk.Label(desc_frame, text=desc_text, wraplength=1000, justify=tk.LEFT).pack(anchor=tk.W)
        
        # Feature comparison
        comp_frame = ttk.LabelFrame(tab, text="Bonafide vs Spoof Feature Comparison", padding="10")
        comp_frame.pack(fill=tk.BOTH, expand=True)
        
        self.comp_canvas_frame = comp_frame
    
    def _create_theory_tab(self):
        """Create the theoretical analysis tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="  Theoretical Analysis  ")
        
        # Split into left (navigation) and right (content)
        nav_frame = ttk.Frame(tab)
        nav_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        ttk.Label(nav_frame, text="Topics", style='Subtitle.TLabel').pack(anchor=tk.W, pady=(0, 5))
        
        # Topic list
        self.theory_list = tk.Listbox(nav_frame, width=35, height=25, 
                                      font=('Segoe UI', 9))
        self.theory_list.pack(fill=tk.Y, expand=True)
        self.theory_list.bind('<<ListboxSelect>>', self._on_theory_select)
        
        # Content area
        content_frame = ttk.LabelFrame(tab, text="Content", padding="10")
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollable text widget
        text_frame = ttk.Frame(content_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.theory_text = tk.Text(text_frame, wrap=tk.WORD, font=('Segoe UI', 10),
                                   padx=10, pady=10, relief='flat')
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, 
                                  command=self.theory_text.yview)
        self.theory_text.configure(yscrollcommand=scrollbar.set)
        
        self.theory_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Make text read-only
        self.theory_text.configure(state=tk.DISABLED)
        
        # Populate theory topics
        self._populate_theory_topics()
    
    def _create_models_tab(self):
        """Create the model comparison tab."""
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text="  Models  ")
        
        # Overview
        overview_frame = ttk.LabelFrame(tab, text="Deep Learning Models for Voice Anti-Spoofing", padding="10")
        overview_frame.pack(fill=tk.X, pady=(0, 10))
        
        overview_text = (
            "This system implements several state-of-the-art deep learning architectures "
            "for deepfake voice detection. Each model has different strengths and is "
            "suited for different deployment scenarios."
        )
        ttk.Label(overview_frame, text=overview_text, wraplength=1000, 
                  justify=tk.LEFT).pack(anchor=tk.W)
        
        # Model details
        models_frame = ttk.Frame(tab)
        models_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create model info cards
        models_info = list_models()
        row = 0
        col = 0
        
        for model_id, model_info in models_info.items():
            card = ttk.LabelFrame(models_frame, text=model_info['name'], padding="10")
            card.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
            
            ttk.Label(card, text=model_info['description'], 
                      wraplength=250, justify=tk.LEFT).pack(anchor=tk.W)
            ttk.Label(card, text=f"Parameters: {model_info['params']}", 
                      style='Status.TLabel').pack(anchor=tk.W, pady=(5, 0))
            
            col += 1
            if col > 1:
                col = 0
                row += 1
        
        # Configure grid weights
        for i in range(2):
            models_frame.grid_columnconfigure(i, weight=1)
    
    def _init_plots(self):
        """Initialize matplotlib plots."""
        self.fig_waveform = Figure(figsize=(6, 3), dpi=100)
        self.ax_waveform = self.fig_waveform.add_subplot(111)
        self.ax_waveform.set_title('Waveform')
        self.ax_waveform.set_xlabel('Time (s)')
        self.ax_waveform.set_ylabel('Amplitude')
        self.ax_waveform.grid(True, alpha=0.3)
        self.fig_waveform.tight_layout()
        
        self.canvas_waveform = FigureCanvasTkAgg(self.fig_waveform, 
                                                  master=self.waveform_canvas_frame)
        self.canvas_waveform.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mel spectrogram
        self.fig_mel = Figure(figsize=(6, 3), dpi=100)
        self.ax_mel = self.fig_mel.add_subplot(111)
        self.ax_mel.set_title('Log-Mel Spectrogram')
        self.fig_mel.tight_layout()
        
        self.canvas_mel = FigureCanvasTkAgg(self.fig_mel, 
                                             master=self.mel_canvas_frame)
        self.canvas_mel.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # F0 contour
        self.fig_f0 = Figure(figsize=(6, 3), dpi=100)
        self.ax_f0 = self.fig_f0.add_subplot(111)
        self.ax_f0.set_title('F0 (Pitch) Contour')
        self.ax_f0.set_xlabel('Time (s)')
        self.ax_f0.set_ylabel('Frequency (Hz)')
        self.ax_f0.grid(True, alpha=0.3)
        self.fig_f0.tight_layout()
        
        self.canvas_f0 = FigureCanvasTkAgg(self.fig_f0, 
                                            master=self.f0_canvas_frame)
        self.canvas_f0.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _initialize_system(self):
        """Initialize the system on startup."""
        def init_worker():
            try:
                # Ensure directories
                Config.ensure_directories()
                
                # Generate sample data if not present
                self.status_var.set("Generating sample data...")
                self.root.update_idletasks()
                ensure_sample_data()
                
                # Load default model
                self.status_var.set("Loading model...")
                self.root.update_idletasks()
                self._load_model()
                
                # Update model description
                self._update_model_description()
                
                self.status_var.set("Ready")
                
            except Exception as e:
                self.status_var.set(f"Init error: {str(e)[:50]}")
        
        threading.Thread(target=init_worker, daemon=True).start()
    
    def _load_model(self):
        """Load the selected model."""
        model_name = self.model_var.get()
        model_path = os.path.join(Config.MODEL_SAVE_PATH, f'best_{model_name}.pth')
        
        # If no trained model exists, use untrained model for demo
        if not os.path.exists(model_path):
            # Try to find any saved model
            model_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pth')
        
        self.predictor = Predictor(model_name=model_name, model_path=model_path)
    
    def _browse_file(self):
        """Browse for audio file."""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.flac *.mp3 *.ogg *.m4a"),
                ("WAV Files", "*.wav"),
                ("FLAC Files", "*.flac"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.current_audio_path = file_path
            self._load_audio_features(file_path)
    
    def _load_sample(self):
        """Load a sample audio file."""
        sample_dir = os.path.join(Config.SAMPLE_DATA_DIR, 'bonafide')
        if os.path.exists(sample_dir):
            files = [f for f in os.listdir(sample_dir) if f.endswith('.wav')]
            if files:
                file_path = os.path.join(sample_dir, files[0])
                self.file_path_var.set(file_path)
                self.current_audio_path = file_path
                self._load_audio_features(file_path)
                return
        
        # Generate if not exists
        sample_dir, _ = ensure_sample_data()
        sample_file = os.path.join(sample_dir, 'bonafide', 'bonafide_0000.wav')
        if os.path.exists(sample_file):
            self.file_path_var.set(sample_file)
            self.current_audio_path = sample_file
            self._load_audio_features(sample_file)
    
    def _load_audio_features(self, audio_path):
        """Load and display audio features."""
        try:
            import librosa
            
            waveform, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
            
            # Update waveform plot
            self.ax_waveform.clear()
            time = np.arange(len(waveform)) / sr
            self.ax_waveform.plot(time, waveform, linewidth=0.5)
            self.ax_waveform.set_title('Waveform')
            self.ax_waveform.set_xlabel('Time (s)')
            self.ax_waveform.set_ylabel('Amplitude')
            self.ax_waveform.grid(True, alpha=0.3)
            self.fig_waveform.tight_layout()
            self.canvas_waveform.draw()
            
            # Update mel spectrogram
            mel_spec = extract_mel_spectrogram(waveform)
            self.ax_mel.clear()
            import librosa.display
            img = librosa.display.specshow(mel_spec, sr=sr, hop_length=Config.HOP_LENGTH,
                                           x_axis='time', y_axis='mel', ax=self.ax_mel)
            self.ax_mel.set_title('Log-Mel Spectrogram')
            self.fig_mel.colorbar(img, ax=self.ax_mel, format='%+2.0f dB')
            self.fig_mel.tight_layout()
            self.canvas_mel.draw()
            
            # Update F0 contour
            f0, _ = extract_f0(waveform, sr)
            self.ax_f0.clear()
            if f0 is not None:
                time_axis = np.arange(len(f0)) * 0.01  # ~10ms frames
                voiced = f0 > 0
                if np.any(voiced):
                    self.ax_f0.plot(time_axis[voiced], f0[voiced], 'b-', linewidth=1)
            self.ax_f0.set_title('F0 (Pitch) Contour')
            self.ax_f0.set_xlabel('Time (s)')
            self.ax_f0.set_ylabel('Frequency (Hz)')
            self.ax_f0.grid(True, alpha=0.3)
            self.fig_f0.tight_layout()
            self.canvas_f0.draw()
            
            self.status_var.set("Audio loaded successfully")
            
        except Exception as e:
            self.status_var.set(f"Error loading audio: {str(e)[:40]}")
            messagebox.showerror("Error", f"Failed to load audio:\n{str(e)}")
    
    def _start_detection(self):
        """Start detection process."""
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            messagebox.showwarning("Warning", "Please select an audio file first.")
            return
        
        if self.predictor is None:
            messagebox.showwarning("Warning", "Model not loaded. Please wait for initialization.")
            return
        
        self.detect_btn.config(state=tk.DISABLED)
        self.status_var.set("Analyzing...")
        
        def detect_worker():
            try:
                result = self.predictor.predict_audio(self.current_audio_path)
                
                self.root.after(0, lambda: self._update_result(result))
                
            except Exception as e:
                self.root.after(0, lambda: self._detection_error(str(e)))
        
        threading.Thread(target=detect_worker, daemon=True).start()
    
    def _update_result(self, result):
        """Update UI with detection result."""
        self.detect_btn.config(state=tk.NORMAL)
        self.current_result = result
        
        if result['success']:
            if result['prediction'] == 'spoof':
                self.result_label.config(text="SYNTHETIC / FAKE", foreground='red')
            else:
                self.result_label.config(text="REAL / AUTHENTIC", foreground='green')
            
            self.confidence_var.set(f"Confidence: {result['confidence']*100:.2f}%")
            self.bonafide_var.set(f"Real (Bonafide): {result['bonafide_probability']*100:.2f}%")
            self.spoof_var.set(f"Fake (Spoof): {result['spoof_probability']*100:.2f}%")
            self.duration_var.set(f"Duration: {result['duration']:.2f}s")
            
            self.status_var.set("Detection complete")
        else:
            self.result_label.config(text="ERROR", foreground='red')
            self.status_var.set(f"Detection failed: {result.get('error', 'Unknown error')[:40]}")
            messagebox.showerror("Error", f"Detection failed:\n{result.get('error', 'Unknown error')}")
    
    def _detection_error(self, error_msg):
        """Handle detection error."""
        self.detect_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Error: {error_msg[:40]}")
        messagebox.showerror("Error", f"Detection failed:\n{error_msg}")
    
    def _on_model_change(self, event=None):
        """Handle model selection change."""
        self._update_model_description()
        
        def reload_worker():
            try:
                self.status_var.set("Loading model...")
                self._load_model()
                self.status_var.set("Model loaded")
            except Exception as e:
                self.status_var.set(f"Model load error: {str(e)[:40]}")
        
        threading.Thread(target=reload_worker, daemon=True).start()
    
    def _update_model_description(self):
        """Update model description text."""
        model_name = self.model_var.get()
        models_info = list_models()
        if model_name in models_info:
            info = models_info[model_name]
            self.model_desc_var.set(info['description'])
    
    def _populate_theory_topics(self):
        """Populate theory topics list."""
        content = self.theoretical_analyzer.get_all_sections()
        for key, section in content.items():
            self.theory_list.insert(tk.END, section['title'])
        
        # Select first item
        if self.theory_list.size() > 0:
            self.theory_list.selection_set(0)
            self._on_theory_select(None)
    
    def _on_theory_select(self, event):
        """Handle theory topic selection."""
        selection = self.theory_list.curselection()
        if not selection:
            return
        
        idx = selection[0]
        content = self.theoretical_analyzer.get_all_sections()
        keys = list(content.keys())
        key = keys[idx]
        section = content[key]
        
        # Build text content
        text = f"{section['title']}\n\n"
        text += "=" * 60 + "\n\n"
        
        for subsection in section['sections']:
            text += f"{subsection['subtitle']}\n\n"
            text += subsection['content']
            text += "\n\n" + "-" * 40 + "\n\n"
        
        # Update text widget
        self.theory_text.configure(state=tk.NORMAL)
        self.theory_text.delete(1.0, tk.END)
        self.theory_text.insert(1.0, text)
        self.theory_text.configure(state=tk.DISABLED)


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = DetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
