#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepFake Voice Detection System
Main entry point - click this file to run the application.

Starts a Flask web server and opens the browser automatically.
All dependencies are checked and installed if missing.
"""

import os
import sys
import time
import webbrowser
import threading

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.config import Config

# Package mapping: import_name -> pip_name
PACKAGE_MAP = {
    'flask': 'flask',
    'numpy': 'numpy',
    'torch': 'torch',
    'torchaudio': 'torchaudio',
    'librosa': 'librosa',
    'soundfile': 'soundfile',
    'matplotlib': 'matplotlib',
    'sklearn': 'scikit-learn',
    'tqdm': 'tqdm',
    'seaborn': 'seaborn',
    'PIL': 'Pillow',
    'pandas': 'pandas',
    'werkzeug': 'flask',
}

# Packages that must be installed (in order of dependency)
REQUIRED_PACKAGES = [
    'numpy',
    'pandas',
    'matplotlib',
    'seaborn',
    'scikit-learn',
    'tqdm',
    'Pillow',
    'soundfile',
    'librosa',
    'torch',
    'flask',
    'werkzeug',
]


def check_package(import_name):
    """Check if a package is importable."""
    try:
        if import_name == 'sklearn':
            __import__('sklearn')
        elif import_name == 'PIL':
            __import__('PIL')
        else:
            __import__(import_name)
        return True
    except ImportError:
        return False


def check_environment():
    """Check all required dependencies, return list of missing pip package names."""
    missing = []
    checked = set()
    
    for import_name, pip_name in PACKAGE_MAP.items():
        if pip_name in checked:
            continue
        if not check_package(import_name):
            if pip_name not in missing:
                missing.append(pip_name)
        checked.add(pip_name)
    
    return missing


def install_packages(packages):
    """Install packages using pip, returns True on success."""
    import subprocess
    
    print(f"\nInstalling {len(packages)} package(s)...")
    
    # Install one by one to show progress
    success_count = 0
    for i, pkg in enumerate(packages, 1):
        print(f"\n  [{i}/{len(packages)}] Installing {pkg}...")
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', pkg, '--quiet']
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                success_count += 1
                print(f"      {pkg} installed successfully")
            else:
                print(f"      Warning: {pkg} may have issues: {result.stderr.strip()[:100]}")
        except Exception as e:
            print(f"      Error installing {pkg}: {e}")
    
    return success_count > 0


def open_browser_delayed(url, delay=2):
    """Open browser after a delay to let server start."""
    def _open():
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass
    
    threading.Thread(target=_open, daemon=True).start()


def print_banner():
    """Print startup banner."""
    print()
    print("=" * 65)
    print("    DeepFake Voice Detection System")
    print("    Based on Deep Learning")
    print("=" * 65)
    print()


def main():
    """Main entry function."""
    print_banner()
    
    # Step 1: Ensure directories
    print("[Step 1/5] Initializing directories...")
    Config.ensure_directories()
    print("  Directories ready.")
    
    # Step 2: Check dependencies
    print("\n[Step 2/5] Checking dependencies...")
    missing = check_environment()
    
    if missing:
        print(f"  Missing {len(missing)} package(s): {', '.join(missing)}")
        print("  Installing automatically (this may take several minutes)...")
        install_packages(missing)
        
        # Re-check after installation
        print("\n  Re-checking dependencies...")
        still_missing = check_environment()
        if still_missing:
            print(f"  Warning: {len(still_missing)} package(s) still missing: {', '.join(still_missing)}")
            print("  You may need to install them manually.")
        else:
            print("  All dependencies installed successfully!")
    else:
        print("  All dependencies found.")
    
    # Step 3: Prepare sample data
    print("\n[Step 3/5] Preparing sample data...")
    try:
        from src.data.sample_generator import ensure_sample_data
        sample_dir, protocol = ensure_sample_data()
        print(f"  Sample data ready: {sample_dir}")
    except Exception as e:
        print(f"  Warning: Sample data generation issue: {e}")
    
    # Step 4: Pre-load default model
    print("\n[Step 4/5] Loading detection model...")
    try:
        from src.trainer import Predictor
        predictor = Predictor(model_name='resnet18')
        print("  Model loaded successfully.")
    except Exception as e:
        print(f"  Note: Model will load on first use: {e}")
    
    # Step 5: Start server
    print("\n[Step 5/5] Starting web server...")
    
    host = '127.0.0.1'
    port = 5000
    
    # Find available port if 5000 is taken
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            # Port taken, try next
            for p in range(5001, 5010):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                r = sock.connect_ex((host, p))
                sock.close()
                if r != 0:
                    port = p
                    break
    except:
        pass
    
    server_url = f"http://{host}:{port}"
    
    print(f"\n{'=' * 65}")
    print(f"  System Ready!")
    print(f"  Access URL: {server_url}")
    print(f"  (Browser should open automatically)")
    print(f"  Press Ctrl+C to stop the server")
    print(f"{'=' * 65}")
    print()
    
    # Open browser
    open_browser_delayed(server_url, delay=1.5)
    
    # Start Flask server
    try:
        from src.server import app
        app.run(host=host, port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
    except Exception as e:
        print(f"\nServer error: {e}")
        print("\nPress Enter to exit...")
        try:
            input()
        except:
            pass


if __name__ == "__main__":
    main()
