/**
 * DeepFake Voice Detection - Main JavaScript
 * Handles detection logic and visualization
 */

let currentFile = null;
let currentSampleType = null;
let currentSampleName = null;
let isDetecting = false;

// Drag and drop
const uploadArea = document.getElementById('uploadArea');
const audioInput = document.getElementById('audioInput');

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('audio/')) {
        handleFile(files[0]);
    }
});

audioInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    // Validate file size (max 50MB)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
        showError('文件太大，最大支持 50MB 的音频文件');
        return;
    }
    
    // Validate file type
    const allowedTypes = ['audio/wav', 'audio/flac', 'audio/mp3', 'audio/mpeg', 'audio/ogg', 'audio/x-m4a', 'audio/mp4'];
    const allowedExts = ['.wav', '.flac', '.mp3', '.ogg', '.m4a', '.aac'];
    const fileName = file.name.toLowerCase();
    const ext = fileName.substring(fileName.lastIndexOf('.'));
    
    if (file.type && !file.type.startsWith('audio/') && !allowedTypes.includes(file.type)) {
        if (!allowedExts.includes(ext)) {
            showError('不支持的音频格式，请上传 WAV, FLAC, MP3, OGG 或 M4A 格式');
            return;
        }
    }
    
    currentFile = file;
    currentSampleType = null;
    currentSampleName = null;
    
    const uploadText = document.querySelector('.upload-text');
    const uploadHint = document.querySelector('.upload-hint');
    
    uploadText.textContent = file.name;
    uploadHint.textContent = formatFileSize(file.size);
    
    // Reset result
    resetResult();
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

function loadSample(type, name) {
    currentFile = null;
    currentSampleType = type;
    currentSampleName = name;
    
    const uploadText = document.querySelector('.upload-text');
    const uploadHint = document.querySelector('.upload-hint');
    
    uploadText.textContent = name;
    uploadHint.textContent = type === 'bonafide' ? '真实语音样本' : '虚假语音样本';
    
    resetResult();
}

function resetResult() {
    const statusEl = document.getElementById('resultStatus');
    statusEl.className = 'result-status idle';
    statusEl.textContent = '等待检测';
    
    // Reset progress section
    const progressSection = document.getElementById('progressSection');
    progressSection.style.display = 'none';
    progressSection.innerHTML = `
        <div class="progress-item">
            <div class="progress-label">
                <span>真实语音 (Bonafide)</span>
                <span id="realPercent">--%</span>
            </div>
            <div class="progress-bar">
                <div id="realProgress" class="progress-fill real" style="width: 0%"></div>
            </div>
        </div>
        <div class="progress-item">
            <div class="progress-label">
                <span>虚假语音 (Spoof)</span>
                <span id="fakePercent">--%</span>
            </div>
            <div class="progress-bar">
                <div id="fakeProgress" class="progress-fill fake" style="width: 0%"></div>
            </div>
        </div>
    `;
    
    document.getElementById('infoSection').style.display = 'none';
    
    // Hide images
    document.getElementById('waveformImage').style.display = 'none';
    document.getElementById('melImage').style.display = 'none';
    document.getElementById('f0Image').style.display = 'none';
    document.getElementById('waveformPlaceholder').style.display = 'flex';
    document.getElementById('melPlaceholder').style.display = 'flex';
    document.getElementById('f0Placeholder').style.display = 'flex';
}

function switchTab(tabName) {
    // Update tabs
    document.querySelectorAll('.viz-tab').forEach(tab => tab.classList.remove('active'));
    event.target.classList.add('active');
    
    // Update content
    document.querySelectorAll('.viz-content').forEach(content => content.classList.remove('active'));
    document.getElementById(tabName + 'Tab').classList.add('active');
}

function startDetection() {
    if (isDetecting) return;
    
    if (!currentFile && !currentSampleName) {
        showError('请先上传音频文件或选择示例音频');
        return;
    }
    
    isDetecting = true;
    const btn = document.getElementById('detectBtn');
    const btnText = document.getElementById('btnText');
    
    btn.disabled = true;
    btnText.innerHTML = '<span class="loading"></span> 检测中...';
    
    document.getElementById('resultStatus').className = 'result-status idle';
    document.getElementById('resultStatus').textContent = '分析中...';
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('infoSection').style.display = 'none';
    
    const formData = new FormData();
    formData.append('model', document.getElementById('modelSelect').value);
    
    if (currentFile) {
        formData.append('audio', currentFile);
    } else {
        formData.append('sample_type', currentSampleType);
        formData.append('sample_name', currentSampleName);
    }
    
    fetch('/api/detect', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json()
                .then(err => { throw new Error(err.error || '请求失败 (' + response.status + ')'); })
                .catch(() => { throw new Error('请求失败 (' + response.status + ')'); });
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            showResult(data);
        } else {
            showError(data.error || '检测失败');
        }
    })
    .catch(error => {
        console.error('Detection error:', error);
        // Check if it's a network error or server error
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            showError('网络连接失败: 服务器无响应，请检查服务器是否运行');
        } else {
            showError(error.message);
        }
    })
    .finally(() => {
        isDetecting = false;
        btn.disabled = false;
        btnText.textContent = '开始检测';
    });
}

function showError(message) {
    const statusEl = document.getElementById('resultStatus');
    statusEl.className = 'result-status fake';
    statusEl.textContent = '检测失败';
    
    // Show error in progress section
    const progressSection = document.getElementById('progressSection');
    progressSection.style.display = 'block';
    progressSection.innerHTML = '<div style="color: #e53e3e; font-size: 0.9rem; line-height: 1.6; padding: 0.5rem;">' + 
                                 '<strong style="color: #c53030;">错误信息：</strong><br>' + message + 
                                 '<br><br><span style="color: #718096; font-size: 0.85rem;">请检查音频格式是否支持（推荐 WAV/FLAC），或尝试使用示例音频。</span>' +
                                 '</div>';
}

function showResult(data) {
    const statusEl = document.getElementById('resultStatus');
    const progressSection = document.getElementById('progressSection');
    const infoSection = document.getElementById('infoSection');
    
    // Update status
    if (data.prediction === 'bonafide') {
        statusEl.className = 'result-status real';
        statusEl.textContent = '✓ 真实语音';
    } else {
        statusEl.className = 'result-status fake';
        statusEl.textContent = '✗ 虚假语音';
    }
    
    // Show progress bars
    progressSection.style.display = 'block';
    
    const realPercent = (data.bonafide_probability * 100).toFixed(1);
    const fakePercent = (data.spoof_probability * 100).toFixed(1);
    
    document.getElementById('realPercent').textContent = realPercent + '%';
    document.getElementById('fakePercent').textContent = fakePercent + '%';
    
    // Animate progress bars
    setTimeout(() => {
        document.getElementById('realProgress').style.width = realPercent + '%';
        document.getElementById('fakeProgress').style.width = fakePercent + '%';
    }, 100);
    
    // Show info
    infoSection.style.display = 'block';
    document.getElementById('confidenceValue').textContent = (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('durationValue').textContent = data.duration.toFixed(2) + 's';
    
    // Show visualizations
    if (data.visualizations) {
        if (data.visualizations.waveform) {
            document.getElementById('waveformImage').src = 'data:image/png;base64,' + data.visualizations.waveform;
            document.getElementById('waveformImage').style.display = 'block';
            document.getElementById('waveformPlaceholder').style.display = 'none';
        }
        
        if (data.visualizations.mel_spectrogram) {
            document.getElementById('melImage').src = 'data:image/png;base64,' + data.visualizations.mel_spectrogram;
            document.getElementById('melImage').style.display = 'block';
            document.getElementById('melPlaceholder').style.display = 'none';
        }
        
        if (data.visualizations.f0_contour) {
            document.getElementById('f0Image').src = 'data:image/png;base64,' + data.visualizations.f0_contour;
            document.getElementById('f0Image').style.display = 'block';
            document.getElementById('f0Placeholder').style.display = 'none';
        }
    }
}
