/**
 * Theory Page JavaScript
 * Handles dynamic loading of theoretical content
 */

let currentSection = 'background';

// Load initial section
document.addEventListener('DOMContentLoaded', () => {
    loadSection('background');
});

function loadSection(sectionKey) {
    currentSection = sectionKey;
    
    // Update nav active state
    document.querySelectorAll('.theory-nav a').forEach(link => {
        link.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Show loading
    const contentEl = document.getElementById('theoryContent');
    contentEl.innerHTML = '<div style="text-align: center; padding: 3rem; color: #718096;">加载中...</div>';
    
    // Fetch content
    fetch('/api/theory/' + sectionKey)
        .then(response => response.json())
        .then(data => {
            renderSection(data);
        })
        .catch(error => {
            contentEl.innerHTML = '<div style="color: #e53e3e;">加载失败: ' + error.message + '</div>';
        });
}

function renderSection(section) {
    const contentEl = document.getElementById('theoryContent');
    
    let html = '<div class="theory-section">';
    html += '<h2>' + escapeHtml(section.title) + '</h2>';
    
    section.sections.forEach(sub => {
        html += '<h3>' + escapeHtml(sub.subtitle) + '</h3>';
        html += formatContent(sub.content);
    });
    
    html += '</div>';
    contentEl.innerHTML = html;
}

function formatContent(text) {
    // Convert markdown-like text to HTML
    let html = '';
    const lines = text.split('\n');
    let inList = false;
    let listType = null;
    
    lines.forEach(line => {
        line = line.trim();
        
        if (line === '') {
            if (inList) {
                html += listType === 'ul' ? '</ul>' : '</ol>';
                inList = false;
            }
            return;
        }
        
        // Bold text
        line = line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        
        // Unordered list
        if (line.startsWith('- ') || line.startsWith('* ')) {
            if (!inList || listType !== 'ul') {
                if (inList) html += '</ul>';
                html += '<ul>';
                inList = true;
                listType = 'ul';
            }
            html += '<li>' + escapeHtml(line.substring(2)) + '</li>';
            return;
        }
        
        // Numbered list
        if (/^\d+\.\s/.test(line)) {
            if (!inList || listType !== 'ol') {
                if (inList) html += '</ol>';
                html += '<ol>';
                inList = true;
                listType = 'ol';
            }
            html += '<li>' + escapeHtml(line.replace(/^\d+\.\s/, '')) + '</li>';
            return;
        }
        
        // Regular paragraph
        if (inList) {
            html += listType === 'ul' ? '</ul>' : '</ol>';
            inList = false;
        }
        html += '<p>' + line + '</p>';
    });
    
    if (inList) {
        html += listType === 'ul' ? '</ul>' : '</ol>';
    }
    
    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
