// Main application controller
document.addEventListener('DOMContentLoaded', async () => {
    await PDFViewer.init();
    AudioPlayer.init();
    AudioRecorder.init();

    await loadPassages();

    document.getElementById('passage-upload').addEventListener('change', handlePassageUpload);
    document.getElementById('audio-upload').addEventListener('change', handleAudioUpload);
    document.getElementById('export-btn').addEventListener('click', () => AnalysisHandler.exportReport());
});

async function loadPassages() {
    try {
        const res = await fetch('/api/passages');
        const data = await res.json();
        const list = document.getElementById('passage-list');
        list.innerHTML = '';

        for (const p of data.passages) {
            const div = document.createElement('div');
            div.className = 'passage-item';
            div.textContent = p.name.replace('.pdf', '');
            div.addEventListener('click', () => {
                document.querySelectorAll('.passage-item').forEach(el => el.classList.remove('active'));
                div.classList.add('active');
                PDFViewer.load(p.url);
            });
            list.appendChild(div);
        }
    } catch (e) {
        console.error('Failed to load passages:', e);
    }
}

function handlePassageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Load directly into browser PDF.js via blob URL — no server storage
    const url = URL.createObjectURL(file);
    PDFViewer.load(url);

    // Also add to passage list visually
    const list = document.getElementById('passage-list');
    const div = document.createElement('div');
    div.className = 'passage-item active';
    div.textContent = file.name.replace('.pdf', '');
    div.addEventListener('click', () => {
        document.querySelectorAll('.passage-item').forEach(el => el.classList.remove('active'));
        div.classList.add('active');
        PDFViewer.load(url);
    });
    list.appendChild(div);

    e.target.value = '';
}

function handleAudioUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    AudioPlayer.loadBlob(file);
    AnalysisHandler.analyzeFile(file);
    e.target.value = '';
}
