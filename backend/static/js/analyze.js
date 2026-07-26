// Analyze page controller
document.addEventListener('DOMContentLoaded', () => {
    // Init audio player (wavesurfer.js)
    if (typeof WaveSurfer !== 'undefined') {
        AudioPlayer.init();
    }

    // Init recorder
    AudioRecorder.init();

    // File upload
    document.getElementById('audio-upload').addEventListener('change', handleAudioUpload);
    document.getElementById('export-btn').addEventListener('click', () => AnalysisHandler.exportReport());

    // Mobile nav toggle
    const toggle = document.getElementById('nav-toggle');
    if (toggle) {
        toggle.addEventListener('click', () => {
            document.querySelector('.nav-links').classList.toggle('open');
        });
    }
});

function handleAudioUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (typeof AudioPlayer !== 'undefined' && AudioPlayer.ws) {
        AudioPlayer.loadBlob(file);
    }
    AnalysisHandler.analyzeFile(file);
    e.target.value = '';
}
