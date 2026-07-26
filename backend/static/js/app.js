/**
 * Main application controller — upload, analyze, results, report.
 */
(function () {
    const LABELS = ['prolongation', 'block', 'soundrep', 'wordrep', 'interjection'];
    const LABEL_NAMES = {
        prolongation: 'Prolongation',
        block: 'Block',
        soundrep: 'Sound Repetition',
        wordrep: 'Word Repetition',
        interjection: 'Interjection',
    };

    let selectedFile = null;
    let analysisResults = null;
    let player = null;

    // DOM refs
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const clearFileBtn = document.getElementById('clear-file');
    const controlsSection = document.getElementById('controls-section');
    const analyzeBtn = document.getElementById('analyze-btn');
    const progressSection = document.getElementById('progress-section');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const playerSection = document.getElementById('player-section');
    const playBtn = document.getElementById('play-btn');
    const resultsSection = document.getElementById('results-section');
    const reportSection = document.getElementById('report-section');
    const downloadBtn = document.getElementById('download-btn');

    // --- File Selection ---

    function initDropZone() {
        dropZone.addEventListener('click', (e) => {
            if (e.target === fileInput || e.target.closest('.browse-link')) return;
            fileInput.click();
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                selectFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                selectFile(fileInput.files[0]);
            }
        });

        clearFileBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            clearFile();
        });
    }

    function selectFile(file) {
        selectedFile = file;
        fileName.textContent = file.name;
        fileInfo.classList.remove('hidden');
        dropZone.classList.add('hidden');
        controlsSection.classList.remove('hidden');
        resetResults();
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = '';
        fileInfo.classList.add('hidden');
        dropZone.classList.remove('hidden');
        controlsSection.classList.add('hidden');
        resetResults();
    }

    // --- Analysis ---

    function initAnalyze() {
        analyzeBtn.addEventListener('click', startAnalysis);
    }

    async function startAnalysis() {
        if (!selectedFile) return;

        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
        progressSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        reportSection.classList.add('hidden');
        playerSection.classList.add('hidden');
        progressBar.style.width = '0%';
        progressText.textContent = 'Uploading...';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let eventType = '';

            analysisResults = { chunks: [], summary: null, filename: selectedFile.name };

            function processLine(line) {
                if (line.startsWith('event: ')) {
                    eventType = line.slice(7).trim();
                } else if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    handleSSEEvent(eventType, data);
                    eventType = '';
                }
            }

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    processLine(line);
                }
            }

            if (buffer.trim()) {
                processLine(buffer);
            }
        } catch (err) {
            progressText.textContent = `Error: ${err.message}`;
            progressBar.style.width = '0%';
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze';
        }
    }

    function handleSSEEvent(type, data) {
        switch (type) {
            case 'started':
                progressText.textContent = 'Loading audio...';
                break;
            case 'progress':
                handleProgress(data);
                break;
            case 'complete':
                handleComplete(data);
                break;
            case 'error':
                progressText.textContent = `Error: ${data.error || data.message}`;
                break;
        }
    }

    function handleProgress(data) {
        const pct = Math.round((data.chunk / data.total) * 100);
        progressBar.style.width = `${pct}%`;
        progressText.textContent = `Analyzing chunk ${data.chunk}/${data.total}...`;

        analysisResults.chunks.push(data);
        updateResultsLive(data.aggregated);
    }

    function handleComplete(data) {
        analysisResults.summary = data.summary;
        analysisResults.duration = data.duration;
        analysisResults.totalChunks = data.total_chunks;

        progressBar.style.width = '100%';
        progressText.textContent = `Analysis complete — ${data.duration.toFixed(1)}s, ${data.total_chunks} chunks`;

        resultsSection.classList.remove('hidden');
        reportSection.classList.remove('hidden');

        if (data.session_id) {
            loadSpectrogram(data.session_id);
        }
        loadAudioPlayer();
    }

    // --- Results Display ---

    function updateResultsLive(aggregated) {
        resultsSection.classList.remove('hidden');

        for (const label of LABELS) {
            const conf = aggregated[label];
            const confEl = document.getElementById(`conf-${label}`);
            const statusEl = document.getElementById(`status-${label}`);
            const card = confEl.closest('.result-card');

            confEl.textContent = `${Math.round(conf.confidence)}%`;
            statusEl.textContent = conf.detected ? 'Detected' : 'Not Detected';
            card.classList.toggle('detected', conf.detected);
        }
    }

    function resetResults() {
        analysisResults = null;
        for (const label of LABELS) {
            const confEl = document.getElementById(`conf-${label}`);
            const statusEl = document.getElementById(`status-${label}`);
            const card = confEl.closest('.result-card');
            confEl.textContent = '0%';
            statusEl.textContent = 'Not Detected';
            card.classList.remove('detected');
        }
    }

    // --- Audio Player ---

    function loadAudioPlayer() {
        if (!selectedFile) return;
        playerSection.classList.remove('hidden');
        if (!player) {
            player = new AudioPlayer();
            player.init('#waveform');
        }
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                player.loadBlob(selectedFile);
            });
        });
    }

    function loadSpectrogram(sessionId) {
        const img = document.getElementById('spectrogram-img');
        if (!img) return;
        img.onload = () => img.classList.remove('hidden');
        img.src = `/api/spectrogram/${sessionId}`;
    }

    function initPlayer() {
        playBtn.addEventListener('click', () => {
            if (player) player.togglePlayPause();
        });
    }

    // --- Report Download ---

    function initReport() {
        downloadBtn.addEventListener('click', downloadReport);
    }

    function downloadReport() {
        if (!analysisResults || !analysisResults.summary) return;

        let report = `DADS Stutter Detection Report\n`;
        report += `${'='.repeat(40)}\n`;
        report += `File: ${analysisResults.filename}\n`;
        report += `Duration: ${analysisResults.duration?.toFixed(1) || 'N/A'}s\n`;
        report += `Chunks: ${analysisResults.totalChunks}\n\n`;

        report += `Detection Results\n`;
        report += `${'-'.repeat(40)}\n`;

        for (const label of LABELS) {
            const s = analysisResults.summary[label];
            if (s) {
                const status = s.percentage > 0 ? 'DETECTED' : 'Not Detected';
                report += `${LABEL_NAMES[label]}: ${s.percentage.toFixed(1)}% (${s.count} chunks) — ${status}\n`;
            }
        }

        report += `\n${'='.repeat(40)}\n`;
        report += `Generated by DADS — Stutter Detection\n`;

        const blob = new Blob([report], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${analysisResults.filename || 'report'}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }

    // --- Init ---

    document.addEventListener('DOMContentLoaded', () => {
        initDropZone();
        initAnalyze();
        initPlayer();
        initReport();
    });
})();
