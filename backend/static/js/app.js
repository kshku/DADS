/**
 * Main application controller — upload, record, analyze, results, report, PDF passage viewer.
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

    // Recording state
    let audioContext = null;
    let mediaStreamSource = null;
    let scriptProcessor = null;
    let recordStream = null;
    let recordSamples = [];
    let recordTotalSamples = 0;
    let recordSampleRate = 16000;
    let recordTimerId = null;
    let recordStartTime = 0;
    let isRecording = false;

    // PDF state
    let pdfDoc = null;
    let pdfPageNum = 1;
    let pdfZoom = 1.0;

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
    const backMainBtn = document.getElementById('back-main-btn');
    const statusBanner = document.getElementById('status-banner');
    const tabMain = document.getElementById('tab-main');
    const tabAnalysis = document.getElementById('tab-analysis');
    const viewMain = document.getElementById('view-main');
    const viewAnalysis = document.getElementById('view-analysis');

    const recordBtn = document.getElementById('record-btn');
    const recordTimerEl = document.getElementById('record-timer');
    const recordStatus = document.getElementById('record-status');

    const spectrogramImg = document.getElementById('spectrogram-img');
    const spectrogramWrap = document.getElementById('spectrogram-wrap');
    const playhead = document.getElementById('playhead');
    const waveform = document.getElementById('waveform');
    const specToggle = document.getElementById('spec-toggle');
    const waveToggle = document.getElementById('wave-toggle');
    const back5Btn = document.getElementById('back-5-btn');
    const fwd5Btn = document.getElementById('fwd-5-btn');
    const seekSlider = document.getElementById('seek-slider');

    // --- Status banner & view navigation ---

    function setStatus(message, type) {
        statusBanner.textContent = message;
        statusBanner.classList.remove('error', 'success', 'recording');
        if (type) statusBanner.classList.add(type);
    }

    function showView(name) {
        const isMain = name === 'main';
        viewMain.classList.toggle('hidden', !isMain);
        viewAnalysis.classList.toggle('hidden', isMain);
        tabMain.classList.toggle('active', isMain);
        tabAnalysis.classList.toggle('active', !isMain);
    }

    function initNavigation() {
        tabMain.addEventListener('click', () => showView('main'));
        tabAnalysis.addEventListener('click', () => showView('analysis'));
        backMainBtn.addEventListener('click', () => showView('main'));
    }

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
        setStatus(`Loaded: ${file.name}`, 'success');
        resetResults();
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = '';
        fileInfo.classList.add('hidden');
        dropZone.classList.remove('hidden');
        controlsSection.classList.add('hidden');
        setStatus('Ready');
        resetResults();
    }

    // --- Microphone Recording (Web Audio API -> 16-bit PCM WAV) ---

    function initRecording() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.AudioContext) {
            recordBtn.disabled = true;
            recordStatus.textContent = 'Recording is not supported in this browser.';
            return;
        }
        recordBtn.addEventListener('click', () => {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        });
    }

    async function startRecording() {
        try {
            recordStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (err) {
            setStatus(`Microphone error: ${err.message}`, 'error');
            recordStatus.textContent = 'Could not access the microphone. Check browser permissions.';
            return;
        }

        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        await audioContext.resume();
        recordSampleRate = audioContext.sampleRate;
        mediaStreamSource = audioContext.createMediaStreamSource(recordStream);
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        mediaStreamSource.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        recordSamples = [];
        recordTotalSamples = 0;
        scriptProcessor.onaudioprocess = (e) => {
            const data = e.inputBuffer.getChannelData(0);
            recordSamples.push(new Float32Array(data));
            recordTotalSamples += data.length;
        };

        isRecording = true;
        recordStartTime = Date.now();
        recordBtn.textContent = 'Stop Recording';
        recordBtn.classList.add('recording');
        setStatus('Recording...', 'recording');
        recordStatus.textContent = 'Recording in progress — reading the passage aloud.';

        recordTimerEl.textContent = '0:00';
        recordTimerTick();
    }

    function recordTimerTick() {
        if (!isRecording) return;
        const elapsed = (Date.now() - recordStartTime) / 1000;
        recordTimerEl.textContent = formatTime(elapsed);
        recordTimerId = setTimeout(recordTimerTick, 250);
    }

    function stopRecording() {
        isRecording = false;
        clearTimeout(recordTimerId);

        if (scriptProcessor) {
            scriptProcessor.onaudioprocess = null;
            scriptProcessor.disconnect();
            scriptProcessor = null;
        }
        if (mediaStreamSource) {
            mediaStreamSource.disconnect();
            mediaStreamSource = null;
        }
        if (audioContext) {
            audioContext.close().catch(() => {});
            audioContext = null;
        }
        if (recordStream) {
            recordStream.getTracks().forEach((track) => track.stop());
            recordStream = null;
        }

        recordBtn.textContent = 'Start Recording';
        recordBtn.classList.remove('recording');

        const blob = encodeWav(recordSamples, recordTotalSamples, recordSampleRate);
        recordSamples = [];
        recordTotalSamples = 0;
        const file = new File([blob], 'recording.wav', { type: 'audio/wav' });
        setStatus(`Recording saved: ${file.name}`, 'success');
        recordStatus.textContent = 'Recording saved — click Analyze to run detection.';
        selectFile(file);
    }

    function encodeWav(chunks, totalSamples, sampleRate) {
        const numChannels = 1;
        const bitsPerSample = 16;
        const bytesPerSample = bitsPerSample / 8;
        const dataSize = totalSamples * numChannels * bytesPerSample;
        const buffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(buffer);

        const writeString = (offset, str) => {
            for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
        view.setUint16(32, numChannels * bytesPerSample, true);
        view.setUint16(34, bitsPerSample, true);
        writeString(36, 'data');
        view.setUint32(40, dataSize, true);

        let offset = 44;
        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            for (let j = 0; j < chunk.length; j++) {
                const s = Math.max(-1, Math.min(1, chunk[j]));
                view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
                offset += 2;
            }
        }

        return new Blob([buffer], { type: 'audio/wav' });
    }

    function formatTime(seconds) {
        if (!seconds || isNaN(seconds)) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
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
        setStatus(`Analyzing: ${selectedFile.name}`, 'recording');

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
            setStatus(`Error: ${err.message}`, 'error');
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
                setStatus(`Error: ${data.error || data.message}`, 'error');
                break;
        }
    }

    function handleProgress(data) {
        const pct = Math.round((data.chunk / data.total) * 100);
        progressBar.style.width = `${pct}%`;
        progressText.textContent = `Analyzing chunk ${data.chunk}/${data.total}...`;

        analysisResults.chunks.push(data);
        analysisResults.finalAggregated = data.aggregated;
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
        setStatus(`Analysis complete — ${selectedFile ? selectedFile.name : ''}`, 'success');
        showView('analysis');

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
            const chunksEl = document.getElementById(`chunks-${label}`);
            const card = confEl.closest('.result-card');

            confEl.textContent = `${Math.round(conf.confidence)}%`;
            statusEl.textContent = conf.detected ? 'Detected' : 'Not Detected';
            chunksEl.textContent = `${conf.detected_chunks} chunk${conf.detected_chunks === 1 ? '' : 's'}`;
            card.classList.toggle('detected', conf.detected);
        }
    }

    function resetResults() {
        analysisResults = null;
        for (const label of LABELS) {
            const confEl = document.getElementById(`conf-${label}`);
            const statusEl = document.getElementById(`status-${label}`);
            const chunksEl = document.getElementById(`chunks-${label}`);
            const card = confEl.closest('.result-card');
            confEl.textContent = '0%';
            statusEl.textContent = 'Not Detected';
            chunksEl.textContent = '0 chunks';
            card.classList.remove('detected');
        }
    }

    // --- Audio Player + Graph Views ---

    function loadAudioPlayer() {
        if (!selectedFile) return;
        playerSection.classList.remove('hidden');
        if (!player) {
            player = new AudioPlayer();
            player.init('#waveform');
            player.setOnProgress(updatePlaybackOverlay);
        }
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                player.loadBlob(selectedFile);
            });
        });
    }

    function loadSpectrogram(sessionId) {
        if (!spectrogramImg) return;
        spectrogramImg.onload = () => spectrogramImg.classList.remove('hidden');
        spectrogramImg.src = `/api/spectrogram/${sessionId}`;
    }

    function updatePlaybackOverlay(currentTime, duration) {
        if (!duration || duration <= 0) return;
        const ratio = Math.min(Math.max(currentTime / duration, 0), 1);

        playhead.style.left = `${ratio * 100}%`;

        if (!seekSlider.dataset.seeking) {
            seekSlider.value = Math.round(ratio * 1000);
        }
    }

    function initPlayerControls() {
        playBtn.addEventListener('click', () => {
            if (player) player.togglePlayPause();
        });

        back5Btn.addEventListener('click', () => {
            if (player) player.seekBy(-5);
        });

        fwd5Btn.addEventListener('click', () => {
            if (player) player.seekBy(5);
        });

        seekSlider.addEventListener('pointerdown', () => {
            seekSlider.dataset.seeking = 'true';
        });
        seekSlider.addEventListener('pointerup', () => {
            delete seekSlider.dataset.seeking;
        });
        seekSlider.addEventListener('input', () => {
            if (player) player.seekToRatio(seekSlider.value / 1000);
        });

        specToggle.addEventListener('click', () => {
            specToggle.classList.add('active');
            waveToggle.classList.remove('active');
            spectrogramWrap.classList.remove('hidden');
            waveform.classList.add('hidden');
        });

        waveToggle.addEventListener('click', () => {
            waveToggle.classList.add('active');
            specToggle.classList.remove('active');
            spectrogramWrap.classList.add('hidden');
            waveform.classList.remove('hidden');
        });
    }

    // --- PDF Passage Viewer (pdf.js) ---

    function initPDFViewer() {
        const pdfUrl = '/static/passages/Rainbow_Passage.pdf';

        document.getElementById('pdf-zoom-in').addEventListener('click', () => setPdfZoom(pdfZoom + 0.25));
        document.getElementById('pdf-zoom-out').addEventListener('click', () => setPdfZoom(pdfZoom - 0.25));
        document.getElementById('pdf-prev').addEventListener('click', () => changePdfPage(-1));
        document.getElementById('pdf-next').addEventListener('click', () => changePdfPage(1));

        const pdfInput = document.getElementById('pdf-input');
        pdfInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                file.arrayBuffer().then((buf) => {
                    loadPdf(buf, file.name);
                    setStatus(`Loaded passage: ${file.name}`, 'success');
                });
            }
        });

        loadPdfFromUrl(pdfUrl);
    }

    function loadPdfFromUrl(url) {
        fetch(url)
            .then((res) => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                return res.arrayBuffer();
            })
            .then((buf) => loadPdf(buf, 'Rainbow_Passage.pdf'))
            .catch(() => {
                document.getElementById('pdf-container').textContent = 'Reading passage not available.';
            });
    }

    function loadPdf(data, name) {
        const pdfjsLib = window.pdfjsLib;
        if (!pdfjsLib) return;
        pdfjsLib.GlobalWorkerOptions.workerSrc =
            'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
        pdfjsLib
            .getDocument({ data: data })
            .promise.then((doc) => {
                pdfDoc = doc;
                pdfPageNum = 1;
                pdfZoom = 1.0;
                document.getElementById('pdf-zoom-label').textContent = '100%';
                renderPdfPage();
            })
            .catch((err) => {
                setStatus(`Could not load PDF: ${err.message}`, 'error');
            });
    }

    function renderPdfPage() {
        if (!pdfDoc) return;
        pdfDoc
            .getPage(pdfPageNum)
            .then((page) => {
                const container = document.getElementById('pdf-container');
                const canvas = document.getElementById('pdf-canvas');
                const context = canvas.getContext('2d');
                const viewport = page.getViewport({ scale: pdfZoom });

                canvas.width = Math.min(viewport.width, 800);
                canvas.height = (canvas.width / viewport.width) * viewport.height;

                const scale = canvas.width / viewport.width;
                const renderViewport = page.getViewport({ scale: pdfZoom * scale });

                context.clearRect(0, 0, canvas.width, canvas.height);
                page.render({ canvasContext: context, viewport: renderViewport }).promise.then(() => {
                    container.style.height = 'auto';
                });
                document.getElementById('pdf-page-label').textContent = `${pdfPageNum} / ${pdfDoc.numPages}`;
            });
    }

    function changePdfPage(delta) {
        if (!pdfDoc) return;
        const target = pdfPageNum + delta;
        if (target >= 1 && target <= pdfDoc.numPages) {
            pdfPageNum = target;
            renderPdfPage();
        }
    }

    function setPdfZoom(zoom) {
        if (!pdfDoc) return;
        pdfZoom = Math.min(Math.max(zoom, 0.5), 3);
        document.getElementById('pdf-zoom-label').textContent = `${Math.round(pdfZoom * 100)}%`;
        renderPdfPage();
    }

    // --- Report Download ---

    function initReport() {
        downloadBtn.addEventListener('click', downloadReport);
    }

    function downloadReport() {
        if (!analysisResults || !analysisResults.summary) return;

        const aggregated = analysisResults.finalAggregated;
        const rows = [];
        for (const label of LABELS) {
            const s = analysisResults.summary[label];
            if (!s) continue;
            const conf = aggregated && aggregated[label] ? aggregated[label] : null;
            const confidence = conf ? conf.confidence : s.percentage;
            rows.push({
                name: LABEL_NAMES[label],
                confidence,
                count: s.count,
                detected: conf ? conf.detected : s.percentage > 0,
            });
        }
        rows.sort((a, b) => b.confidence - a.confidence);

        const detectedCount = rows.filter((r) => r.detected).length;
        const timestamp = new Date().toLocaleString();

        let report = `===STUTTER DETECTION REPORT===\n`;
        report += `Report Generated: ${timestamp}\n`;
        report += `Audio File: ${analysisResults.filename}\n`;
        report += `Duration: ${(analysisResults.duration || 0).toFixed(1)}s\n`;
        report += `Total Chunks: ${analysisResults.totalChunks || 0}\n\n`;

        report += `Detection Results\n`;
        report += `${'-'.repeat(40)}\n`;

        for (const row of rows) {
            const flag = row.detected ? '✓ DETECTED' : '○ Not Detected';
            report += `${row.name}: ${row.confidence.toFixed(1)}% ${flag} [Count: ${row.count}]\n`;
        }

        report += `\nSUMMARY\n${'='.repeat(40)}\n`;
        report += `Total Classes: ${rows.length}\n`;
        report += `Detected: ${detectedCount}\n`;
        report += `\nGenerated by DADS — Stutter Detection\n`;

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
        initNavigation();
        initDropZone();
        initRecording();
        initAnalyze();
        initPlayerControls();
        initReport();
        initPDFViewer();
    });
})();
