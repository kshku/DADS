// SSE-based analysis handler — no server-side storage, client generates report
const AnalysisHandler = {
    lastSummary: null,
    lastFilename: null,

    async analyzeBlob(blob) {
        const formData = new FormData();
        formData.append('file', blob, 'recording.webm');
        await this.sendForAnalysis(formData);
    },

    async analyzeFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        await this.sendForAnalysis(formData);
    },

    async sendForAnalysis(formData) {
        const statusEl = document.getElementById('detection-status');
        statusEl.style.display = 'block';
        statusEl.className = 'detection-status';
        statusEl.textContent = 'Analyzing...';

        ['prolongation', 'block', 'soundrep', 'wordrep', 'interjection'].forEach(type => {
            const el = document.getElementById(`class-${type}`);
            el.querySelector('.value').textContent = 'Analyzing...';
            el.querySelector('.count').textContent = '[0]';
            el.classList.remove('detected');
        });

        try {
            const response = await fetch('/api/analysis/analyze', {
                method: 'POST',
                body: formData,
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                let eventType = null;
                for (const line of lines) {
                    if (line.startsWith('event: ')) {
                        eventType = line.slice(7).trim();
                    } else if (line.startsWith('data: ') && eventType) {
                        const data = JSON.parse(line.slice(6));
                        this.handleEvent(eventType, data);
                        eventType = null;
                    }
                }
            }
        } catch (e) {
            statusEl.textContent = 'Error: ' + e.message;
            statusEl.className = 'detection-status error';
        }
    },

    handleEvent(type, data) {
        const statusEl = document.getElementById('detection-status');

        if (type === 'progress') {
            const pct = Math.round((data.chunk / data.total) * 100);
            statusEl.textContent = `Analyzing... ${pct}%`;
            this.updateClasses(data.aggregated);
        } else if (type === 'complete') {
            statusEl.textContent = 'Analysis Complete!';
            statusEl.className = 'detection-status complete';
            this.lastSummary = data.summary;
            this.lastFilename = data.filename;
            document.getElementById('export-btn').disabled = false;
            setTimeout(() => { statusEl.style.display = 'none'; }, 3000);
        } else if (type === 'error') {
            statusEl.textContent = 'Error: ' + data.error;
            statusEl.className = 'detection-status error';
        }
    },

    updateClasses(aggregated) {
        for (const [type, stats] of Object.entries(aggregated)) {
            const el = document.getElementById(`class-${type}`);
            if (!el) continue;

            const confidence = stats.confidence.toFixed(1);
            const detected = stats.detected;
            const count = stats.detected_chunks;

            el.querySelector('.value').textContent = `${confidence}%${detected ? ' ✓' : ''}`;
            el.querySelector('.count').textContent = `[${count}]`;
            el.classList.toggle('detected', detected);
        }
    },

    exportReport() {
        if (!this.lastSummary) return;

        const lines = [];
        lines.push('='.repeat(60));
        lines.push('STUTTER DETECTION REPORT');
        lines.push('='.repeat(60));
        lines.push('');
        lines.push(`Audio File: ${this.lastFilename || 'unknown'}`);
        lines.push('');
        lines.push('DETECTION RESULTS');
        lines.push('-'.repeat(60));

        const sorted = Object.entries(this.lastSummary)
            .sort((a, b) => b[1].percentage - a[1].percentage);

        for (const [type, stats] of sorted) {
            const name = type.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
            const status = stats.percentage > 40 ? 'DETECTED' : 'Not Detected';
            lines.push(`${name.padEnd(25)}: ${stats.percentage.toFixed(1)}%  ${status}  [Count: ${stats.count}]`);
        }

        lines.push('');
        lines.push('='.repeat(60));
        lines.push('END OF REPORT');
        lines.push('='.repeat(60));

        const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        a.download = `stutter_report_${ts}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    }
};
