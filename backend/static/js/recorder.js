// MediaRecorder API wrapper
const AudioRecorder = {
    mediaRecorder: null,
    chunks: [],
    isRecording: false,

    init() {
        document.getElementById('record-btn').addEventListener('click', () => this.toggle());
    },

    async toggle() {
        if (this.isRecording) {
            this.stop();
        } else {
            await this.start();
        }
    },

    async start() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
            this.chunks = [];

            this.mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) this.chunks.push(e.data);
            };

            this.mediaRecorder.onstop = () => {
                const blob = new Blob(this.chunks, { type: 'audio/webm' });
                stream.getTracks().forEach(t => t.stop());
                this.isRecording = false;
                document.getElementById('recording-indicator').classList.remove('active');
                document.getElementById('record-btn').textContent = 'Record';

                // Send to analysis
                if (typeof AnalysisHandler !== 'undefined') {
                    AnalysisHandler.analyzeBlob(blob);
                }
            };

            this.mediaRecorder.start();
            this.isRecording = true;
            document.getElementById('recording-indicator').classList.add('active');
            document.getElementById('record-btn').textContent = 'Stop';
        } catch (e) {
            console.error('Microphone access denied:', e);
            alert('Could not access microphone. Please allow microphone permissions.');
        }
    },

    stop() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
        }
    }
};
