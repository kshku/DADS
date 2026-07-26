// wavesurfer.js wrapper for audio playback + spectrogram/waveform
const AudioPlayer = {
    ws: null,
    spectrogram: null,
    isSpectrogram: true,

    init() {
        this.ws = WaveSurfer.create({
            container: '#waveform',
            waveColor: '#3498db',
            progressColor: '#2980b9',
            cursorColor: '#fff',
            height: 80,
            barWidth: 2,
            barGap: 1,
            responsive: true,
            backend: 'WebAudio',
        });

        this.spectrogram = this.ws.registerPlugin(
            WaveSurfer.Spectrogram.create({
                container: '#waveform',
                labels: true,
                height: 80,
                splitChannels: false,
                colorMap: 'magma',
            })
        );

        this.ws.on('timeupdate', (t) => this.onTimeUpdate(t));
        this.ws.on('finish', () => this.onFinish());
        this.ws.on('ready', () => this.onReady());

        document.getElementById('play-pause').addEventListener('click', () => this.togglePlay());
        document.getElementById('seek-back').addEventListener('click', () => this.seekRelative(-5));
        document.getElementById('seek-forward').addEventListener('click', () => this.seekRelative(5));
        document.getElementById('toggle-spectrogram').addEventListener('click', () => this.setMode(true));
        document.getElementById('toggle-waveform').addEventListener('click', () => this.setMode(false));
    },

    loadBlob(blob) {
        const url = URL.createObjectURL(blob);
        this.ws.load(url);
    },

    loadUrl(url) {
        this.ws.load(url);
    },

    togglePlay() {
        this.ws.playPause();
        document.getElementById('play-pause').textContent =
            this.ws.isPlaying() ? 'Pause' : 'Play';
    },

    seekRelative(seconds) {
        const current = this.ws.getCurrentTime();
        this.ws.seekTo(Math.max(0, (current + seconds) / this.ws.getDuration()));
    },

    setMode(spectrogram) {
        this.isSpectrogram = spectrogram;
        const url = this.ws.getSrc();
        if (url) {
            this.ws.load(url);
        }
    },

    onTimeUpdate(time) {
        const duration = this.ws.getDuration();
        const format = (s) => {
            const m = Math.floor(s / 60);
            const sec = Math.floor(s % 60);
            return `${m}:${sec.toString().padStart(2, '0')}`;
        };
        document.getElementById('time-display').textContent =
            `${format(time)} / ${format(duration)}`;
        document.getElementById('progress-fill').style.width =
            `${(time / duration) * 100}%`;
    },

    onFinish() {
        document.getElementById('play-pause').textContent = 'Play';
        document.getElementById('progress-fill').style.width = '100%';
    },

    onReady() {
        const duration = this.ws.getDuration();
        document.getElementById('time-display').textContent =
            `0:00 / ${Math.floor(duration / 60)}:${(Math.floor(duration) % 60).toString().padStart(2, '0')}`;
    },

    getDuration() {
        return this.ws ? this.ws.getDuration() : 0;
    }
};
