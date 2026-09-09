/**
 * Audio player with wavesurfer.js — play/pause, ±5s seek, scrubber, progress callbacks.
 */
class AudioPlayer {
    constructor() {
        this.wavesurfer = null;
        this.playing = false;
        this.onProgress = null;
    }

    init(containerId) {
        this.wavesurfer = WaveSurfer.create({
            container: containerId,
            waveColor: '#4a6a8a',
            progressColor: '#4fc3f7',
            cursorColor: '#fff',
            height: 64,
            responsive: true,
            barWidth: 2,
            barGap: 1,
        });

        this.wavesurfer.on('play', () => {
            this.playing = true;
            this._updatePlayButton();
        });

        this.wavesurfer.on('pause', () => {
            this.playing = false;
            this._updatePlayButton();
        });

        this.wavesurfer.on('finish', () => {
            this.playing = false;
            this._updatePlayButton();
            this._emitProgress(this.wavesurfer.getDuration(), this.wavesurfer.getDuration());
        });

        this.wavesurfer.on('timeupdate', (currentTime) => {
            this._updateTimeDisplay(currentTime);
            this._emitProgress(currentTime, this.wavesurfer.getDuration());
        });

        this.wavesurfer.on('ready', () => {
            this._updateTimeDisplay(0);
            this._emitProgress(0, this.wavesurfer.getDuration());
        });

        this.wavesurfer.on('decode', () => {
            this._updateTimeDisplay(0);
            this._emitProgress(0, this.wavesurfer.getDuration());
        });
    }

    setOnProgress(cb) {
        this.onProgress = cb;
    }

    loadBlob(blob) {
        if (!this.wavesurfer) return;
        this.wavesurfer.loadBlob(blob);
    }

    togglePlayPause() {
        if (this.wavesurfer) {
            this.wavesurfer.playPause();
        }
    }

    seekBy(seconds) {
        if (!this.wavesurfer) return;
        const current = this.wavesurfer.getCurrentTime() || 0;
        const duration = this.wavesurfer.getDuration() || 0;
        const target = Math.min(Math.max(current + seconds, 0), duration);
        this.wavesurfer.setTime(target);
        this._updateTimeDisplay(target);
        this._emitProgress(target, duration);
    }

    seekToRatio(ratio) {
        if (!this.wavesurfer) return;
        const clamped = Math.min(Math.max(ratio, 0), 1);
        this.wavesurfer.seekTo(clamped);
        this._emitProgress(this.wavesurfer.getCurrentTime(), this.wavesurfer.getDuration());
    }

    _emitProgress(currentTime, duration) {
        if (this.onProgress) {
            this.onProgress(currentTime, duration);
        }
    }

    _updatePlayButton() {
        const playIcon = document.getElementById('play-icon');
        const pauseIcon = document.getElementById('pause-icon');
        if (playIcon && pauseIcon) {
            playIcon.classList.toggle('hidden', this.playing);
            pauseIcon.classList.toggle('hidden', !this.playing);
        }
    }

    _updateTimeDisplay(currentTime) {
        const display = document.getElementById('time-display');
        if (!display) return;
        const duration = this.wavesurfer ? this.wavesurfer.getDuration() : 0;
        display.textContent = `${this._formatTime(currentTime)} / ${this._formatTime(duration)}`;
    }

    _formatTime(seconds) {
        if (!seconds || isNaN(seconds)) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
}

window.AudioPlayer = AudioPlayer;
