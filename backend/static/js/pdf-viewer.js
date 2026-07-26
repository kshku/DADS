// PDF.js viewer wrapper
const PDFViewer = {
    pdfDoc: null,
    currentPage: 1,
    totalPages: 0,

    async init() {
        pdfjsLib.GlobalWorkerOptions.workerSrc =
            'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
        document.getElementById('pdf-prev').addEventListener('click', () => this.prevPage());
        document.getElementById('pdf-next').addEventListener('click', () => this.nextPage());
    },

    async load(url) {
        try {
            this.pdfDoc = await pdfjsLib.getDocument(url).promise;
            this.totalPages = this.pdfDoc.numPages;
            this.currentPage = 1;
            document.getElementById('pdf-nav').style.display = 'flex';
            await this.renderPage();
        } catch (e) {
            console.error('PDF load error:', e);
            document.getElementById('pdf-container').innerHTML =
                '<span style="color: #ff5555;">Error loading PDF</span>';
        }
    },

    async renderPage() {
        if (!this.pdfDoc) return;
        const page = await this.pdfDoc.getPage(this.currentPage);
        const container = document.getElementById('pdf-container');
        const scale = Math.min(
            (container.clientWidth - 20) / page.getViewport({ scale: 1 }).width,
            (container.clientHeight - 20) / page.getViewport({ scale: 1 }).height,
            2.0
        );
        const viewport = page.getViewport({ scale });
        const canvas = document.createElement('canvas');
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        container.innerHTML = '';
        container.appendChild(canvas);
        await page.render({ canvasContext: canvas.getContext('2d'), viewport }).promise;
        document.getElementById('pdf-page-info').textContent =
            `Page ${this.currentPage} / ${this.totalPages}`;
    },

    prevPage() {
        if (this.currentPage > 1) {
            this.currentPage--;
            this.renderPage();
        }
    },

    nextPage() {
        if (this.currentPage < this.totalPages) {
            this.currentPage++;
            this.renderPage();
        }
    }
};
