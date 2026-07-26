// Home page controller
document.addEventListener('DOMContentLoaded', async () => {
    // Wait for PDF.js to be ready
    if (typeof pdfjsLib !== 'undefined') {
        await PDFViewer.init();
    }

    await loadPassages();
    document.getElementById('passage-upload').addEventListener('change', handlePassageUpload);

    // Mobile nav toggle
    const toggle = document.getElementById('nav-toggle');
    if (toggle) {
        toggle.addEventListener('click', () => {
            document.querySelector('.nav-links').classList.toggle('open');
        });
    }
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

    const url = URL.createObjectURL(file);
    PDFViewer.load(url);

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
