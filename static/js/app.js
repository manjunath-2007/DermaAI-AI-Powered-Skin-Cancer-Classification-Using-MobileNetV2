const $ = id => document.getElementById(id);
const input = $('fileInput'), zone = $('dropzone'), btn = $('analyseBtn');
let file = null;

function setFile(f) {
  if (!f) return;
  file = f;
  const reader = new FileReader();
  reader.onload = e => {
    $('previewImg').src = e.target.result;
    $('fileName').textContent = f.name;
    $('prompt').hidden = true;
    $('preview').hidden = false;
  };
  reader.readAsDataURL(f);
  btn.disabled = false;
  $('error').hidden = true;
  $('result').hidden = true;
}

input.addEventListener('change', () => setFile(input.files[0]));
zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
zone.addEventListener('drop', e => {
  e.preventDefault();
  zone.classList.remove('dragover');
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});

btn.addEventListener('click', async () => {
  if (!file) return;
  btn.disabled = true; btn.classList.add('loading'); btn.textContent = 'Analysing';
  $('error').hidden = true;
  try {
    const fd = new FormData();
    fd.append('image', file);
    const res = await fetch('/predict', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Something went wrong.');
    showResult(data);
  } catch (err) {
    $('error').textContent = err.message;
    $('error').hidden = false;
  } finally {
    btn.classList.remove('loading'); btn.textContent = 'Analyse Image'; btn.disabled = false;
  }
});

function showResult(d) {
  const box = $('result');
  box.className = 'result ' + (d.cancer_detected ? 'cancer' : 'clear');
  $('resultLabel').textContent = d.label;
  $('resultNote').textContent = d.cancer_detected
    ? 'Please consult a dermatologist as soon as possible.'
    : 'Keep monitoring the spot for any change in size, shape or colour.';
  $('confText').textContent = d.confidence.toFixed(1) + '%';
  $('barFill').style.width = '0';
  box.hidden = false;
  requestAnimationFrame(() => $('barFill').style.width = d.confidence + '%');
}

$('resetBtn').addEventListener('click', () => {
  file = null; input.value = '';
  $('preview').hidden = true; $('prompt').hidden = false;
  $('result').hidden = true; btn.disabled = true;
});
