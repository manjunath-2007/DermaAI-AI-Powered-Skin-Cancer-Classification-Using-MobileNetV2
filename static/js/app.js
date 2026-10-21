'use strict';

const $ = id => document.getElementById(id);

// ── DOM refs ──────────────────────────────────────────────────
const fileInput      = $('fileInput');
const dropzone       = $('dropzone');
const dropPrompt     = $('dropPrompt');
const previewWrap    = $('previewWrap');
const previewImg     = $('previewImg');
const changeBtn      = $('changeBtn');
const removeBtn      = $('removeBtn');
const analyseBtn     = $('analyseBtn');
const analyseBtnText = $('analyseBtnText');
const analyseSpinner = $('analyseSpinner');
const clientError    = $('clientError');
const serverError    = $('serverError');
const serverErrorMsg = $('serverErrorMsg');
const resultsCard    = $('resultsCard');
const modelBadge     = $('modelBadge');
const themeToggle    = $('themeToggle');

const MAX_SIZE = 10 * 1024 * 1024;
const ALLOWED  = ['image/jpeg', 'image/png'];

let currentFile = null;

// ── Theme ─────────────────────────────────────────────────────
const html = document.documentElement;
const saved = localStorage.getItem('theme') || 'dark';
html.setAttribute('data-theme', saved);
updateThemeIcon(saved);

themeToggle.addEventListener('click', () => {
  const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  updateThemeIcon(next);
});

function updateThemeIcon(theme) {
  // swap sun/moon SVG path
  const svg = themeToggle.querySelector('svg');
  if (theme === 'light') {
    svg.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
    themeToggle.setAttribute('aria-label', 'Toggle dark mode');
  } else {
    svg.innerHTML = '<circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>';
    themeToggle.setAttribute('aria-label', 'Toggle light mode');
  }
}

// ── Health check ──────────────────────────────────────────────
(async function checkHealth() {
  try {
    const res  = await fetch('/health', { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    if (data.model_loaded) {
      modelBadge.className = 'status-pill ready';
      modelBadge.innerHTML = '<span class="status-dot"></span>Model ready';
    } else {
      modelBadge.className = 'status-pill error';
      modelBadge.innerHTML = '<span class="status-dot"></span>Model not loaded';
    }
  } catch {
    modelBadge.className = 'status-pill error';
    modelBadge.innerHTML = '<span class="status-dot"></span>Server offline';
  }
})();

// ── File handling ─────────────────────────────────────────────
function setFile(f) {
  if (!f) return;
  if (!ALLOWED.includes(f.type)) {
    showClientError('Only JPG and PNG images are accepted.');
    return;
  }
  if (f.size > MAX_SIZE) {
    showClientError('That file is too large. Maximum size is 10 MB.');
    return;
  }
  currentFile = f;
  hideClientError();
  hideServerError();
  resultsCard.hidden = true;

  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    dropPrompt.hidden  = true;
    previewWrap.hidden = false;
  };
  reader.readAsDataURL(f);
  analyseBtn.disabled = false;
}

fileInput.addEventListener('change', () => setFile(fileInput.files[0]));

dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
dropzone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

changeBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
removeBtn.addEventListener('click', e => { e.stopPropagation(); resetUpload(); });

// ── Analyse ───────────────────────────────────────────────────
analyseBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  setLoading(true);
  hideClientError();
  hideServerError();

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);

  try {
    const fd = new FormData();
    fd.append('file', currentFile);

    const res = await fetch('/predict', { method: 'POST', body: fd, signal: controller.signal });
    clearTimeout(timeout);
    const data = await res.json();

    if (!res.ok || !data.success) {
      throw new Error(data.error || `Server error (${res.status})`);
    }
    showResults(data);
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  } catch (err) {
    clearTimeout(timeout);
    let msg;
    if (err.name === 'AbortError') {
      msg = 'Request timed out after 30 seconds. The server may be busy — try again.';
    } else if (!navigator.onLine || err.message === 'Failed to fetch' || err.message.includes('NetworkError')) {
      msg = "Can't reach the server. Make sure the Flask app is running, then try again.";
    } else {
      msg = err.message;
    }
    showServerError(msg);
  } finally {
    setLoading(false);
  }
});

// ── Results ───────────────────────────────────────────────────
// Descriptions for binary output
const RESULT_INFO = {
  malignant: {
    label: 'Most resembles a malignant lesion',
    desc:  'The model found features associated with malignant skin lesions. This is not a diagnosis — please show it to a dermatologist.',
  },
  benign: {
    label: 'Most resembles a benign lesion',
    desc:  'The model found no strong indicators of malignancy. Keep an eye on any changes in size, shape or colour.',
  },
};

const CLASS_LABELS = {
  'akiec': 'Actinic keratoses',
  'bcc':   'Basal cell carcinoma',
  'bkl':   'Benign keratosis-like lesions',
  'df':    'Dermatofibroma',
  'mel':   'Melanoma',
  'nv':    'Melanocytic nevi',
  'vasc':  'Vascular lesions',
};

function showResults(d) {
  showQualityWarnings(d.quality_warnings);

  // The backend refuses to diagnose images outside the lesion domain.
  if (d.analysable === false) {
    $('resultLabel').textContent = 'This image cannot be analysed';
    $('resultDesc').textContent  = "This doesn't look like a close-up photo of a skin lesion, so no result was produced. Please upload a photo of the skin area that fills the frame, in focus and in good light.";
    $('riskPill').className = 'risk-pill low';
    $('riskPill').textContent = 'No result produced';
    $('binaryBars').hidden = true;
    $('multiClassBars').hidden = true;
    $('confidenceSection').hidden = true;
    resultsCard.hidden = false;
    return;
  }

  // Check if multi-class probabilities are provided
  if (d.probabilities && typeof d.probabilities === 'object') {
    showMultiClassResults(d);
    return;
  }

  // Binary output
  const isMalignant = d.cancer_detected;
  const type  = isMalignant ? 'malignant' : 'benign';
  const info  = RESULT_INFO[type];
  const cancerPct = Math.round(d.cancer_probability * 100);
  const confPct   = Math.round(d.confidence * 100);

  $('resultLabel').textContent = info.label;
  $('resultDesc').textContent  = info.desc;

  const pill = $('riskPill');
  pill.className = `risk-pill ${d.risk_level}`;
  const pillText = {
    high:   'Worth showing a doctor soon',
    medium: 'Keep an eye on it and consider a check-up',
    low:    'Likely low risk, keep an eye on changes',
  };
  pill.textContent = pillText[d.risk_level] || d.risk_level;

  // Show binary bars, hide multi-class
  $('binaryBars').hidden = false;
  $('multiClassBars').hidden = true;
  $('confidenceSection').hidden = false;

  // malignancy bar
  animateBar('confFill', 'confPct', 'confTrack', cancerPct, !isMalignant);

  // confidence bar
  animateBar('confConfFill', 'confConfPct', 'confConfTrack', confPct, false);

  resultsCard.hidden = false;
}

// ── Image-quality warnings (backend: d.quality_warnings) ───────
// Advisory only — the prediction is still shown. No-op when the
// active page has no .quality-warnings container.
function showQualityWarnings(warnings) {
  const boxes = document.querySelectorAll('.quality-warnings');
  if (!boxes.length) return;
  const list = Array.isArray(warnings) ? warnings : [];
  boxes.forEach(box => {
    box.textContent = '';
    if (!list.length) { box.hidden = true; return; }
    const title = document.createElement('div');
    title.className = 'qw-title';
    title.textContent = 'Image quality notice';
    const ul = document.createElement('ul');
    list.forEach(w => {
      const li = document.createElement('li');
      li.textContent = (w && w.message) ? w.message : String(w);
      ul.appendChild(li);
    });
    box.appendChild(title);
    box.appendChild(ul);
    box.hidden = false;
  });
}

function showMultiClassResults(d) {
  const probs = d.probabilities;
  // Sort classes by probability descending
  const sorted = Object.entries(probs)
    .map(([key, val]) => ({ key, val: Number(val) }))
    .sort((a, b) => b.val - a.val);

  // Top class determines main label
  const top = sorted[0];
  const topLabel = CLASS_LABELS[top.key] || top.key;
  const topPct = Math.round(top.val * 100);

  const isMalignant = top.key === 'mel' || top.key === 'bcc' || top.key === 'akiec';
  $('resultLabel').textContent = `Most resembles ${topLabel.toLowerCase()}`;
  $('resultDesc').textContent = isMalignant
    ? 'The model found features associated with malignant skin lesions. This is not a diagnosis — please show it to a dermatologist.'
    : 'The model found no strong indicators of malignancy. Keep an eye on any changes in size, shape or colour.';

  // Risk pill based on top malignant probability
  const malignantProb = (probs.mel || 0) + (probs.bcc || 0) + (probs.akiec || 0);
  let risk = 'low';
  if (malignantProb >= 0.6) risk = 'high';
  else if (malignantProb >= 0.3) risk = 'medium';

  const pill = $('riskPill');
  pill.className = `risk-pill ${risk}`;
  const pillText = {
    high:   'Worth showing a doctor soon',
    medium: 'Keep an eye on it and consider a check-up',
    low:    'Likely low risk, keep an eye on changes',
  };
  pill.textContent = pillText[risk];

  // Show multi-class bars, hide binary
  $('binaryBars').hidden = true;
  $('multiClassBars').hidden = false;
  $('confidenceSection').hidden = true;

  // Build class bars
  const container = $('classBars');
  container.innerHTML = '';
  sorted.forEach(({ key, val }) => {
    const pct = Math.round(val * 100);
    const label = CLASS_LABELS[key] || key;
    const isMal = key === 'mel' || key === 'bcc' || key === 'akiec';
    const row = document.createElement('div');
    row.className = 'class-bar-row';
    row.innerHTML = `
      <div class="class-bar-label">
        <span>${label}</span>
        <strong>${pct}%</strong>
      </div>
      <div class="conf-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}">
        <div class="conf-fill ${isMal ? '' : 'benign'}" style="width: 0%"></div>
      </div>
    `;
    container.appendChild(row);
    // Animate
    const fill = row.querySelector('.conf-fill');
    animateFill(fill, pct);
  });

  resultsCard.hidden = false;
}

function animateBar(fillId, pctId, trackId, pct, isBenign) {
  const fill  = $(fillId);
  const label = $(pctId);
  const track = $(trackId);

  fill.className = 'conf-fill' + (fillId === 'confConfFill' ? ' conf-fill--conf' : (isBenign ? ' benign' : ''));
  fill.style.width = '0%';
  track.setAttribute('aria-valuenow', pct);
  label.textContent = '0%';

  // count-up
  let start = null;
  const duration = 700;
  function step(ts) {
    if (!start) start = ts;
    const progress = Math.min((ts - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(eased * pct);
    label.textContent = current + '%';
    fill.style.width  = current + '%';
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function animateFill(fill, pct) {
  fill.style.width = '0%';
  let start = null;
  const duration = 700;
  function step(ts) {
    if (!start) start = ts;
    const progress = Math.min((ts - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(eased * pct);
    fill.style.width = current + '%';
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Reset ─────────────────────────────────────────────────────
$('resetBtn').addEventListener('click', () => {
  resetUpload();
  resultsCard.hidden = true;
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

$('printBtn').addEventListener('click', () => window.print());

function resetUpload() {
  currentFile = null;
  fileInput.value = '';
  previewWrap.hidden = true;
  dropPrompt.hidden  = false;
  analyseBtn.disabled = true;
  hideClientError();
  hideServerError();
  showQualityWarnings([]);
}

// ── Helpers ───────────────────────────────────────────────────
function setLoading(on) {
  analyseBtn.disabled = on;
  analyseBtnText.textContent = on ? 'Analyzing…' : 'Analyze photo';
  analyseSpinner.hidden = !on;
}

function showClientError(msg) {
  clientError.textContent = msg;
  clientError.hidden = false;
}
function hideClientError() { clientError.hidden = true; }

function showServerError(msg) {
  serverErrorMsg.textContent = msg;
  serverError.hidden = false;
}
function hideServerError() { serverError.hidden = true; }
