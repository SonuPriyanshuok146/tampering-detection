const dropzone      = document.getElementById('dropzone');
const fileInput     = document.getElementById('fileInput');
const prompt        = document.getElementById('prompt');
const scanFrame      = document.getElementById('scanFrame');
const scanImg        = document.getElementById('scanImg');
const results        = document.getElementById('results');
const badge          = document.getElementById('badge');
const confNum        = document.getElementById('confNum');
const explainBox     = document.getElementById('explainBox');
const errorBox       = document.getElementById('errorBox');
const imgOriginal    = document.getElementById('imgOriginal');
const imgOverlay     = document.getElementById('imgOverlay');
const resetLink      = document.getElementById('resetLink');
const apiBaseInput   = document.getElementById('apiBase');
const statusChip     = document.getElementById('statusChip');
const statusText     = document.getElementById('statusText');
const settingsPanel  = document.getElementById('settingsPanel');
const checkBtn       = document.getElementById('checkBtn');
const barPanel       = document.getElementById('barPanel');
const gaugePanel     = document.getElementById('gaugePanel');

let barChart = null;
let gaugeChart = null;

function showError(msg){
  errorBox.innerHTML = msg;
  errorBox.classList.add('show');
}
function clearError(){
  errorBox.classList.remove('show');
  errorBox.innerHTML = '';
}

function resetAll(){
  dropzone.classList.remove('hidden');
  prompt.style.display = '';
  scanFrame.classList.remove('show');
  results.classList.remove('show');
  fileInput.value = '';
  clearError();
}
resetLink.addEventListener('click', resetAll);

// ---- Collapsible connection settings (hidden by default) ----
function toggleSettings(){
  const isOpen = settingsPanel.classList.toggle('open');
  statusChip.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
}
statusChip.addEventListener('click', toggleSettings);
statusChip.addEventListener('keydown', e=>{
  if(e.key==='Enter' || e.key===' '){ e.preventDefault(); toggleSettings(); }
});
checkBtn.addEventListener('click', pingApi);
apiBaseInput.addEventListener('keydown', e=>{ if(e.key==='Enter') pingApi(); });

// ---- Drag & drop wiring ----
['dragenter','dragover'].forEach(evt=>{
  dropzone.addEventListener(evt, e=>{
    e.preventDefault();
    dropzone.classList.add('drag');
  });
});
['dragleave','drop'].forEach(evt=>{
  dropzone.addEventListener(evt, e=>{
    e.preventDefault();
    dropzone.classList.remove('drag');
  });
});
dropzone.addEventListener('drop', e=>{
  const file = e.dataTransfer.files[0];
  if(file) handleFile(file);
});
dropzone.addEventListener('click', ()=> fileInput.click());
dropzone.addEventListener('keydown', e=>{
  if(e.key==='Enter' || e.key===' '){
    e.preventDefault(); fileInput.click();
  }
});
fileInput.addEventListener('change', ()=>{
  // If the user cancels the file dialog (e.g. presses Escape), no file is
  // chosen and files.length is 0 -- correctly do nothing in that case.
  if(fileInput.files && fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file){
  if(!file.type.startsWith('image/')){
    showError('<b>That file can\'t be used —</b> please upload a JPG or PNG image.');
    return;
  }
  if(file.size > 15*1024*1024){
    showError('<b>That file is too large —</b> please upload an image under 15MB.');
    return;
  }
  clearError();
  results.classList.remove('show');

  const url = URL.createObjectURL(file);

  dropzone.classList.add('hidden');
  scanImg.src = url;
  scanFrame.classList.add('show');

  analyze(file, url);
}

async function analyze(file, previewUrl){
  const base = apiBaseInput.value.trim().replace(/\/$/,'');
  const form = new FormData();
  form.append('file', file);

  let data;
  try{
    const res = await fetch(base + '/predict', { method:'POST', body: form });

    if(!res.ok){
      let detail = '';
      try{ const errJson = await res.json(); detail = errJson.detail || ''; }catch(_){}
      throw new Error(detail || `The detection service returned an error (status ${res.status}).`);
    }
    data = await res.json();
    setStatus('ok');

  }catch(err){
    console.error('Tampering Detector — request failed:', err);
    setStatus('bad');
    scanFrame.classList.remove('show');
    dropzone.classList.remove('hidden');
    showError(
      `<b>Couldn't analyze this image —</b> ${err.message || 'the detection service could not be reached'}.<br>` +
      `Make sure the detection service is running, then click the status label at the top of the page to check ` +
      `or update its address.`
    );
    return;
  }

  // Rendering (images, badge, charts) is handled separately from the network
  // call, so a display-side problem (e.g. a charting library failing to
  // load) is never confused with the API request itself failing.
  try{
    renderResult(data, previewUrl);
  }catch(err){
    console.error('Tampering Detector — failed to render results:', err);
    scanFrame.classList.remove('show');
    showError('<b>Got a result, but couldn\'t display it fully —</b> check the browser console (F12) for details.');
  }
}

function renderResult(data, previewUrl){
  scanFrame.classList.remove('show');

  const isTampered = data.prediction === 'tampered';
  const pct = Math.round(data.confidence * 1000) / 10;

  const authenticPct = data.probabilities
    ? Math.round(data.probabilities.authentic * 1000) / 10
    : (isTampered ? 100 - pct : pct);
  const tamperedPct = data.probabilities
    ? Math.round(data.probabilities.tampered * 1000) / 10
    : (isTampered ? pct : 100 - pct);

  badge.className = 'badge ' + (isTampered ? 'tampered' : 'authentic');
  badge.innerHTML = isTampered
    ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 9v4M12 17h.01M10.3 3.9L2.7 18a2 2 0 001.7 3h15.2a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z"/></svg> Tampered'
    : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M20 6L9 17l-5-5"/></svg> Authentic';
  confNum.innerHTML = pct + '% <span>confidence</span>';

  explainBox.className = 'explain ' + (isTampered ? 'tampered' : 'authentic');
  explainBox.textContent = isTampered
    ? 'The heatmap on the right highlights the region driving this decision — typically an area with compression or pixel-level inconsistencies from splicing or copy-move editing.'
    : 'No significant compression inconsistencies were detected. The heatmap response is diffuse and low-intensity, consistent with an unaltered image.';

  imgOriginal.src = previewUrl;
  imgOverlay.src = data.gradcam_overlay_base64
    ? 'data:image/png;base64,' + data.gradcam_overlay_base64
    : '';

  drawCharts(authenticPct, tamperedPct, pct, isTampered);

  results.classList.add('show');
}

function drawCharts(authenticPct, tamperedPct, confidencePct, isTampered){
  if(typeof Chart === 'undefined'){
    // Charting library didn't load (e.g. blocked by a firewall/ad-blocker).
    // Fall back to plain readouts instead of leaving a blank panel or
    // throwing an error that derails the rest of the results.
    barPanel.innerHTML = `<div class="chart-fallback">Authentic: ${authenticPct}%<br>Tampered: ${tamperedPct}%</div>`;
    gaugePanel.innerHTML = `<div class="chart-fallback">Confidence<br><strong style="font-size:22px;color:${isTampered ? '#FF5A3C' : '#35D6B8'}">${confidencePct}%</strong></div>`;
    return;
  }

  const barCtx = document.getElementById('barChart').getContext('2d');
  if(barChart) barChart.destroy();
  barChart = new Chart(barCtx, {
    type: 'bar',
    data: {
      labels: ['Authentic', 'Tampered'],
      datasets: [{
        data: [authenticPct, tamperedPct],
        backgroundColor: ['#35D6B8', '#FF5A3C'],
        borderRadius: 6,
        maxBarThickness: 60,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: { legend: { display: false }, title: { display: true, text: 'Class probability (%)', color:'#8C99A6', font:{size:11} } },
      scales: {
        x: { min:0, max:100, ticks:{ color:'#8C99A6' }, grid:{ color:'#1C2530' } },
        y: { ticks:{ color:'#E7ECF0' }, grid:{ display:false } },
      }
    }
  });

  const gaugeCtx = document.getElementById('gaugeChart').getContext('2d');
  if(gaugeChart) gaugeChart.destroy();
  const gaugeColor = isTampered ? '#FF5A3C' : '#35D6B8';
  gaugeChart = new Chart(gaugeCtx, {
    type: 'doughnut',
    data: {
      labels: ['Confidence', ''],
      datasets: [{
        data: [confidencePct, 100 - confidencePct],
        backgroundColor: [gaugeColor, '#1C2530'],
        borderWidth: 0,
      }]
    },
    options: {
      responsive: true,
      circumference: 270,
      rotation: 225,
      cutout: '72%',
      plugins: {
        legend: { display:false },
        title: { display:true, text:'Confidence', color:'#8C99A6', font:{size:11} },
        tooltip: { enabled:false },
      }
    },
    plugins: [{
      id: 'centerText',
      afterDraw(chart){
        const {ctx, chartArea:{width, height, top, left}} = chart;
        ctx.save();
        ctx.font = "600 22px 'IBM Plex Mono', monospace";
        ctx.fillStyle = gaugeColor;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(confidencePct + '%', left + width/2, top + height/2);
        ctx.restore();
      }
    }]
  });
}

// ---- Connection status (plain language, checked quietly in the background) ----
function setStatus(state){
  statusChip.className = 'status-chip ' + state;
  statusText.textContent = state === 'ok' ? 'Ready' : state === 'bad' ? 'Not connected' : 'Checking…';
}

async function pingApi(){
  setStatus('checking');
  const base = apiBaseInput.value.trim().replace(/\/$/,'');
  try{
    const res = await fetch(base + '/', { method:'GET' });
    setStatus(res.ok ? 'ok' : 'bad');
  }catch(_){
    setStatus('bad');
  }
}
window.addEventListener('load', pingApi);
