const $ = (id) => document.getElementById(id);
const ACTS = ['Linear', 'ReLU', 'Leaky ReLU', 'Sigmoid', 'Tanh', 'GELU', 'Step'];
const COLORS = {
  input: { stroke: '#315fe7', fill: '#dce8ff', curve: '#254fce' },
  hidden: { stroke: '#783bf1', fill: '#eee6ff', curve: '#6530d6' },
  output: { stroke: '#1d9b4e', fill: '#daf8e4', curve: '#16833e' }
};
const defaultLayers = [
  { id: 1, type: 'input', name: 'Input', count: 50 },
  { id: 2, type: 'hidden', name: 'Hidden 1', count: 25, activation: 'ReLU' },
  { id: 3, type: 'output', name: 'Output', count: 1, activation: 'Sigmoid' }
];
let layers;
try {
  const stored = JSON.parse(localStorage.getItem('nn-sketch-layers'));
  layers = Array.isArray(stored) && stored.length >= 2 && stored[0].type === 'input' && stored.at(-1).type === 'output' ? stored : defaultLayers;
} catch { layers = defaultLayers; }
let nextId = Math.max(...layers.map(l => Number(l.id) || 0)) + 1;
let toastTimer;
const LOSSES = ['MSE', 'MAE', 'Huber', 'Binary CE', 'Categorical CE'];
let loss = { show: true, type: 'MSE' };
try {
  const savedLoss = JSON.parse(localStorage.getItem('nn-sketch-loss'));
  if (savedLoss && typeof savedLoss.show === 'boolean' && LOSSES.includes(savedLoss.type)) loss = savedLoss;
} catch { /* Use the default loss. */ }
let diagramWidth = Number(localStorage.getItem('nn-sketch-width')) || 1280;

function escapeXml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c])); }
function defaultName(layer, index) { return layer.type === 'hidden' ? `Hidden ${layers.filter((l, i) => i <= index && l.type === 'hidden').length}` : layer.type === 'input' ? 'Input' : 'Output'; }
function labelFor(layer, index) { return (typeof layer.name === 'string' && layer.name.trim()) || defaultName(layer, index); }
function minimumWidth() { return Math.max(800, 200 + (layers.length - 1) * 190 + (loss.show ? 230 : 0)); }
function syncWidthControl() {
  const slider = $('graphWidth');
  slider.min = minimumWidth();
  diagramWidth = Math.min(3000, Math.max(minimumWidth(), Math.round(diagramWidth / 10) * 10));
  slider.value = diagramWidth;
  $('graphWidthValue').textContent = `${diagramWidth} px`;
}
function persist() { localStorage.setItem('nn-sketch-layers', JSON.stringify(layers)); localStorage.setItem('nn-sketch-loss', JSON.stringify(loss)); localStorage.setItem('nn-sketch-width', String(diagramWidth)); }
function notify(message) { const el = $('toast'); el.textContent = message; el.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('show'), 2700); }

function renderControls() {
  $('layerCount').textContent = layers.length;
  $('layerList').innerHTML = layers.map((layer, i) => {
    const c = COLORS[layer.type];
    const options = ACTS.map(act => `<option value="${act}" ${act === layer.activation ? 'selected' : ''}>${act}</option>`).join('');
    const activationField = layer.type === 'input' ? '' : `<div class="field"><label for="act-${layer.id}">ACTIVATION</label><select id="act-${layer.id}" data-activation="${layer.id}">${options}</select></div>`;
    return `<div class="layer-card" style="--accent:${c.stroke};--soft:${c.fill}" data-id="${layer.id}">
      <div class="layer-head"><span class="layer-name"><i class="layer-badge" aria-hidden="true"></i><span class="layer-name-text">${escapeXml(labelFor(layer, i))}</span></span>${layer.type === 'hidden' ? `<button class="remove-layer" type="button" data-remove="${layer.id}" title="Remove layer" aria-label="Remove layer">×</button>` : ''}</div>
      <div class="field field-name"><label for="name-${layer.id}">NAME</label><input id="name-${layer.id}" type="text" maxlength="24" value="${escapeXml(labelFor(layer, i))}" data-name="${layer.id}"></div>
      <div class="field-row ${layer.type === 'input' ? 'single' : ''}"><div class="field"><label for="count-${layer.id}">NEURONS</label><input id="count-${layer.id}" type="number" min="1" max="9999" step="1" value="${layer.count}" data-count="${layer.id}"></div>${activationField}</div>
    </div>`;
  }).join('');
  $('addLayer').disabled = layers.length >= 8;
  $('addLayer').title = layers.length >= 8 ? 'Maximum of 6 hidden layers' : '';
  $('showLoss').checked = loss.show;
  $('lossFunction').value = loss.type;
  $('lossOptions').hidden = !loss.show;
}

function activationValue(type, x) {
  switch(type) {
    case 'ReLU': return Math.max(0, x) / 2;
    case 'Leaky ReLU': return (x < 0 ? .12*x : x) / 2;
    case 'Sigmoid': return 2/(1+Math.exp(-2*x)) - 1;
    case 'Tanh': return Math.tanh(1.4*x);
    case 'GELU': return (x * .5 * (1 + Math.tanh(Math.sqrt(2/Math.PI) * (x + .044715*x*x*x)))) / 2;
    case 'Step': return x < 0 ? -.72 : .72;
    default: return x / 2;
  }
}
function curvePath(type, x, y) {
  let d = '';
  for (let i = 0; i <= 40; i++) {
    const input = -2 + i/10;
    const v = Math.max(-1, Math.min(1, activationValue(type, input)));
    d += `${i ? 'L' : 'M'}${(x + input*7.2).toFixed(2)} ${(y-v*13.5).toFixed(2)} `;
  }
  return d;
}
function lossCurvePath(type, x, y) {
  let d = '';
  for (let i = 0; i <= 40; i++) {
    const t = i / 40;
    const error = (t - .5) * 2;
    let value;
    if (type === 'MAE') value = Math.abs(error);
    else if (type === 'Huber') { const a = Math.abs(error); value = a <= .4 ? .5*a*a/.4 : a-.2; value /= .8; }
    else if (type === 'Binary CE' || type === 'Categorical CE') value = Math.min(1, -Math.log(.04 + .96*t) / 3.2);
    else value = error * error;
    d += `${i ? 'L' : 'M'}${(x - 25 + t*50).toFixed(2)} ${(y + 15 - value*30).toFixed(2)} `;
  }
  return d;
}
function lossFormula(type) {
  switch (type) {
    case 'MAE': return 'L = |ŷ − y|';
    case 'Huber': return 'L = Hδ(ŷ − y)';
    case 'Binary CE': return 'L = BCE(ŷ, y)';
    case 'Categorical CE': return 'L = −log p(target)';
    default: return 'L = (ŷ − y)²';
  }
}
function visibleCount(count) { return Math.min(8, Math.max(1, count)); }
function positions(count, centerY) {
  const shown = visibleCount(count), gap = 48;
  return Array.from({length: shown}, (_,i) => centerY + (i-(shown-1)/2)*gap);
}
function makeSvg() {
  const width = diagramWidth;
  const height = 600, cy = 267, x0 = 100, x1 = width - (loss.show ? 330 : 100);
  const xs = layers.map((_,i) => x0 + (x1-x0)*i/(layers.length-1));
  let lines = '', nodes = '', labels = '';
  const palettes = layers.map(l => COLORS[l.type]);
  for (let i=0;i<layers.length-1;i++) {
    const left = positions(layers[i].count,cy), right = positions(layers[i+1].count,cy);
    const total = left.length*right.length;
    const opacity = total > 45 ? .28 : .43;
    for (const y1 of left) for (const y2 of right) {
      lines += `<line x1="${xs[i]}" y1="${y1}" x2="${xs[i+1]}" y2="${y2}" stroke="#9bacbf" stroke-width="1.15" opacity="${opacity}"/>`;
    }
  }
  layers.forEach((layer,i) => {
    const ys = positions(layer.count, cy), color = palettes[i], x = xs[i];
    ys.forEach((y,j) => {
      nodes += `<circle cx="${x}" cy="${y}" r="22" fill="${color.fill}" stroke="${color.stroke}" stroke-width="2.8"/>`;
      if (layer.type !== 'input') {
        const clip = `clip-${i}-${j}`;
        nodes += `<defs><clipPath id="${clip}"><circle cx="${x}" cy="${y}" r="16"/></clipPath></defs>`;
        nodes += `<path d="M${x-15} ${y}H${x+15} M${x} ${y-15}V${y+15}" stroke="${color.stroke}" stroke-width=".75" opacity=".25" clip-path="url(#${clip})"/>`;
        nodes += `<path d="${curvePath(layer.activation,x,y)}" fill="none" stroke="${color.curve}" stroke-width="2.25" stroke-linecap="round" stroke-linejoin="round" clip-path="url(#${clip})"/>`;
      }
    });
    if (layer.count > 8) labels += `<text x="${x}" y="${cy+207}" text-anchor="middle" fill="#8d9bb1" font-size="12" font-weight="600">+ ${layer.count-8} more</text>`;
    const titleY = cy+242;
    const name = labelFor(layer,i);
    labels += `<text x="${x}" y="${titleY}" text-anchor="middle" fill="#25314b" font-size="${name.length > 18 ? 12 : 16}" font-weight="700">${escapeXml(name)}</text>`;
    labels += `<text x="${x}" y="${titleY+22}" text-anchor="middle" fill="#697890" font-size="12">${layer.count} ${layer.count === 1 ? 'neuron' : 'neurons'}${layer.type === 'input' ? '' : ` · ${escapeXml(layer.activation || 'Linear')}`}</text>`;
  });
  if (loss.show) {
    const x = width - 110, outputX = xs.at(-1);
    const outputYs = positions(layers.at(-1).count, cy);
    for (const y of outputYs) lines += `<line x1="${outputX}" y1="${y}" x2="${x-42}" y2="${cy}" stroke="#b7a18e" stroke-width="1.3" opacity=".6"/>`;
    nodes += `<line x1="${x}" y1="${cy-95}" x2="${x}" y2="${cy-43}" stroke="#d4935f" stroke-width="1.5" stroke-dasharray="4 4"/>`;
    nodes += `<path d="M${x-4} ${cy-49}L${x} ${cy-43}L${x+4} ${cy-49}" fill="none" stroke="#d4935f" stroke-width="1.5"/>`;
    nodes += `<rect x="${x-42}" y="${cy-42}" width="84" height="84" rx="17" fill="#fff2e3" stroke="#d98235" stroke-width="2.5"/>`;
    nodes += `<path d="M${x-27} ${cy+15}H${x+27} M${x} ${cy-17}V${cy+18}" fill="none" stroke="#d98235" stroke-width=".8" opacity=".35"/>`;
    nodes += `<path d="${lossCurvePath(loss.type,x,cy)}" fill="none" stroke="#bd6b27" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>`;
    labels += `<text x="${x}" y="${cy-109}" text-anchor="middle" fill="#aa754c" font-size="12">target y</text>`;
    labels += `<text x="${(outputX+x)/2}" y="${cy-28}" text-anchor="middle" fill="#a18b78" font-size="11">prediction ŷ</text>`;
    labels += `<text x="${x}" y="${cy+242}" text-anchor="middle" fill="#25314b" font-size="16" font-weight="700">Loss</text>`;
    labels += `<text x="${x}" y="${cy+264}" text-anchor="middle" fill="#697890" font-size="12">${escapeXml(loss.type)} · ${escapeXml(lossFormula(loss.type))}</text>`;
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="Neural network architecture diagram"><title>Neural network architecture</title><rect width="${width}" height="${height}" fill="#ffffff"/>${lines}${nodes}${labels}</svg>`;
}
function renderDiagram() {
  syncWidthControl();
  $('diagram').innerHTML = makeSvg();
}
function render() { renderControls(); renderDiagram(); persist(); }

$('layerList').addEventListener('input', e => {
  const nameId = Number(e.target.dataset.name);
  if (nameId) {
    const layer = layers.find(l => l.id === nameId);
    if (layer) {
      layer.name = e.target.value;
      e.target.closest('.layer-card').querySelector('.layer-name-text').textContent = labelFor(layer, layers.indexOf(layer));
      renderDiagram(); persist();
    }
    return;
  }
  const id = Number(e.target.dataset.count);
  if (!id) return;
  const layer = layers.find(l => l.id === id);
  const n = Number(e.target.value);
  if (layer && Number.isInteger(n) && n >= 1 && n <= 9999) { layer.count = n; renderDiagram(); persist(); }
});
$('layerList').addEventListener('change', e => {
  const id = Number(e.target.dataset.activation || e.target.dataset.count || e.target.dataset.name);
  const layer = layers.find(l => l.id === id);
  if (!layer) return;
  if (e.target.dataset.activation && ACTS.includes(e.target.value)) layer.activation = e.target.value;
  if (e.target.dataset.count) { layer.count = Math.min(9999, Math.max(1, Math.round(Number(e.target.value) || 1))); e.target.value = layer.count; }
  if (e.target.dataset.name) { layer.name = e.target.value.trim() || defaultName(layer, layers.indexOf(layer)); e.target.value = layer.name; e.target.closest('.layer-card').querySelector('.layer-name-text').textContent = layer.name; }
  renderDiagram(); persist();
});
$('layerList').addEventListener('click', e => {
  const id = Number(e.target.dataset.remove);
  if (id) { layers = layers.filter(l => l.id !== id); render(); }
});
$('addLayer').addEventListener('click', () => {
  if (layers.length >= 8) return;
  const hiddenCount = layers.filter(l => l.type === 'hidden').length + 1;
  layers.splice(-1, 0, {id:nextId++, type:'hidden', name:`Hidden ${hiddenCount}`, count:8, activation:'ReLU'});
  render();
});
$('showLoss').addEventListener('change', e => { loss.show = e.target.checked; $('lossOptions').hidden = !loss.show; renderDiagram(); persist(); });
$('lossFunction').addEventListener('change', e => { if (LOSSES.includes(e.target.value)) { loss.type = e.target.value; renderDiagram(); persist(); } });
$('graphWidth').addEventListener('input', e => { diagramWidth = Number(e.target.value); renderDiagram(); persist(); });

function svgBlob() { return new Blob([makeSvg()], {type:'image/svg+xml;charset=utf-8'}); }
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob), a = document.createElement('a');
  a.href = url; a.download = filename; document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
function pngBlob() {
  return new Promise((resolve,reject) => {
    const img = new Image(), url = URL.createObjectURL(svgBlob());
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width*2; canvas.height = img.height*2;
      const ctx = canvas.getContext('2d');
      ctx.scale(2,2); ctx.drawImage(img,0,0);
      canvas.toBlob(blob => { URL.revokeObjectURL(url); blob ? resolve(blob) : reject(new Error('PNG conversion failed')); },'image/png');
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Could not render image')); };
    img.src = url;
  });
}
function closeMenu() { $('downloadMenu').hidden = true; $('downloadBtn').setAttribute('aria-expanded','false'); }
$('downloadBtn').addEventListener('click', () => { const menu = $('downloadMenu'); menu.hidden = !menu.hidden; $('downloadBtn').setAttribute('aria-expanded',String(!menu.hidden)); });
document.addEventListener('click', e => { if (!e.target.closest('.download-wrap')) closeMenu(); });
$('svgBtn').addEventListener('click', () => { downloadBlob(svgBlob(),'neural-network.svg'); closeMenu(); notify('SVG downloaded'); });
$('pngBtn').addEventListener('click', async () => { try { downloadBlob(await pngBlob(),'neural-network.png'); notify('PNG downloaded'); } catch { notify('Could not create PNG'); } closeMenu(); });
$('copyBtn').addEventListener('click', async () => {
  try { await navigator.clipboard.write([new ClipboardItem({'image/png': await pngBlob()})]); notify('Image copied to clipboard'); }
  catch { notify('Clipboard unavailable here. Use Download instead.'); }
});
render();
