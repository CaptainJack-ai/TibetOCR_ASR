'use strict';

const $ = (s) => document.querySelector(s);
const el = (tag, cls, txt) => { const n = document.createElement(tag); if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; };
const fmtPct = (v) => (v == null ? '—' : (v * 100).toFixed(2) + '%');
const fmtNum = (v) => (v == null ? '—' : Number(v).toLocaleString());

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}
async function jpost(url, body, headers) {
  const r = await fetch(url, {
    method: 'POST',
    headers: Object.assign({ 'Content-Type': 'application/json' }, headers || {}),
    body: typeof body === 'string' || body instanceof Blob ? body : JSON.stringify(body),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}

/* ---------------- tabs ---------------- */
document.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tabpane').forEach((p) => p.classList.toggle('active', p.id === 'tab-' + b.dataset.tab));
}));

/* ---------------- health ---------------- */
jget('/api/health').then((h) => {
  $('#health').textContent = h.torch ? `${h.device.toUpperCase()} · torch ${h.torch}` : '评测结果服务在线';
}).catch((e) => { $('#health').textContent = '离线'; });

/* ---------------- job polling ---------------- */
async function pollJob(jobId, onTick) {
  const t0 = Date.now();
  for (;;) {
    const j = await jget('/api/job/' + jobId);
    const secs = j.elapsed != null ? j.elapsed : (Date.now() - t0) / 1000;
    onTick(j, secs);
    if (j.status === 'done' || j.status === 'error') return j;
    await new Promise((r) => setTimeout(r, 500));
  }
}

/* =======================================================================
   OCR
   ======================================================================= */
const ocr = { id: null, ref: '', view: 'inline', last: null, uploaded: false };

async function loadOcrList() {
  const p = new URLSearchParams();
  const t = $('#ocr-tech').value, s = $('#ocr-script').value, l = $('#ocr-leg').value;
  if (t) p.set('technology', t);
  if (s) p.set('script', s);
  if (l) p.set('legibility', l);
  p.set('limit', '300');
  const d = await jget('/api/samples/ocr?' + p);
  $('#ocr-count').textContent = `${d.total} 页命中（显示前 ${d.items.length}）`;
  const ul = $('#ocr-list');
  ul.innerHTML = '';
  d.items.forEach((it) => {
    const li = el('li');
    li.dataset.id = it.id;
    li.append(el('span', 'mono', it.id));
    li.append(el('span', 'tag', `${it.technology}·${it.script}·${fmtNum(it.n_chars)}字`));
    li.onclick = () => selectOcrSample(it.id);
    ul.append(li);
  });
}

async function selectOcrSample(id) {
  ocr.uploaded = false;
  ocr.id = id;
  document.querySelectorAll('#ocr-list li').forEach((li) => li.classList.toggle('sel', li.dataset.id === id));
  const d = await jget('/api/samples/ocr/' + id);
  $('#ocr-img').src = d.image_url + '?t=' + Date.now();
  $('#ocr-dims').textContent = '';
  $('#ocr-img').onload = () => {
    const im = $('#ocr-img');
    $('#ocr-dims').textContent = `${im.naturalWidth}×${im.naturalHeight}`;
  };
  ocr.ref = d.ref || '';
  const meta = $('#ocr-meta');
  meta.innerHTML = '';
  [['id', d.id], ['technology', d.technology], ['script', d.script], ['script_4', d.script_4],
   ['legibility', d.legibility], ['period', d.period], ['参考字符', d.n_chars]].forEach(([k, v]) => {
    const tr = el('tr'); tr.append(el('th', null, k)); tr.append(el('td', null, String(v ?? '—'))); meta.append(tr);
  });
  resetOcrResult();
  $('#ocr-status').textContent = '已载入样例，点「运行识别」';
}

function resetOcrResult() {
  $('#ocr-metrics').innerHTML = '';
  $('#ocr-diff').innerHTML = '';
  $('#ocr-counts').textContent = '';
  ocr.last = null;
}

function metricCard(k, v, cls) {
  const d = el('div', 'metric');
  d.append(el('div', 'k', k));
  d.append(el('div', 'v' + (cls ? ' ' + cls : ''), v));
  return d;
}

function renderOcrMetrics(r) {
  const m = $('#ocr-metrics');
  m.innerHTML = '';
  if (r.cer != null) {
    m.append(metricCard('CER', fmtPct(r.cer), r.cer <= 0.05 ? 'good' : r.cer > 0.3 ? 'bad' : ''));
    m.append(metricCard('字符正确率', fmtPct(1 - r.cer)));
    m.append(metricCard('替换', fmtNum(r.metrics.replace)));
    m.append(metricCard('漏识别', fmtNum(r.metrics.delete)));
    m.append(metricCard('多识别', fmtNum(r.metrics.insert)));
    m.append(metricCard('参考长度', fmtNum(r.ref_len)));
  }
  m.append(metricCard('预测字符', fmtNum(r.chars ?? r.hyp_len)));
  m.append(metricCard('耗时 s', r.took));
  if (r.took && (r.chars ?? r.hyp_len)) m.append(metricCard('字符/秒', (r.chars / r.took).toFixed(1)));
  if (r.cer != null) {
    $('#ocr-counts').textContent =
      `匹配 ${r.metrics.match} · 替换 ${r.metrics.replace} · 漏识别 ${r.metrics.delete} · 多识别 ${r.metrics.insert} · 参考 ${r.ref_len} 字符`;
  } else {
    $('#ocr-counts').textContent = `未提供标准答案，只显示预测文本（${r.chars} 字符）`;
  }
}

function renderOcrDiff(r) {
  const box = $('#ocr-diff');
  if (!r.diff_inline) { box.textContent = r.text || ''; return; }
  if (ocr.view === 'inline') {
    box.className = 'diff';
    box.innerHTML = r.diff_inline;
  } else {
    box.className = 'diff';
    box.innerHTML = '';
    const wrap = el('div', 'side');
    const a = el('div', 'pane'), b = el('div', 'pane');
    a.append(el('h3', null, '参考 reference'));
    const aDiv = el('div'); aDiv.innerHTML = r.diff_ref; a.append(aDiv);
    b.append(el('h3', null, '预测 hypothesis'));
    const bDiv = el('div'); bDiv.innerHTML = r.diff_hyp; b.append(bDiv);
    wrap.append(a, b); box.append(wrap);
  }
}

async function runOcrSample() {
  if (!ocr.id) { $('#ocr-status').textContent = '先选一个样例'; return; }
  $('#ocr-run').disabled = true;
  resetOcrResult();
  try {
    const { job } = await jpost('/api/sample/ocr', { id: ocr.id, engine: $('#ocr-engine').value });
    const j = await pollJob(job, (jj, s) => {
      $('#ocr-status').textContent = `运行中 ${s.toFixed(1)}s · ${jj.status === 'queued' ? '排队' : '识别中'}`;
    });
    if (j.status === 'error') { $('#ocr-status').textContent = j.error; return; }
    const r = j.result; ocr.last = r;
    $('#ocr-status').textContent = `完成：${r.engine} · ${r.chars} 字符 · ${r.took}s · ${r.device}`;
    renderOcrMetrics(r); renderOcrDiff(r);
  } catch (e) { $('#ocr-status').textContent = '错误：' + e.message; }
  finally { $('#ocr-run').disabled = false; }
}

async function runOcrUpload() {
  const f = $('#ocr-file').files[0];
  if (!f) { $('#ocr-status').textContent = '先选择图片'; return; }
  $('#ocr-upload-run').disabled = true;
  resetOcrResult();
  try {
    const ref = $('#ocr-ref-input').value.trim();
    const { job } = await jpost('/api/infer/ocr', f, {
      'Content-Type': 'application/octet-stream', 'X-Filename': f.name,
      'X-Engine': $('#ocr-engine').value, ...(ref ? { 'X-Ref': encodeURIComponent(ref) } : {}),
    });
    const j = await pollJob(job, (jj, s) => {
      $('#ocr-status').textContent = `运行中 ${s.toFixed(1)}s · 识别中`;
    });
    if (j.status === 'error') { $('#ocr-status').textContent = j.error; return; }
    const r = j.result;
    $('#ocr-img').src = URL.createObjectURL(f);
    $('#ocr-meta').innerHTML = '';
    $('#ocr-status').textContent = `完成：${r.chars} 字符 · ${r.took}s`;
    renderOcrMetrics(r); renderOcrDiff(r);
  } catch (e) { $('#ocr-status').textContent = '错误：' + e.message; }
  finally { $('#ocr-upload-run').disabled = false; }
}

/* =======================================================================
   ASR
   ======================================================================= */
const asr = { dataset: 'nict', id: null, ref: '', last: null };

async function loadAsrList() {
  const ds = $('#asr-dataset').value;
  asr.dataset = ds;
  const d = await jget(`/api/samples/asr?dataset=${ds}&limit=300`);
  const sel = $('#asr-dept');
  const cur = sel.value;
  sel.innerHTML = '<option value="">全部</option>';
  d.depts.forEach((x) => { const o = el('option', null, x); o.value = x; sel.append(o); });
  sel.value = cur;
  await renderAsrItems();
}

async function renderAsrItems() {
  const ds = $('#asr-dataset').value;
  const p = new URLSearchParams({ dataset: ds, limit: '300' });
  if ($('#asr-dept').value) p.set('dept', $('#asr-dept').value);
  const d = await jget('/api/samples/asr?' + p);
  $('#asr-count').textContent = `${d.total} 条命中（显示前 ${d.items.length}）`;
  const ul = $('#asr-list'); ul.innerHTML = '';
  d.items.forEach((it) => {
    const li = el('li'); li.dataset.id = it.id;
    li.append(el('span', 'mono', it.id.slice(0, 18)));
    li.append(el('span', 'tag', `${it.dept || ''}·${Number(it.duration).toFixed(1)}s`));
    li.onclick = () => selectAsrSample(it.id);
    ul.append(li);
  });
}

async function selectAsrSample(id) {
  asr.id = id;
  document.querySelectorAll('#asr-list li').forEach((li) => li.classList.toggle('sel', li.dataset.id === id));
  const d = await jget(`/api/samples/asr/${$('#asr-dataset').value}/${id}`);
  $('#asr-audio').src = d.audio_url;
  asr.ref = d.ref || '';
  $('#asr-dur').textContent = `${Number(d.duration).toFixed(2)} s`;
  const meta = $('#asr-meta'); meta.innerHTML = '';
  [['id', d.id], ['dept', d.dept], ['speaker', d.speaker || '—'], ['duration', d.duration + ' s'],
   ['参考字符', (d.ref || '').length]].forEach(([k, v]) => {
    const tr = el('tr'); tr.append(el('th', null, k)); tr.append(el('td', null, String(v))); meta.append(tr);
  });
  $('#asr-segments').querySelector('tbody').innerHTML = '';
  $('#asr-metrics').innerHTML = ''; $('#asr-diff').innerHTML = '';
  $('#asr-worddiff').innerHTML = ''; $('#asr-counts').textContent = '';
  $('#asr-status').textContent = '已载入样例，点「开始转写」';
}

function renderAsrResult(r, withRef) {
  const m = $('#asr-metrics'); m.innerHTML = '';
  if (withRef) {
    m.append(metricCard('CER', fmtPct(r.cer), r.cer <= 0.1 ? 'good' : r.cer > 0.3 ? 'bad' : ''));
    m.append(metricCard('SWER', fmtPct(r.swer)));
    m.append(metricCard('字符正确率', fmtPct(1 - r.cer)));
    m.append(metricCard('词正确率', fmtPct(1 - r.swer)));
  }
  m.append(metricCard('时长 s', r.duration));
  m.append(metricCard('字符', fmtNum(r.chars)));
  m.append(metricCard('耗时 s', r.took));
  m.append(metricCard('RTF', r.rtf));

  const tb = $('#asr-segments').querySelector('tbody');
  tb.innerHTML = '';
  (r.segments || []).forEach((s, i) => {
    const tr = el('tr');
    tr.append(el('td', 'num', String(i + 1)));
    tr.append(el('td', 'num', s.start.toFixed(2)));
    tr.append(el('td', 'num', s.end.toFixed(2)));
    tr.append(el('td', 'num', String(s.text.length)));
    tr.append(el('td', null, s.text));
    tb.append(tr);
  });

  if (withRef) {
    $('#asr-diff').className = 'diff';
    $('#asr-diff').innerHTML = r.diff_inline;
    $('#asr-worddiff').innerHTML = r.word_diff || '';
    $('#asr-counts').textContent =
      `字符：匹配 ${r.metrics.match} · 替换 ${r.metrics.replace} · 漏 ${r.metrics.delete} · 多 ${r.metrics.insert}` +
      ` ｜ 词：${r.word_metrics.match} 对 / ${r.word_metrics.ref_chars} 词`;
  } else {
    $('#asr-diff').className = 'diff';
    $('#asr-diff').textContent = r.text;
    $('#asr-worddiff').innerHTML = '';
    $('#asr-counts').textContent = '未提供标准答案，只显示转写文本';
  }
}

async function runAsrSample() {
  if (!asr.id) { $('#asr-status').textContent = '先选一个样例'; return; }
  $('#asr-run').disabled = true;
  $('#asr-metrics').innerHTML = ''; $('#asr-diff').innerHTML = '';
  try {
    const { job } = await jpost('/api/sample/asr', {
      dataset: $('#asr-dataset').value, id: asr.id,
      model: $('#asr-model').value, chunk: Number($('#asr-chunk').value) || 0,
    });
    const j = await pollJob(job, (jj, s) => {
      $('#asr-status').textContent = `转写中 ${s.toFixed(1)}s · ${jj.status}`;
    });
    if (j.status === 'error') { $('#asr-status').textContent = j.error; return; }
    const r = j.result; asr.last = r;
    $('#asr-status').textContent = `完成：${r.model} · ${r.chars} 字符 · ${r.took}s · RTF ${r.rtf}`;
    renderAsrResult(r, !!asr.ref);
  } catch (e) { $('#asr-status').textContent = '错误：' + e.message; }
  finally { $('#asr-run').disabled = false; }
}

async function runAsrUpload() {
  const f = $('#asr-file').files[0];
  if (!f) { $('#asr-status').textContent = '先选择音频/视频'; return; }
  $('#asr-upload-run').disabled = true;
  try {
    const ref = $('#asr-ref-input').value.trim();
    const { job } = await jpost('/api/infer/asr', f, {
      'Content-Type': 'application/octet-stream', 'X-Filename': f.name,
      'X-Model': $('#asr-model').value, 'X-Chunk': String(Number($('#asr-chunk').value) || 0),
      ...(ref ? { 'X-Ref': encodeURIComponent(ref) } : {}),
    });
    const j = await pollJob(job, (jj, s) => { $('#asr-status').textContent = `转码/转写中 ${s.toFixed(1)}s`; });
    if (j.status === 'error') { $('#asr-status').textContent = j.error; return; }
    const r = j.result;
    $('#asr-audio').src = URL.createObjectURL(f);
    $('#asr-status').textContent = `完成：${r.chars} 字符 · ${r.took}s · RTF ${r.rtf}`;
    renderAsrResult(r, !!ref);
  } catch (e) { $('#asr-status').textContent = '错误：' + e.message; }
  finally { $('#asr-upload-run').disabled = false; }
}

/* =======================================================================
   BATCH
   ======================================================================= */
function syncBatchControls() {
  const ocrTask = $('#batch-task').value === 'ocr';
  $('#batch-ocr-engine-wrap').hidden = !ocrTask;
  $('#batch-tech-wrap').hidden = !ocrTask;
  $('#batch-asr-model-wrap').hidden = ocrTask;
  $('#batch-dept-wrap').hidden = ocrTask;
}

async function runBatch() {
  const task = $('#batch-task').value;
  $('#batch-run').disabled = true;
  $('#batch-summary').innerHTML = ''; $('#batch-table').querySelector('tbody').innerHTML = '';
  const body = {
    n: Number($('#batch-n').value) || 5,
    seed: Number($('#batch-seed').value) || 42,
  };
  if (task === 'ocr') {
    body.engine = $('#batch-engine').value;
    body.technology = $('#batch-tech').value;
  } else {
    body.dataset = $('#asr-dataset').value;
    body.model = $('#batch-model').value;
    body.dept = $('#batch-dept').value;
  }
  try {
    const { job } = await jpost(`/api/batch/${task}`, body);
    const j = await pollJob(job, (jj, s) => {
      $('#batch-status').textContent = `运行中 ${s.toFixed(1)}s · ${jj.progress || jj.status}`;
    });
    if (j.status === 'error') { $('#batch-status').textContent = j.error; return; }
    const { summary, table } = j.result;
    $('#batch-status').textContent = `完成 ${summary.n} 条`;
    const m = $('#batch-summary');
    m.append(metricCard('mean CER', fmtPct(summary.mean_cer), summary.mean_cer <= 0.05 ? 'good' : summary.mean_cer > 0.3 ? 'bad' : ''));
    m.append(metricCard('median CER', fmtPct(summary.median_cer)));
    m.append(metricCard('p90 CER', fmtPct(summary.p90_cer)));
    m.append(metricCard('灾难率', summary.catastrophic_pct + '%'));
    if (summary.mean_swer != null) m.append(metricCard('mean SWER', fmtPct(summary.mean_swer)));

    const tb = $('#batch-table').querySelector('tbody');
    table.forEach((r) => {
      const tr = el('tr');
      tr.append(el('td', 'num', r.id));
      tr.append(el('td', null, r.technology || r.dept || '—'));
      tr.append(el('td', 'num', fmtPct(r.cer)));
      tr.append(el('td', 'num', r.swer != null ? fmtPct(r.swer) : '—'));
      tr.append(el('td', 'num', String(r.seconds)));
      tr.append(el('td', null, r.pred));
      tb.append(tr);
    });
  } catch (e) { $('#batch-status').textContent = '错误：' + e.message; }
  finally { $('#batch-run').disabled = false; }
}

/* =======================================================================
   wiring
   ======================================================================= */
document.querySelectorAll('[data-src]').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('[data-src]').forEach((x) => x.classList.toggle('active', x === b));
  const sample = b.dataset.src === 'sample';
  $('#ocr-sample-pane').hidden = !sample;
  $('#ocr-upload-pane').hidden = sample;
}));
document.querySelectorAll('[data-asrsrc]').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('[data-asrsrc]').forEach((x) => x.classList.toggle('active', x === b));
  const sample = b.dataset.asrsrc === 'sample';
  $('#asr-sample-pane').hidden = !sample;
  $('#asr-upload-pane').hidden = sample;
}));
document.querySelectorAll('[data-ocrview]').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('[data-ocrview]').forEach((x) => x.classList.toggle('active', x === b));
  ocr.view = b.dataset.ocrview;
  if (ocr.last) renderOcrDiff(ocr.last);
}));

['#ocr-tech', '#ocr-script', '#ocr-leg'].forEach((s) => $(s).addEventListener('change', loadOcrList));
$('#asr-dataset').addEventListener('change', loadAsrList);
$('#asr-dept').addEventListener('change', renderAsrItems);
$('#ocr-run').addEventListener('click', runOcrSample);
$('#ocr-upload-run').addEventListener('click', runOcrUpload);
$('#asr-run').addEventListener('click', runAsrSample);
$('#asr-upload-run').addEventListener('click', runAsrUpload);
$('#batch-run').addEventListener('click', runBatch);
$('#batch-task').addEventListener('change', syncBatchControls);

loadOcrList();
loadAsrList();
syncBatchControls();
