'use strict';

const report = {catalog: null, selected: null, sample: null, sampleVersion: null, request: 0};
const stateNames = {complete: '已完成', running: '推理中', queued: '待开始', needs_attention: '需处理', completed_with_errors: '部分接口失败'};
const statusNames = {ok: '已返回', empty: '空识别', refused: '拒识', truncated: '输出截断', error: '接口失败', missing: '待推理'};
const fixed = (value, digits = 2) => value == null ? '—' : Number(value).toFixed(digits);

function reportCell(tr, value, tag = 'td') {
  tr.append(el(tag, null, String(value ?? '—')));
}

function renderReport() {
  const data = report.catalog;
  const dataset = $('#report-dataset').value;
  const table = $('#report-table');
  table.replaceChildren();
  const header = el('tr');
  const isOcr = dataset === 'ocr';
  const metricColumns = isOcr ? ['CER ↓', '中位 CER', '灾难页 >50%', '秒 / 页', '评测进程峰值 GB'] : ['CER ↓', 'SWER ↓', '≤150 字 CER', '≤150 字 SWER', 'RTF', '模型峰值 GB', '评测进程峰值 GB'];
  ['模型', '状态 / 有效返回', '接口失败', ...metricColumns].forEach(x => reportCell(header, x, 'th'));
  table.append(header);
  const runs = data.runs.filter(x => x.dataset === dataset);
  const baselines = data.baselines.filter(x => x.dataset === dataset);
  for (const row of [...runs, ...baselines]) {
    const tr = el('tr', row.baseline ? 'baseline' : 'selectable');
    const metrics = row.baseline ? row : row.metrics;
    reportCell(tr, row.model);
    reportCell(tr, row.baseline ? `原报告 · ${row.n} 条` : `${stateNames[row.state] || row.state} · ${row.attempted - row.errors}/${row.n}${row.empty ? ' · 空识别 ' + row.empty : ''}${row.refused ? ' · 拒识 ' + row.refused : ''}${row.truncated ? ' · 截断 ' + row.truncated : ''}`);
    reportCell(tr, row.baseline ? '—' : row.errors);
    reportCell(tr, fmtPct(metrics.cer));
    if (isOcr) {
      reportCell(tr, fmtPct(metrics.median_cer));
      reportCell(tr, metrics.catastrophic_pct == null ? '—' : fixed(metrics.catastrophic_pct) + '%');
      reportCell(tr, fixed(metrics.mean_seconds));
    } else {
      reportCell(tr, fmtPct(metrics.swer));
      reportCell(tr, fmtPct(metrics.short_cer));
      reportCell(tr, fmtPct(metrics.short_swer));
      reportCell(tr, fixed(metrics.rtf, 3));
      reportCell(tr, fixed(metrics.remote_peak_memory_gb, 3));
    }
    reportCell(tr, fixed(metrics.client_peak_rss_gb, 3));
    if (!row.baseline) {
      tr.title = row.error || row.note || '查看逐条结果';
      if (row.id === report.selected) tr.classList.add('selected');
      tr.onclick = () => selectReport(row.id);
    }
    table.append(tr);
  }
  const chart = $('#report-chart');
  chart.replaceChildren();
  for (const row of [...baselines, ...runs.filter(x => x.comparable)]) {
    const value = row.baseline ? row.cer : row.metrics.cer;
    if (value == null) continue;
    const line = el('div', 'report-bar');
    line.append(el('span', 'report-bar-name', row.model));
    const track = el('div', 'report-track');
    const fill = el('div', row.baseline ? 'report-fill baseline-fill' : 'report-fill');
    fill.style.width = `${Math.max(value * 100, 0.2)}%`;
    track.append(fill);
    line.append(track, el('span', 'mono', fmtPct(value)));
    chart.append(line);
  }
  $('#report-note').textContent = data.metrics_note + ' 选择模型行可查看逐条结果。完整评测指标在全部样本返回后显示。';
  $('#report-timing').textContent = data.timing_note;
  $('#report-updated').textContent = '更新于 ' + new Date(data.updated_utc).toLocaleString();
}

async function loadReport() {
  try {
    report.catalog = await jget('/api/results');
    if (report.selected && !report.catalog.runs.some(row => row.id === report.selected)) {
      report.selected = report.sample = report.sampleVersion = null;
      report.request++;
      $('#report-details').hidden = true;
    }
    renderReport();
    $('#report-error').textContent = '';
    if (report.selected) await selectReport(report.selected);
  } catch (error) {
    $('#report-error').textContent = error.message;
  }
}

async function selectReport(id) {
  const request = ++report.request;
  try {
    const data = await jget('/api/results/' + id);
    if (request !== report.request) return;
    if (report.selected !== id) {
      report.sample = report.sampleVersion = null;
      $('#report-sample').replaceChildren(el('p', 'muted', '选择样本查看原始数据、识别文本和逐字差异。'));
    }
    report.selected = id;
    renderReport();
    $('#report-details').hidden = false;
    $('#report-title').textContent = data.run.model + ' · 逐条结果';
    $('#report-model-note').textContent = [data.run.note, data.run.error].filter(Boolean).join(' ');
    $('#report-download').href = '/api/results/' + id + '/export.csv';
    $('#report-download').hidden = !data.run.has_details;
    const groups = (data.groups || []).filter(row => row.value);
    $('#report-groups').hidden = groups.length === 0;
    const groupTable = $('#report-group-table');
    groupTable.replaceChildren();
    const groupHeader = el('tr');
    ['分组字段', '分组', '样本数', 'CER', 'SWER'].forEach(x => reportCell(groupHeader, x, 'th'));
    groupTable.append(groupHeader);
    const fieldNames = {technology: '材料', script: '文字类型', dept: '领域', speaker: '说话人'};
    for (const row of groups) {
      const tr = el('tr');
      [fieldNames[row.field] || row.field, row.value, row.n, fmtPct(row.cer ?? row.mean_cer), fmtPct(row.swer)].forEach(x => reportCell(tr, x));
      groupTable.append(tr);
    }
    const table = $('#report-items');
    table.replaceChildren();
    const header = el('tr');
    ['样本', '状态', 'CER', 'SWER', '秒'].forEach(x => reportCell(header, x, 'th'));
    table.append(header);
    for (const row of data.items) {
      const tr = el('tr', 'selectable');
      if (row.id === report.sample) tr.classList.add('selected');
      [row.id, statusNames[row.status] || row.status, fmtPct(row.cer), fmtPct(row.swer), fixed(row.seconds)].forEach(x => reportCell(tr, x));
      tr.onclick = () => selectReportSample(id, row.id);
      table.append(tr);
    }
    const selected = data.items.find(row => row.id === report.sample);
    if (selected && JSON.stringify([selected.status, selected.cer, selected.seconds]) !== report.sampleVersion) {
      await selectReportSample(id, selected.id);
    }
  } catch (error) {
    $('#report-error').textContent = error.message;
  }
}

async function selectReportSample(runId, sampleId) {
  report.sample = sampleId;
  const box = $('#report-sample');
  box.replaceChildren(el('p', 'muted', '正在载入样本…'));
  try {
    const data = await jget(`/api/results/${runId}/${sampleId}`);
    if (report.selected !== runId || report.sample !== sampleId) return;
    report.sampleVersion = JSON.stringify([data.sample.status, data.sample.cer, data.sample.seconds]);
    box.replaceChildren(el('h2', null, sampleId + ' · ' + (statusNames[data.sample.status] || data.sample.status)));
    if (data.sample.error) box.append(el('p', 'status', data.sample.error));
    const media = el(data.run.task === 'ocr' ? 'img' : 'audio', 'report-media');
    media.src = data.media_url;
    if (data.run.task === 'asr') media.controls = true;
    else media.alt = '评测原图 ' + sampleId;
    box.append(media);
    for (const [title, value] of [['参考文本', data.sample.reference], ['识别文本', data.sample.transcription]]) {
      box.append(el('h2', null, title), el('pre', 'report-text', value || '（空）'));
    }
    if (data.diff) {
      box.append(el('h2', null, '逐字差异 · 替换红 / 漏识别灰 / 多识别黄'));
      const diff = el('div', 'diff');
      diff.innerHTML = data.diff.diff_inline;
      box.append(diff);
    }
  } catch (error) {
    $('#report-error').textContent = error.message;
  }
}

$('#report-dataset').onchange = () => {
  report.selected = null;
  report.sample = report.sampleVersion = null;
  report.request++;
  $('#report-details').hidden = true;
  renderReport();
};
$('#report-refresh').onclick = loadReport;
loadReport();
setInterval(loadReport, 20000);
