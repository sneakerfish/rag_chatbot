#!/usr/bin/env python3
"""
Render benchmarks/results.json (from compare_chunking.py) as a self-contained
HTML report: headline numbers, a per-query judge-score chart, and side-by-side
top chunks and answers for every query.

Usage:
    python benchmarks/build_report.py [--results results.json] [--output report.html]
"""

import argparse
import json
from pathlib import Path

TEMPLATE = r"""<title>Where the Chunks Break</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Familjen+Grotesk:wght@500;600;700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  color-scheme: light;
  --ground: #f4f5f8;
  --surface: #ffffff;
  --surface-2: #eceef3;
  --ink: #171b22;
  --ink-2: #545b69;
  --ink-3: #868d9b;
  --hair: #dcdfe6;
  --fixed: #eb6834;
  --fixed-soft: #fbe6dc;
  --semantic: #2a78d6;
  --semantic-soft: #dbe8f9;
  --mark: #fff1a8;
  --good: #0ca30c;
  --focus: #2a78d6;
  --display: "Familjen Grotesk", "Helvetica Neue", Arial, sans-serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", "SFMono-Regular", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground: #13161b;
    --surface: #1b1f26;
    --surface-2: #242932;
    --ink: #eceef2;
    --ink-2: #b1b8c5;
    --ink-3: #7f8797;
    --hair: #2d323c;
    --fixed: #d95926;
    --fixed-soft: #3a2419;
    --semantic: #3987e5;
    --semantic-soft: #182b45;
    --mark: #4d4310;
    --good: #2fbf2f;
    --focus: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground: #13161b;
  --surface: #1b1f26;
  --surface-2: #242932;
  --ink: #eceef2;
  --ink-2: #b1b8c5;
  --ink-3: #7f8797;
  --hair: #2d323c;
  --fixed: #d95926;
  --fixed-soft: #3a2419;
  --semantic: #3987e5;
  --semantic-soft: #182b45;
  --mark: #4d4310;
  --good: #2fbf2f;
  --focus: #3987e5;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--ground); color: var(--ink);
  font-family: var(--body); font-size: 16px; line-height: 1.5;
  padding-block: 32px 64px; padding-inline: 20px;
}
a { color: var(--semantic); }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.wrap { max-width: 1040px; margin: 0 auto; display: grid; gap: 40px; }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; margin: 0; line-height: 1.15; }
h1 { font-size: clamp(30px, 5vw, 44px); font-weight: 700; letter-spacing: -0.01em; }
h2 { font-size: 24px; font-weight: 600; }
h3 { font-size: 18px; font-weight: 600; }
p { margin: 0; max-width: 68ch; }
.lede { font-size: 19px; color: var(--ink-2); }
.eyebrow { font-family: var(--mono); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); }
header { display: grid; gap: 14px; }
.legend { display: flex; gap: 20px; flex-wrap: wrap; font-size: 14px; color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: 8px; }
.swatch { width: 14px; height: 14px; border-radius: 3px; display: inline-block; }
.swatch.fixed { background: var(--fixed); }
.swatch.semantic { background: var(--semantic); }

/* Share card */
.card { background: var(--surface); border: 1px solid var(--hair); border-radius: 10px; padding: 24px; }
.share { display: grid; gap: 20px; }
.share-head { display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap; align-items: baseline; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 24px; }
.stat { display: grid; gap: 10px; }
.stat .label { font-size: 14px; color: var(--ink-2); }
.pair { display: grid; grid-template-columns: 72px 1fr 64px; gap: 10px; align-items: center; font-family: var(--mono); font-size: 13px; }
.pair .who { color: var(--ink-2); }
.bar { height: 14px; background: var(--surface-2); border-radius: 3px; overflow: hidden; }
.bar i { display: block; height: 100%; border-radius: 3px; }
.pair.fixed .bar i { background: var(--fixed); }
.pair.semantic .bar i { background: var(--semantic); }
.pair .val { text-align: right; font-variant-numeric: tabular-nums; color: var(--ink); font-weight: 500; }
.share-foot { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; font-size: 13px; color: var(--ink-3); }

/* Chart */
.chart { display: grid; gap: 14px; }
.chart svg { width: 100%; height: auto; display: block; font-family: var(--body); }
.chart .row-label { fill: var(--ink-2); font-size: 13px; }
.chart .lang-label { fill: var(--ink); font-family: var(--display); font-weight: 600; font-size: 14px; }
.chart .tick { fill: var(--ink-3); font-size: 12px; }
.chart .grid { stroke: var(--hair); stroke-width: 1; }
.chart .stem { stroke: var(--ink-3); stroke-width: 2; opacity: .55; }
.chart .dot { stroke: var(--surface); stroke-width: 2; }
.chart .dot.fixed { fill: var(--fixed); }
.chart .dot.semantic { fill: var(--semantic); }
.chart .hit { fill: transparent; cursor: default; }
.chart .hit:hover ~ .stem, .chart g:hover .stem { opacity: 1; }
.tip {
  position: fixed; pointer-events: none; z-index: 10; display: none;
  background: var(--surface); color: var(--ink); border: 1px solid var(--hair);
  border-radius: 6px; padding: 8px 10px; font-size: 13px; max-width: 320px;
  box-shadow: 0 6px 20px rgba(0,0,0,.12);
}
.tip b { font-family: var(--mono); font-weight: 500; }

/* Query cards */
section.lang { display: grid; gap: 18px; }
.lang-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; border-bottom: 2px solid var(--ink); padding-bottom: 8px; }
.lang-head .src { font-size: 14px; color: var(--ink-3); }
.q { display: grid; gap: 14px; }
.q-head { display: grid; grid-template-columns: auto 1fr; gap: 12px; align-items: start; }
.level { font-family: var(--mono); font-size: 12px; color: var(--ink-3); padding-top: 4px; white-space: nowrap; }
.q h3 { font-size: 19px; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 720px) { .cols { grid-template-columns: 1fr; } }
.col { background: var(--surface); border: 1px solid var(--hair); border-radius: 10px; padding: 16px; display: grid; gap: 12px; align-content: start; border-top-width: 4px; }
.col.fixed { border-top-color: var(--fixed); }
.col.semantic { border-top-color: var(--semantic); }
.col-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }
.who-lbl { font-family: var(--display); font-weight: 600; font-size: 15px; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; }
.chip { font-family: var(--mono); font-size: 11.5px; padding: 2px 8px; border-radius: 999px; background: var(--surface-2); color: var(--ink-2); white-space: nowrap; }
.chip.ok { color: var(--good); background: color-mix(in srgb, var(--good) 12%, transparent); }
.chip.bad { color: var(--fixed); }
.excerpt { font-family: var(--mono); font-size: 12.5px; line-height: 1.55; color: var(--ink); background: var(--surface-2); border-radius: 6px; padding: 12px; white-space: pre-wrap; overflow-wrap: anywhere; position: relative; }
.excerpt .ragged { color: var(--fixed); font-weight: 500; }
.excerpt mark { background: var(--mark); color: inherit; padding: 0 1px; border-radius: 2px; }
.excerpt .more { color: var(--ink-3); }
.meta { display: flex; gap: 14px; flex-wrap: wrap; font-family: var(--mono); font-size: 12px; color: var(--ink-3); }
.score { display: flex; align-items: center; gap: 10px; }
.pips { display: inline-flex; gap: 3px; }
.pip { width: 12px; height: 12px; border-radius: 50%; background: var(--surface-2); border: 1px solid var(--hair); }
.col.fixed .pip.on { background: var(--fixed); border-color: var(--fixed); }
.col.semantic .pip.on { background: var(--semantic); border-color: var(--semantic); }
.score .num { font-family: var(--mono); font-size: 13px; }
.reason { font-size: 14px; color: var(--ink-2); font-style: italic; }
details { border-top: 1px solid var(--hair); padding-top: 10px; }
summary { cursor: pointer; font-size: 14px; color: var(--ink-2); font-weight: 600; }
summary:hover { color: var(--ink); }
.answer { margin-top: 8px; font-size: 14.5px; white-space: pre-wrap; max-height: 320px; overflow: auto; }

/* Table */
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--hair); vertical-align: top; }
th { font-family: var(--display); font-weight: 600; font-size: 13px; color: var(--ink-2); }
td.num, th.num { text-align: right; font-family: var(--mono); font-size: 13px; }
.method { display: grid; gap: 10px; font-size: 15px; color: var(--ink-2); }
.method ul { margin: 0; padding-left: 20px; display: grid; gap: 4px; }
@media (prefers-reduced-motion: no-preference) { .bar i { transition: width .4s ease; } }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">RAG chunking benchmark</div>
    <h1>Where the Chunks Break</h1>
    <p class="lede">The same manuals, the same embedding model, the same questions. The only difference is where each document was cut into chunks before it went into ChromaDB.</p>
    <div class="legend">
      <span><i class="swatch fixed"></i>Fixed window: every 1,000 characters, 200 overlap</span>
      <span><i class="swatch semantic"></i>Semantic breakpoints: cut where adjacent sentences stop being about the same thing</span>
    </div>
  </header>

  <section class="card share" id="share">
    <div class="share-head">
      <h2 id="share-title"></h2>
      <span class="eyebrow" id="share-sub"></span>
    </div>
    <div class="stats" id="stats"></div>
    <div class="share-foot"><span id="share-models"></span><span id="share-date"></span></div>
  </section>

  <section class="chart card">
    <div>
      <h2>Answer quality per question</h2>
      <p style="color:var(--ink-2);font-size:15px;margin-top:6px">A judge model scored each answer 1 to 5 against a reference written from the same manual, without knowing which chunking produced it. Questions run from simple (1) to complex (5).</p>
    </div>
    <div id="dumbbell"></div>
  </section>

  <div id="langs" style="display:grid;gap:48px"></div>

  <section class="card" style="display:grid;gap:14px">
    <h2>All numbers</h2>
    <div class="tablewrap"><table id="summary"></table></div>
    <div class="tablewrap"><table id="detail"></table></div>
  </section>

  <section class="method card">
    <h2>Method</h2>
    <ul>
      <li>Each manual was ingested twice into ChromaDB with <code>ingest.py</code>: once with <code>--chunking fixed</code> (the original 1,000-character window with 200-character overlap, backing up to the nearest sentence end) and once with <code>--chunking semantic</code> (sentences embedded with their neighbours; a cut wherever the cosine distance between adjacent sentences exceeds the 90th percentile or a numbered heading begins; 200 to 1,500 characters per chunk).</li>
      <li>Both regimes use ChromaDB's default <code>all-MiniLM-L6-v2</code> embeddings for retrieval and take the top 5 chunks as context.</li>
      <li><b>Top-1 has the answer</b> means the single best chunk contains every key term of the expected answer. <b>Clean start</b> means the chunk begins at a sentence or heading rather than mid-sentence.</li>
      <li>Answers were generated by the answer model from the retrieved context with the chatbot's own prompt; the judge model scored them blind against a reference answer.</li>
      <li>Reproduce with <code>benchmarks/compare_chunking.py</code>; this page is built from <code>benchmarks/results.json</code> by <code>benchmarks/build_report.py</code>.</li>
    </ul>
    <p id="credit"></p>
  </section>
</div>
<div class="tip" id="tip"></div>

<script id="data" type="application/json">__DATA__</script>
<script>
const R = JSON.parse(document.getElementById('data').textContent);
const REG = ['fixed', 'semantic'];
const NAME = { fixed: 'Fixed window', semantic: 'Semantic' };
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

const allQ = [];
for (const [key, L] of Object.entries(R.languages)) for (const q of L.queries) allQ.push({ key, lang: L.name, ...q });
const n = allQ.length;
const hasLLM = allQ.some(q => q.regimes.fixed.judge_score != null);

/* ---- share card ---- */
const sum = R.summary;
const stats = [];
if (hasLLM) stats.push({ label: 'Mean answer score (1 to 5, judged blind)', fmt: r => sum[r].mean_judge_score.toFixed(2), frac: r => sum[r].mean_judge_score / 5 });
stats.push({ label: 'Top chunk contains the answer', fmt: r => `${sum[r].hit_at_1} / ${n}`, frac: r => sum[r].hit_at_1 / n });
stats.push({ label: 'Top chunk starts and ends on a clean boundary', fmt: r => `${sum[r].clean_boundaries} / ${n}`, frac: r => sum[r].clean_boundaries / n });
document.getElementById('stats').innerHTML = stats.map(s => `
  <div class="stat"><div class="label">${s.label}</div>
    ${REG.map(r => `<div class="pair ${r}"><span class="who">${r}</span><div class="bar"><i style="width:${(s.frac(r)*100).toFixed(1)}%"></i></div><span class="val">${s.fmt(r)}</span></div>`).join('')}
  </div>`).join('');
const langs = Object.values(R.languages);
document.getElementById('share-title').textContent = hasLLM
  ? `Semantic chunking scored ${sum.semantic.mean_judge_score.toFixed(2)} vs ${sum.fixed.mean_judge_score.toFixed(2)} across ${n} questions`
  : `${n} questions across ${langs.length} manuals`;
document.getElementById('share-sub').textContent = langs.map(l => l.name).join(' · ') + ' manuals';
document.getElementById('share-models').textContent = hasLLM ? `Answers: ${R.answer_model} · Judge: ${R.judge_model} · Ollama, local` : 'Retrieval only';
document.getElementById('share-date').textContent = 'Run ' + R.run_at.slice(0, 10);

/* ---- dumbbell chart ---- */
(function () {
  if (!hasLLM) { document.getElementById('dumbbell').innerHTML = '<p style="color:var(--ink-3)">Run without --skip-llm to get judge scores.</p>'; return; }
  const W = 960, left = 330, right = 40, rowH = 28, groupGap = 22, top = 30;
  let H = top;
  const rows = [];
  for (const L of langs) { H += groupGap; for (const q of L.queries) { rows.push({ L, q, y: H + rowH / 2 }); H += rowH; } }
  H += 20;
  const x = v => left + (v - 1) / 4 * (W - left - right);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Judge score per question, fixed vs semantic">`;
  for (let v = 1; v <= 5; v++) s += `<line class="grid" x1="${x(v)}" x2="${x(v)}" y1="${top}" y2="${H - 14}"/><text class="tick" x="${x(v)}" y="${top - 10}" text-anchor="middle">${v}</text>`;
  let lastL = null;
  for (const { L, q, y } of rows) {
    if (L !== lastL) { s += `<text class="lang-label" x="0" y="${y - rowH / 2 - 6}">${esc(L.name)}</text>`; lastL = L; }
    const f = q.regimes.fixed.judge_score, m = q.regimes.semantic.judge_score;
    const label = q.question.length > 46 ? q.question.slice(0, 45).trimEnd() + '…' : q.question;
    s += `<g data-i="${rows.indexOf(rows.find(r => r.q === q))}">`;
    s += `<text class="row-label" x="${left - 14}" y="${y + 4}" text-anchor="end">${esc(label)}</text>`;
    s += `<text class="tick" x="${left - 14 - 8 * Math.min(label.length, 46) * 0.62 - 20}" y="${y + 4}" text-anchor="end">${q.level}</text>`;
    if (f != null && m != null) s += `<line class="stem" x1="${x(f)}" x2="${x(m)}" y1="${y}" y2="${y}"/>`;
    if (f != null) s += `<circle class="dot fixed" cx="${x(f)}" cy="${y}" r="7"/>`;
    if (m != null) s += `<circle class="dot semantic" cx="${x(m)}" cy="${y}" r="7"/>`;
    s += `<rect class="hit" x="${left}" y="${y - rowH / 2}" width="${W - left - right}" height="${rowH}" data-q="${esc(q.question)}" data-f="${f}" data-m="${m}"/>`;
    s += `</g>`;
  }
  s += `<text class="tick" x="${x(1)}" y="${H - 2}" text-anchor="start">wrong or missing</text><text class="tick" x="${x(5)}" y="${H - 2}" text-anchor="end">complete and correct</text></svg>`;
  const el = document.getElementById('dumbbell'); el.innerHTML = s;
  const tip = document.getElementById('tip');
  el.querySelectorAll('.hit').forEach(h => {
    h.addEventListener('mousemove', e => {
      tip.style.display = 'block';
      tip.innerHTML = `${esc(h.dataset.q)}<br><b>fixed ${h.dataset.f}</b> · <b>semantic ${h.dataset.m}</b>`;
      tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 340) + 'px'; tip.style.top = (e.clientY + 14) + 'px';
    });
    h.addEventListener('mouseleave', () => tip.style.display = 'none');
  });
})();

/* ---- per-query cards ---- */
function highlight(text, terms) {
  let html = esc(text);
  for (const t of terms) {
    const pat = esc(t).split('').map(c => c.trim() ? c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') : '\\s*').join('\\s*');
    html = html.replace(new RegExp(pat, 'gi'), m => `<mark>${m}</mark>`);
  }
  return html;
}
function excerpt(chunk, terms, clean) {
  const LIM = 420;
  const body = chunk.length > LIM ? chunk.slice(0, LIM).trimEnd() : chunk;
  let html = highlight(body, terms);
  if (!clean) {
    const firstWord = body.match(/^\S*/)[0];
    html = `<span class="ragged" title="chunk starts mid-sentence">…${esc(firstWord)}</span>` + highlight(body.slice(firstWord.length), terms);
  }
  if (chunk.length > LIM) html += `<span class="more"> … (${chunk.length - LIM} more chars)</span>`;
  return html;
}
function pips(score) {
  return `<span class="pips">${[1,2,3,4,5].map(i => `<i class="pip ${score >= i ? 'on' : ''}"></i>`).join('')}</span>`;
}
const langsEl = document.getElementById('langs');
for (const L of langs) {
  const sec = document.createElement('section'); sec.className = 'lang';
  sec.innerHTML = `<div class="lang-head"><h2>${esc(L.name)}</h2><span class="src">${esc(L.source)} · ${L.chunk_counts.fixed} fixed chunks, ${L.chunk_counts.semantic} semantic chunks</span></div>`;
  for (const q of L.queries) {
    const d = document.createElement('div'); d.className = 'q';
    d.innerHTML = `<div class="q-head"><span class="level">Level ${q.level} of 5</span><h3>${esc(q.question)}</h3></div>
      <div class="cols">${REG.map(r => {
        const g = q.regimes[r], m = g.metrics, top = g.chunks[0] || { text: '', distance: 0 };
        return `<div class="col ${r}">
          <div class="col-head"><span class="who-lbl">${NAME[r]}</span>
            <div class="chips">
              <span class="chip ${m.hit_at_1 ? 'ok' : ''}">${m.hit_at_1 ? 'top-1 has the answer' : m.hit_at_5 ? 'answer in top-5' : 'answer not retrieved'}</span>
              <span class="chip ${m.top1_clean_start ? 'ok' : 'bad'}">${m.top1_clean_start ? 'clean start' : 'starts mid-sentence'}</span>
            </div></div>
          <div class="excerpt">${excerpt(top.text, q.key_terms, m.top1_clean_start)}</div>
          <div class="meta"><span>top chunk ${m.top1_length} chars</span><span>distance ${m.top1_distance}</span><span>terms ${Math.round(m.top1_term_coverage * 100)}%</span></div>
          ${g.judge_score != null ? `<div class="score">${pips(g.judge_score)}<span class="num">${g.judge_score} / 5</span></div><div class="reason">${esc(g.judge_reason)}</div>` : ''}
          ${g.answer ? `<details><summary>Generated answer (${g.answer_seconds}s)</summary><div class="answer">${esc(g.answer)}</div></details>` : ''}
        </div>`; }).join('')}</div>`;
    sec.appendChild(d);
  }
  langsEl.appendChild(sec);
}

/* ---- tables ---- */
const S = document.getElementById('summary');
S.innerHTML = `<thead><tr><th>Metric</th><th class="num">Fixed window</th><th class="num">Semantic</th></tr></thead><tbody>
  ${hasLLM ? `<tr><td>Mean judge score (1 to 5)</td><td class="num">${sum.fixed.mean_judge_score.toFixed(2)}</td><td class="num">${sum.semantic.mean_judge_score.toFixed(2)}</td></tr>` : ''}
  <tr><td>Top-1 chunk contains all key terms</td><td class="num">${sum.fixed.hit_at_1} / ${n}</td><td class="num">${sum.semantic.hit_at_1} / ${n}</td></tr>
  <tr><td>Any top-5 chunk contains all key terms</td><td class="num">${sum.fixed.hit_at_5} / ${n}</td><td class="num">${sum.semantic.hit_at_5} / ${n}</td></tr>
  <tr><td>Mean key-term coverage of top-1 chunk</td><td class="num">${(sum.fixed.mean_top1_term_coverage*100).toFixed(0)}%</td><td class="num">${(sum.semantic.mean_top1_term_coverage*100).toFixed(0)}%</td></tr>
  <tr><td>Top-1 chunk starts and ends cleanly</td><td class="num">${sum.fixed.clean_boundaries} / ${n}</td><td class="num">${sum.semantic.clean_boundaries} / ${n}</td></tr>
  <tr><td>Mean top-1 cosine distance (lower is closer)</td><td class="num">${sum.fixed.mean_top1_distance.toFixed(3)}</td><td class="num">${sum.semantic.mean_top1_distance.toFixed(3)}</td></tr>
</tbody>`;
const D = document.getElementById('detail');
D.innerHTML = `<thead><tr><th>Question</th><th>Level</th>${hasLLM ? '<th class="num">Fixed score</th><th class="num">Semantic score</th>' : ''}<th class="num">Fixed top-1 hit</th><th class="num">Semantic top-1 hit</th></tr></thead><tbody>
  ${allQ.map(q => `<tr><td>${esc(q.lang)}: ${esc(q.question)}</td><td class="num">${q.level}</td>${hasLLM ? `<td class="num">${q.regimes.fixed.judge_score ?? '–'}</td><td class="num">${q.regimes.semantic.judge_score ?? '–'}</td>` : ''}<td class="num">${q.regimes.fixed.metrics.hit_at_1 ? 'yes' : 'no'}</td><td class="num">${q.regimes.semantic.metrics.hit_at_1 ? 'yes' : 'no'}</td></tr>`).join('')}
</tbody>`;
document.getElementById('credit').innerHTML = R.credit ? esc(R.credit) : '';
</script>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(Path(__file__).with_name("results.json")))
    parser.add_argument("--output", default=str(Path(__file__).with_name("report.html")))
    parser.add_argument("--credit", default=None, help="Optional credit line shown in the Method section")
    args = parser.parse_args()

    data = json.load(open(args.results))
    if args.credit:
        data["credit"] = args.credit
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    Path(args.output).write_text(TEMPLATE.replace("__DATA__", payload))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
