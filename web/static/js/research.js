/**
 * Research results view.
 *
 * Renders the ACTIVE, validated run only (plus the supplementary runs that
 * supplement it).  Every number comes from /api/research/summary (the runs' own
 * benchmark_results.json and metadata.json) or from /api/figures (the figure
 * manifest).  Nothing is hard-coded, and live-session values never appear here:
 * this view shows offline evaluation results, a different thing from the live
 * simulation metrics on the Simulation tab.
 *
 * Charts are drawn as inline SVG; hover a mark for its exact value.  The summary
 * is revalidated with an ETag, so a reload costs one 304 when nothing changed.
 */

let researchLoaded = false;

function rEsc(text) {
  return String(text === null || text === undefined ? '' : text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function rNum(x, d = 2) { return (x === null || x === undefined || Number.isNaN(x)) ? 'N/A' : Number(x).toFixed(d); }

/** "mean ± ci" for a summarize_metric object, or N/A. */
function rStat(s, digits = 2) {
  if (!s || s.mean === null || s.mean === undefined) return 'N/A';
  const ci = (s.ci95 === null || s.ci95 === undefined) ? '' : ` ± ${Number(s.ci95).toFixed(digits)}`;
  return `${Number(s.mean).toFixed(digits)}${ci}`;
}

function wilson(k, n) {
  if (!n) return null;
  const z = 1.959964, p = k / n, d = 1 + z * z / n;
  const c = (p + z * z / (2 * n)) / d;
  const h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  return [Math.max(0, c - h), Math.min(1, c + h), p];
}
function rRate(k, n) {
  const w = wilson(k, n);
  return w ? `${w[2].toFixed(3)} [${w[0].toFixed(3)}, ${w[1].toFixed(3)}]` : 'N/A';
}
function title(key) { return String(key).replace(/^experiment_/, '').replace(/_/g, ' '); }

/* ------------------------------------------------------------ SVG helpers */
const KIND_COLOR = {
  reference: 'var(--text-muted)', heuristic: 'var(--warn)', undefended: 'var(--bad)',
  defence: 'var(--accent)', oracle: 'var(--ok)', learned: 'var(--accent)', neutral: 'var(--text-muted)',
};

function svgWrap(w, h, inner, label) {
  return `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${rEsc(label)}" preserveAspectRatio="xMidYMid meet">${inner}</svg>`;
}
function ticks(min, max, n = 5) {
  const out = [];
  for (let i = 0; i <= n; i++) out.push(min + (max - min) * i / n);
  return out;
}

/** Horizontal bars with a 95 % CI whisker and optional per-seed dots. */
function barsCI(rows, o = {}) {
  const W = o.width || 820, labelW = o.labelW || 240, rowH = o.rowH || 25, top = 22, right = 86;
  const min = o.min !== undefined ? o.min : 0, max = o.max !== undefined ? o.max : 100;
  const H = top + rows.length * rowH + 34;
  const x = (v) => labelW + (Math.min(max, Math.max(min, v)) - min) / (max - min) * (W - labelW - right);
  let s = '';
  ticks(min, max).forEach((t) => {
    s += `<line x1="${x(t)}" x2="${x(t)}" y1="${top - 6}" y2="${H - 26}" class="gridl"/>` +
         `<text x="${x(t)}" y="${H - 10}" class="axt" text-anchor="middle">${Math.round(t * 10) / 10}</text>`;
  });
  s += `<text x="${(labelW + W - right) / 2}" y="${H - 0}" class="axl" text-anchor="middle">${rEsc(o.xLabel || '')}</text>`;
  if (o.ref !== undefined) {
    s += `<line x1="${x(o.ref)}" x2="${x(o.ref)}" y1="${top - 6}" y2="${H - 26}" class="refl"/>` +
         `<text x="${x(o.ref) + 3}" y="${top - 8}" class="axt">${rEsc(o.refLabel || '')}</text>`;
  }
  rows.forEach((r, i) => {
    const y = top + i * rowH, cy = y + rowH / 2;
    const col = KIND_COLOR[r.kind] || KIND_COLOR.neutral;
    const tip = `${r.label}: ${rNum(r.mean)}${r.lo !== undefined ? ` [${rNum(r.lo)}, ${rNum(r.hi)}]` : ''}${r.note ? ' — ' + r.note : ''}`;
    s += `<g class="mark"><title>${rEsc(tip)}</title>` +
         `<text x="${labelW - 8}" y="${cy + 4}" class="rowl" text-anchor="end">${rEsc(r.label)}</text>` +
         `<rect x="${x(min)}" y="${y + 5}" width="${Math.max(0, x(r.mean) - x(min))}" height="${rowH - 10}" style="fill:${col}" opacity="0.78"/>`;
    if (r.lo !== undefined && r.hi !== undefined) {
      s += `<line x1="${x(r.lo)}" x2="${x(r.hi)}" y1="${cy}" y2="${cy}" class="whisk"/>` +
           `<line x1="${x(r.lo)}" x2="${x(r.lo)}" y1="${cy - 4}" y2="${cy + 4}" class="whisk"/>` +
           `<line x1="${x(r.hi)}" x2="${x(r.hi)}" y1="${cy - 4}" y2="${cy + 4}" class="whisk"/>`;
    }
    (r.dots || []).forEach((d, k) => {
      if (d === null || d === undefined) return;
      s += `<circle cx="${x(d)}" cy="${cy + ((k % 5) - 2) * 2.2}" r="1.9" class="dot"><title>seed value ${rNum(d)}</title></circle>`;
    });
    s += `<text x="${W - right + 6}" y="${cy + 4}" class="valt">${rNum(r.mean, o.digits !== undefined ? o.digits : 1)}</text></g>`;
  });
  return svgWrap(W, H, s, o.label || 'bar chart with confidence intervals');
}

/** Forest plot of paired differences. */
function forestPlot(rows, o = {}) {
  const W = 820, labelW = 340, rowH = 28, top = 18, right = 190;
  const lo = Math.min(0, ...rows.map((r) => r.lo)) - 1, hi = Math.max(0, ...rows.map((r) => r.hi)) + 1;
  const H = top + rows.length * rowH + 36;
  const x = (v) => labelW + (v - lo) / (hi - lo) * (W - labelW - right);
  let s = '';
  ticks(lo, hi, 6).forEach((t) => {
    s += `<line x1="${x(t)}" x2="${x(t)}" y1="${top - 6}" y2="${H - 28}" class="gridl"/>` +
         `<text x="${x(t)}" y="${H - 12}" class="axt" text-anchor="middle">${Math.round(t)}</text>`;
  });
  s += `<line x1="${x(0)}" x2="${x(0)}" y1="${top - 6}" y2="${H - 28}" class="zerol"/>`;
  s += `<text x="${(labelW + W - right) / 2}" y="${H - 0}" class="axl" text-anchor="middle">${rEsc(o.xLabel || 'percentage points')}</text>`;
  rows.forEach((r, i) => {
    const cy = top + i * rowH + rowH / 2;
    const col = r.verdict === 'SUPPORTED' ? 'var(--ok)' : (String(r.verdict).indexOf('CONTRADICTED') === 0 ? 'var(--bad)' : 'var(--text-muted)');
    const tip = `${r.id}: ${rNum(r.mean)} [${rNum(r.lo)}, ${rNum(r.hi)}] pp, p=${rNum(r.p, 3)}${r.pHolm !== null && r.pHolm !== undefined ? ', Holm ' + rNum(r.pHolm, 3) : ''} — ${r.verdict}`;
    s += `<g class="mark"><title>${rEsc(tip)}</title>` +
         `<text x="${labelW - 8}" y="${cy + 4}" class="rowl" text-anchor="end">${rEsc(r.short)}</text>` +
         `<line x1="${x(r.lo)}" x2="${x(r.hi)}" y1="${cy}" y2="${cy}" style="stroke:${col}" stroke-width="2.2"/>` +
         `<line x1="${x(r.lo)}" x2="${x(r.lo)}" y1="${cy - 5}" y2="${cy + 5}" style="stroke:${col}" stroke-width="2"/>` +
         `<line x1="${x(r.hi)}" x2="${x(r.hi)}" y1="${cy - 5}" y2="${cy + 5}" style="stroke:${col}" stroke-width="2"/>` +
         `<circle cx="${x(r.mean)}" cy="${cy}" r="4.5" style="fill:${col}"/>` +
         `<text x="${W - right + 8}" y="${cy + 4}" class="valt" style="fill:${col}">${rEsc(String(r.verdict).split(' (')[0].toLowerCase())}  p=${rNum(r.p, 3)}</text></g>`;
  });
  return svgWrap(W, H, s, o.label || 'forest plot');
}

/** Lines with CI whiskers; x positions numeric. */
function lineCI(series, o = {}) {
  const W = 760, H = 300, l = 54, r = 24, t = 20, b = 44;
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p.x)))].sort((a, c) => a - c);
  const ys = series.flatMap((s) => s.points.flatMap((p) => [p.lo, p.hi]));
  const ymin = o.ymin !== undefined ? o.ymin : Math.floor(Math.min(...ys) / 10) * 10;
  const ymax = o.ymax !== undefined ? o.ymax : Math.ceil(Math.max(...ys) / 10) * 10;
  const xp = (v) => l + (v - xs[0]) / ((xs[xs.length - 1] - xs[0]) || 1) * (W - l - r);
  const yp = (v) => t + (1 - (v - ymin) / (ymax - ymin)) * (H - t - b);
  let s = '';
  ticks(ymin, ymax, 5).forEach((v) => {
    s += `<line x1="${l}" x2="${W - r}" y1="${yp(v)}" y2="${yp(v)}" class="gridl"/><text x="${l - 6}" y="${yp(v) + 3}" class="axt" text-anchor="end">${Math.round(v)}</text>`;
  });
  xs.forEach((v) => { s += `<text x="${xp(v)}" y="${H - b + 16}" class="axt" text-anchor="middle">${v}</text>`; });
  s += `<text x="${(l + W - r) / 2}" y="${H - 8}" class="axl" text-anchor="middle">${rEsc(o.xLabel || '')}</text>`;
  s += `<text transform="translate(14 ${(t + H - b) / 2}) rotate(-90)" class="axl" text-anchor="middle">${rEsc(o.yLabel || '')}</text>`;
  if (o.ref !== undefined) {
    s += `<line x1="${l}" x2="${W - r}" y1="${yp(o.ref)}" y2="${yp(o.ref)}" class="refl"/><text x="${W - r - 4}" y="${yp(o.ref) - 4}" class="axt" text-anchor="end">${rEsc(o.refLabel || '')}</text>`;
  }
  series.forEach((sr) => {
    const pts = sr.points.slice().sort((a, c) => a.x - c.x);
    s += `<polyline points="${pts.map((p) => xp(p.x) + ',' + yp(p.mean)).join(' ')}" fill="none" style="stroke:${sr.color}" stroke-width="2"/>`;
    pts.forEach((p) => {
      s += `<g class="mark"><title>${rEsc(sr.name)} @ ${p.x}: ${rNum(p.mean)} [${rNum(p.lo)}, ${rNum(p.hi)}] (n=${p.n})</title>` +
           `<line x1="${xp(p.x)}" x2="${xp(p.x)}" y1="${yp(p.lo)}" y2="${yp(p.hi)}" style="stroke:${sr.color}" stroke-width="1.6"/>` +
           `<circle cx="${xp(p.x)}" cy="${yp(p.mean)}" r="4" style="fill:${sr.color}"/></g>`;
    });
  });
  return svgWrap(W, H, s, o.label || 'line chart') + legend(series.map((x) => [x.name, x.color]));
}

/** Plain line chart of one series (e.g. a per-round value). */
function lineSimple(vals, o = {}) {
  const W = 520, H = 220, l = 46, r = 14, t = 14, b = 34;
  const clean = vals.filter((v) => v !== null && v !== undefined);
  if (!clean.length) return '<div class="empty">No data.</div>';
  const ymax = o.ymax !== undefined ? o.ymax : Math.max(...clean) * 1.15 || 1;
  const xp = (i) => l + i / Math.max(1, vals.length - 1) * (W - l - r);
  const yp = (v) => t + (1 - v / ymax) * (H - t - b);
  let s = '';
  ticks(0, ymax, 4).forEach((v) => { s += `<line x1="${l}" x2="${W - r}" y1="${yp(v)}" y2="${yp(v)}" class="gridl"/><text x="${l - 5}" y="${yp(v) + 3}" class="axt" text-anchor="end">${rNum(v, ymax < 10 ? 1 : 0)}</text>`; });
  s += `<text x="${(l + W - r) / 2}" y="${H - 6}" class="axl" text-anchor="middle">${rEsc(o.xLabel || 'federated round')}</text>`;
  s += `<text transform="translate(12 ${(t + H - b) / 2}) rotate(-90)" class="axl" text-anchor="middle">${rEsc(o.yLabel || '')}</text>`;
  const pts = vals.map((v, i) => (v === null || v === undefined) ? null : `${xp(i)},${yp(v)}`).filter(Boolean);
  s += `<polyline points="${pts.join(' ')}" fill="none" style="stroke:${o.color || 'var(--bad)'}" stroke-width="2"/>`;
  vals.forEach((v, i) => { if (v !== null && v !== undefined) s += `<circle cx="${xp(i)}" cy="${yp(v)}" r="2.6" style="fill:${o.color || 'var(--bad)'}"><title>round ${i + 1}: ${rNum(v, 2)}</title></circle>`; });
  return svgWrap(W, H, s, o.label || 'line chart');
}

/** Grouped bars of rates in [0, 1]. */
function groupedBars(cats, series, o = {}) {
  const W = o.width || 760, H = 280, l = 48, r = 16, t = 16, b = 64;
  const gw = (W - l - r) / cats.length, bw = Math.min(34, gw / (series.length + 1));
  const yp = (v) => t + (1 - v) * (H - t - b);
  let s = '';
  ticks(0, 1, 5).forEach((v) => { s += `<line x1="${l}" x2="${W - r}" y1="${yp(v)}" y2="${yp(v)}" class="gridl"/><text x="${l - 6}" y="${yp(v) + 3}" class="axt" text-anchor="end">${v.toFixed(1)}</text>`; });
  cats.forEach((c, i) => {
    const cx = l + gw * (i + 0.5);
    s += `<text x="${cx}" y="${H - b + 16}" class="axt" text-anchor="middle">${rEsc(c.length > 16 ? c.slice(0, 15) + '…' : c)}</text>`;
    series.forEach((sr, j) => {
      const v = sr.vals[i];
      if (v === null || v === undefined) return;
      const bx = cx + (j - (series.length - 1) / 2) * (bw + 3) - bw / 2;
      const ci = sr.cis && sr.cis[i];
      s += `<g class="mark"><title>${rEsc(sr.name)} — ${rEsc(c)}: ${v.toFixed(3)}${ci ? ` [${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]` : ''}</title>` +
           `<rect x="${bx}" y="${yp(v)}" width="${bw}" height="${Math.max(0, yp(0) - yp(v))}" style="fill:${sr.color}" opacity="0.85"/>` +
           (ci ? `<line x1="${bx + bw / 2}" x2="${bx + bw / 2}" y1="${yp(ci[0])}" y2="${yp(ci[1])}" class="whisk"/>` : '') + '</g>';
    });
  });
  return svgWrap(W, H, s, o.label || 'grouped bars') + legend(series.map((x) => [x.name, x.color]));
}

function legend(items) {
  return '<div class="chart-legend">' + items.map((i) => `<span><i style="background:${i[1]}"></i>${rEsc(i[0])}</span>`).join('') + '</div>';
}

/* ----------------------------------------------------------- tables, cards */
function card(heading, lede, body, id) {
  return `<section class="card" ${id ? `id="${id}"` : ''}><h2>${rEsc(heading)}</h2>${lede ? `<p class="lede">${lede}</p>` : ''}${body}</section>`;
}

function methodTable(group) {
  const rows = Object.entries(group).filter(([, v]) => v && v.avg_latency_ms);
  if (!rows.length) return '<div class="empty">No comparable records.</div>';
  const hasCompleted = rows.some(([, v]) => v.avg_completed_latency_ms);
  const head = `<tr><th>Method / scenario</th><th>Completion, %</th>` +
    `<th>${hasCompleted ? 'Latency (completed), ms' : 'Latency, ms'}</th><th>Reward</th><th>Energy, J/task</th>` +
    `<th>FL traffic, MiB</th><th>Seeds</th></tr>`;
  const body = rows.map(([name, v]) => {
    const lat = hasCompleted ? v.avg_completed_latency_ms : v.avg_latency_ms;
    const comm = v.comm_overhead_kb && typeof v.comm_overhead_kb === 'object' ? v.comm_overhead_kb.mean : v.comm_overhead_kb;
    return `<tr><td>${rEsc(name)}</td><td>${rStat(v.completion_rate)}</td><td>${rStat(lat)}</td>` +
      `<td>${rStat(v.avg_reward)}</td><td>${rStat(v.avg_energy_j, 4)}</td><td>${comm ? (comm / 1024).toFixed(1) : '0'}</td><td>${rEsc(v.num_seeds)}</td></tr>`;
  }).join('');
  return `<div class="scroll-x"><table><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
}

function claimsTable(rows, ruleText) {
  const pill = (v) => `<span class="pill ${v === 'SUPPORTED' ? 'ok' : (v.indexOf('CONTRADICTED') === 0 ? 'bad' : 'warn')}">${rEsc(v.toLowerCase())}</span>`;
  const body = rows.map((c) =>
    `<tr><td>${rEsc(c.id)} ${rEsc(c.text)}</td><td>${rNum(c.mean_difference)} [${rNum(c.ci95_low)}, ${rNum(c.ci95_high)}]</td><td>${c.n}</td>` +
    `<td>${rNum(c.p_value, 3)}</td><td>${c.p_holm === null || c.p_holm === undefined ? 'N/A' : rNum(c.p_holm, 3)}</td>` +
    `<td>${rNum(c.mde_80, 1)}</td><td>${pill(c.verdict)}</td></tr>`).join('');
  return `<p class="lede">${rEsc(ruleText || '')}</p><div class="scroll-x"><table><thead><tr><th>Claim</th><th>Difference, pp [95 % CI]</th><th>n</th><th>raw p</th><th>Holm p</th><th>Min. detectable, pp</th><th>Verdict</th></tr></thead><tbody>${body}</tbody></table></div>`;
}

function kvGrid(obj, keys) {
  return '<dl class="kvgrid">' + keys.filter((k) => obj && obj[k] !== undefined && obj[k] !== null).map((k) => {
    let v = obj[k];
    if (typeof v === 'object') v = JSON.stringify(v);
    return `<dt>${rEsc(k)}</dt><dd>${rEsc(v)}</dd>`;
  }).join('') + '</dl>';
}

const KIND_OF = {
  FedAvg_Clean_Reference: 'reference', Oracle_Exclusion_Upper_Bound: 'oracle', Proposed_TFRL: 'defence',
  Standard_FedAvg_RL: 'undefended', Robust_FL_MultiKrum: 'defence', Robust_FL_Median: 'defence', Robust_FL_TrimmedMean: 'defence',
  Non_Federated_D3QN: 'learned', Nearest_RSU: 'heuristic', Resource_Aware: 'heuristic', Local_Only: 'heuristic', Random_Valid: 'heuristic',
};
const CLAIM_SHORT = {
  H1: 'H1 attack impact: FedAvg − clean', H2: 'H2 defence recovery: trust-aware − FedAvg', H3: 'H3 oracle headroom: oracle − FedAvg',
  H4: 'H4 residual loss: trust-aware − clean', H5: 'H5 trust-aware − Multi-Krum', H6: 'H6 federation gain: clean − non-federated',
  H7: 'H7 trust-aware − Nearest-RSU', H8: 'H8 trust-aware − greedy resource-aware',
  S1: 'S1 trust features − masked (faulty RSU)', S2: 'S2 learned − Nearest-RSU (faulty RSU)', S3: 'S3 learned − trust-greedy (faulty RSU)',
  S4: 'S4 faulty RSU − no fault', S5: 'S5 staleness discount − none', S6: 'S6 trust-aware − FedAvg (intermittent)',
};

function cellRow(label, cell, kind, perSeed) {
  const c = cell && cell.completion_rate;
  if (!c || c.mean === null || c.mean === undefined) return null;
  return { label, mean: c.mean, lo: c.mean - (c.ci95 || 0), hi: c.mean + (c.ci95 || 0), kind,
           dots: perSeed ? Object.values(perSeed) : [], note: `${cell.num_seeds} seeds` };
}

/* ------------------------------------------------------------------ main */
async function loadResearch() {
  const root = document.getElementById('research-root');
  if (!root || researchLoaded) return;
  root.innerHTML = '<div class="empty">Loading the active run…</div>';
  let summary, figures;
  try {
    const [s, f] = await Promise.all([
      fetch('/api/research/summary').then((r) => r.json()),
      fetch('/api/figures').then((r) => r.json()).catch(() => ({ status: 'error', figures: [] })),
    ]);
    summary = s; figures = f;
  } catch (e) {
    root.innerHTML = `<div class="card"><h2>Research results</h2><p class="lede">The results could not be loaded: ${rEsc(e && e.message ? e.message : e)}.</p></div>`;
    return;
  }
  if (!summary || summary.status !== 'ok') {
    root.innerHTML = `<div class="card"><h2>Research results</h2><p class="lede">No validated active run is available (${rEsc(summary && summary.reason)}).</p></div>`;
    return;
  }

  const md = summary.metadata || {};
  const R = summary.results || {};
  const D = summary.derived || {};
  const A = R.experiment_a_baseline_comparison || {};
  const parts = [];
  const nav = [];
  const section = (id, label, html) => { nav.push([id, label]); parts.push(html); };

  const valid = md.validation_status === 'passed';
  section('sec-run', 'Run', card('Active run',
    'Offline evaluation results (not the live simulation). Every number below is read from this run’s artefacts.',
    `<p><span class="pill ${valid ? 'ok' : 'bad'}">validation ${rEsc(md.validation_status)}</span><span class="pill">${rEsc(md.run_kind)}</span>` +
    `<span class="pill">${(md.seeds || []).length} paired seeds</span></p>` +
    kvGrid({ run_id: summary.run_id, timestamp_utc: md.timestamp_utc, seeds: (md.seeds || []).join(', '),
             sweep_seeds: (md.sweep_seeds || []).join(', '), evaluation: md.evaluation_config,
             training: md.training_protocol && { rounds: md.training_protocol.num_rounds, steps_per_round: md.training_protocol.steps_per_round, local_gradient_steps_per_round: md.training_protocol.local_epochs_gradient_steps_per_round },
             environment_model: md.environment_model, attack: md.attack_settings,
             code_sha256: md.code_revision && md.code_revision.content_sha256 },
           ['run_id', 'timestamp_utc', 'seeds', 'sweep_seeds', 'evaluation', 'training', 'environment_model', 'attack', 'code_sha256']), 'sec-run'));

  // ---- headline: policies ------------------------------------------------------
  const order = ['FedAvg_Clean_Reference', 'Oracle_Exclusion_Upper_Bound', 'Robust_FL_MultiKrum', 'Nearest_RSU', 'Proposed_TFRL',
                 'Robust_FL_TrimmedMean', 'Standard_FedAvg_RL', 'Resource_Aware', 'Non_Federated_D3QN', 'Robust_FL_Median', 'Local_Only', 'Random_Valid'];
  const polRows = order.filter((m) => A[m]).map((m) => cellRow(m.replace(/_/g, ' '), A[m], KIND_OF[m] || 'neutral', (D.per_seed_completion || {})[m])).filter(Boolean);
  if (polRows.length) {
    const nr = A.Nearest_RSU && A.Nearest_RSU.completion_rate.mean;
    section('sec-policies', 'Policies', card('Task completion by method',
      'Bars are the mean over seeds; whiskers are the 95 % CI; dots are the individual seeds. Colour: ' +
      'grey = no-attack reference, green = oracle bound, red = undefended FedAvg, blue = defences, amber = non-learning heuristics. ' +
      'The dashed line is the Nearest-RSU heuristic.',
      barsCI(polRows, { min: 0, max: 100, ref: nr, refLabel: 'Nearest-RSU', xLabel: 'Task completion rate, %', label: 'Task completion by method' }) +
      legend([['clean reference', KIND_COLOR.reference], ['oracle bound', KIND_COLOR.oracle], ['undefended FedAvg', KIND_COLOR.undefended],
              ['defence / learned', KIND_COLOR.defence], ['heuristic', KIND_COLOR.heuristic]]), 'sec-policies'));
  }

  if (summary.claims && summary.claims.rows) {
    const rows = summary.claims.rows.map((c) => ({ id: c.id, short: CLAIM_SHORT[c.id] || c.id, mean: c.mean_difference, lo: c.ci95_low, hi: c.ci95_high,
                                                    p: c.p_value, pHolm: c.p_holm, verdict: c.verdict }));
    section('sec-claims', 'Claims', card('Downstream claims',
      'Task-completion differences in percentage points, paired by seed. Green = supported after Holm correction, grey = not established, red = contradicted.',
      forestPlot(rows, { xLabel: 'Paired difference in completion, percentage points (95 % CI)', label: 'Forest plot of the eight pre-specified claims' }) +
      claimsTable(summary.claims.rows, summary.claims.rule), 'sec-claims'));
  }

  // ---- defence by attack ratio ------------------------------------------------
  const tr = R.experiment_c_attack_ratio_sweep, fa = R.experiment_c2_attack_ratio_fedavg;
  if (tr && fa) {
    const mk = (g) => Object.keys(g).filter((k) => g[k] && g[k].completion_rate).map((k) => ({
      x: Number(k), mean: g[k].completion_rate.mean, lo: g[k].completion_rate.mean - (g[k].completion_rate.ci95 || 0),
      hi: g[k].completion_rate.mean + (g[k].completion_rate.ci95 || 0), n: g[k].num_seeds }));
    section('sec-ratio', 'Attack ratio', card('Benefit and cost of the defence by attack ratio',
      'The defence matters when contamination is high and costs completion when there is nothing to defend against (false exclusions). The 0.24 point uses 16 seeds, the others 6.',
      lineCI([{ name: 'FedAvg (undefended)', color: 'var(--bad)', points: mk(fa) }, { name: 'Trust-aware', color: 'var(--accent)', points: mk(tr) }],
             { xLabel: 'Fraction of malicious clients', yLabel: 'Task completion, %', ref: A.Nearest_RSU && A.Nearest_RSU.completion_rate.mean, refLabel: 'Nearest-RSU', label: 'Completion by attack ratio' }), 'sec-ratio'));
  }

  // ---- detector ----------------------------------------------------------------
  const P = A.Proposed_TFRL;
  if (P && P.pooled_confusion_counts) {
    const dets = [['deployed', P.pooled_confusion_counts]].concat(Object.entries(P.pooled_shadow_confusion || {}).filter(([k]) => k !== 'no_detection'));
    const tpr = dets.map(([, c]) => c.tp / Math.max(1, c.tp + c.fn)), fpr = dets.map(([, c]) => c.fp / Math.max(1, c.fp + c.tn));
    const ci = (k, n) => { const w = wilson(k, n); return w ? [w[0], w[1]] : null; };
    const grouped = groupedBars(dets.map(([n]) => n.replace(/_/g, ' ')), [
      { name: 'TPR (attackers found)', color: 'var(--ok)', vals: tpr, cis: dets.map(([, c]) => ci(c.tp, c.tp + c.fn)) },
      { name: 'FPR (honest clients excluded)', color: 'var(--bad)', vals: fpr, cis: dets.map(([, c]) => ci(c.fp, c.fp + c.tn)) }],
      { label: 'Detector comparison' });
    const fe = D.proposed_false_exclusion_by_round;
    const tl = D.proposed_train_loss_by_round;
    section('sec-detector', 'Detector', card('Detector on real federated updates',
      'All detectors judged the same updates (shadow evaluation). Wilson intervals treat decisions as independent, which they are not, so they are optimistic. ' +
      'Right: the deployed detector excludes more honest clients as training converges.',
      `<div class="two">${'<div>' + grouped + '</div>'}<div>${fe ? lineSimple(fe, { yLabel: 'honest clients excluded, %', label: 'False exclusions by round' }) : ''}` +
      `${tl ? lineSimple(tl, { yLabel: 'mean local training loss', color: 'var(--accent)', label: 'Training loss by round' }) : ''}</div></div>` +
      `<div class="scroll-x"><table><thead><tr><th>Detector</th><th>TP</th><th>FN</th><th>FP</th><th>TN</th><th>TPR [95 % CI]</th><th>FPR [95 % CI]</th></tr></thead><tbody>` +
      dets.map(([n, c]) => `<tr><td>${rEsc(n)}</td><td>${c.tp}</td><td>${c.fn}</td><td>${c.fp}</td><td>${c.tn}</td><td>${rRate(c.tp, c.tp + c.fn)}</td><td>${rRate(c.fp, c.fp + c.tn)}</td></tr>`).join('') +
      '</tbody></table></div>', 'sec-detector'));
  }

  // ---- attack families --------------------------------------------------------
  const fam = R.experiment_f_attack_families;
  if (fam) {
    const attacks = [...new Set(Object.keys(fam).map((k) => k.split('/')[0]))];
    const have = attacks.filter((a) => fam[a + '/trust_aware'] && fam[a + '/trust_aware'].pooled_confusion_counts);
    const rate = (a, nm) => { const cell = fam[a + '/trust_aware']; const c = nm === 'deployed' ? cell.pooled_confusion_counts : (cell.pooled_shadow_confusion || {})[nm]; return c ? c.tp / Math.max(1, c.tp + c.fn) : null; };
    section('sec-attacks', 'Attack families', card('Detection by attack family',
      'The deployed detector and two shadow rules on identical updates. Partial, norm-matched and intermittent attacks are largely missed.',
      groupedBars(have, [
        { name: 'deployed', color: 'var(--accent)', vals: have.map((a) => rate(a, 'deployed')) },
        { name: 'distance_mad', color: 'var(--ok)', vals: have.map((a) => rate(a, 'distance_mad')) },
        { name: 'krum', color: 'var(--warn)', vals: have.map((a) => rate(a, 'krum')) }], { label: 'TPR by attack family' }) +
      '<h3>Completion under each attack</h3>' + methodTable(fam), 'sec-attacks'));
  }

  // ---- ablation / alpha -----------------------------------------------------------
  const abl = R.experiment_d_ablation;
  if (abl) {
    const rows = Object.keys(abl).map((k) => cellRow(k.replace(/_/g, ' '), abl[k], k === 'FedAvg_RL_No_Trust' ? 'undefended' : 'defence', null)).filter(Boolean);
    section('sec-ablation', 'Ablation', card('Which trust component matters?',
      'Behavioural weight alpha = 0 uses model trust only; alpha = 1 uses behavioural trust only. Attackers are behaviourally honest, so behavioural trust alone cannot find them.',
      barsCI(rows, { min: 40, max: 100, xLabel: 'Task completion, %', label: 'Ablation' }) + '<h3>Detection per variant</h3>' + methodTable(abl), 'sec-ablation'));
  }

  // ---- supplementary: edge node, connectivity ----------------------------------
  (summary.supplementary || []).forEach((sup) => {
    const SR = sup.results || {};
    const L = SR.experiment_l_faulty_edge_node, M = SR.experiment_m_intermittent_connectivity, K2 = SR.experiment_k2_mixed_threat_relabelled;
    const PS = sup.per_seed_completion || {};
    if (sup.claims && sup.claims.length) {
      const rows = sup.claims.map((c) => ({ id: c.id, short: CLAIM_SHORT[c.id] || c.id, mean: c.mean_difference, lo: c.ci95_low, hi: c.ci95_high, p: c.p_value, pHolm: c.p_holm, verdict: c.verdict }));
      section('sec-supp-claims', 'Supplementary claims', card('Supplementary claims: faulty edge node and intermittent connectivity',
        `Run ${rEsc(sup.run_id)} supplements the active run (same seeds, protocol and code paths); these are the problem statement’s edge-node and connectivity requirements. Exploratory: 6 seeds, Holm-adjusted across the six claims.`,
        forestPlot(rows, { xLabel: 'Paired difference in completion, percentage points (95 % CI)', label: 'Supplementary claims' }) + claimsTable(sup.claims, ''), 'sec-supp-claims'));
    }
    if (L) {
      const ref = (m) => A[m];
      const rows = [
        cellRow('learned, no fault (main run)', ref('Proposed_TFRL'), 'reference', null),
        cellRow('learned + trust features, faulty RSU', L.learned_with_trust_features, 'defence', (PS.experiment_l_faulty_edge_node || {}).learned_with_trust_features),
        cellRow('learned, trust features masked, faulty RSU', L.learned_trust_features_masked, 'undefended', (PS.experiment_l_faulty_edge_node || {}).learned_trust_features_masked),
        cellRow('trust-greedy heuristic, faulty RSU', L.Trust_Greedy_faulty, 'heuristic', null),
        cellRow('resource-aware heuristic, faulty RSU', L.Resource_Aware_faulty, 'heuristic', null),
        cellRow('nearest-RSU heuristic, faulty RSU', L.Nearest_RSU_faulty, 'heuristic', null),
        cellRow('local only, faulty RSU', L.Local_Only_faulty, 'heuristic', null)].filter(Boolean);
      section('sec-edge', 'Faulty edge node', card('Faulty edge node: does behavioural trust help the offloading decision?',
        'RSU_2 silently drops 60 % of the tasks it is given (injected fault). Behavioural trust of each RSU is part of the learned policy’s state; the masked policy cannot see it. ' +
        'The no-fault bar is the main-run policy on the same seeds.',
        barsCI(rows, { min: 20, max: 100, xLabel: 'Task completion, %', label: 'Faulty edge node' }) + '<h3>All cells</h3>' + methodTable(L), 'sec-edge'));
    }
    if (M) {
      const rows = [cellRow('FedAvg, stale uploads', M.fedavg_stale_uploads, 'undefended', null),
                    cellRow('trust-aware, no staleness handling', M.trust_aware_no_staleness_handling, 'defence', null),
                    cellRow('trust-aware + staleness discount', M.trust_aware_staleness_discount, 'oracle', null),
                    cellRow('no dropout (main run, FedAvg)', A.Standard_FedAvg_RL, 'reference', null),
                    cellRow('no dropout (main run, trust-aware)', A.Proposed_TFRL, 'reference', null)].filter(Boolean);
      section('sec-conn', 'Connectivity', card('Intermittent connectivity: stale model updates',
        'Each client is offline in a round with probability 0.4 and then also misses the model broadcast, so its next upload is stale. The discount divides a client’s aggregation weight by 1 + staleness.',
        barsCI(rows, { min: 40, max: 100, xLabel: 'Task completion, %', label: 'Intermittent connectivity' }) + '<h3>All cells</h3>' + methodTable(M), 'sec-conn'));
    }
    if (K2 && sup.unreliable_summary) {
      const rows = Object.entries(sup.unreliable_summary).map(([n, u]) => {
        const c = u.confusion || { tp: 0, fn: 0, fp: 0, tn: 0 };
        return `<tr><td>${rEsc(n)}</td><td>${rRate(c.tp, c.tp + c.fn)}</td><td>${rRate(c.fp, c.fp + c.tn)}</td><td>${rRate(u.unreliable_excluded, u.unreliable_decisions)}</td></tr>`;
      }).join('');
      section('sec-k2', 'Mixed threat', card('Mixed threat with correct labels',
        'Task-dropping vehicles are honest participants of the federated round; only poisoning and free-riding clients count as attackers.',
        `<div class="scroll-x"><table><thead><tr><th>Variant</th><th>TPR on real attackers</th><th>FPR on honest clients</th><th>Unreliable-but-honest vehicles excluded</th></tr></thead><tbody>${rows}</tbody></table></div>` + methodTable(K2), 'sec-k2'));
    }
  });

  // ---- remaining sweeps as tables ---------------------------------------------
  let tables = '';
  [['experiment_b_mobility_sweep', 'Mobility source (trust-aware)'], ['experiment_b2_mobility_fedavg', 'Mobility source (FedAvg)'],
   ['experiment_e_alpha_sensitivity', 'Sensitivity to alpha'], ['experiment_g_participation', 'Partial participation and connectivity gating'],
   ['experiment_h_heterogeneity', 'Task heterogeneity'], ['experiment_i_reward_trust_bonus', 'Reward with the legacy trust bonus'],
   ['experiment_j_environment_v1', 'Environment version 1 (earlier simulator semantics)'], ['experiment_k_mixed_threat', 'Mixed threat (original labels)']]
    .forEach(([key, name]) => { if (R[key]) tables += `<details><summary>${rEsc(name)}</summary>${methodTable(R[key])}</details>`; });
  section('sec-tables', 'Tables', card('Further sweeps (exploratory, 6 seeds)', 'Expand a sweep to see its table.', tables, 'sec-tables'));

  // ---- figures ------------------------------------------------------------------
  if (figures && figures.status === 'ok' && figures.figures.length) {
    section('sec-figures', 'Figures', card('Figures',
      `Generated from the run’s raw records (source SHA-256 ${rEsc((figures.results_file_sha256 || '').slice(0, 16))}). Click a figure to enlarge it.`,
      '<div class="figs">' + figures.figures.map((f) =>
        `<figure><img src="/api/plots/${encodeURIComponent(f.filename)}" alt="${rEsc(f.title || f.filename)}" loading="lazy" data-full="/api/plots/${encodeURIComponent(f.filename)}">` +
        `<figcaption>${rEsc(f.title || f.filename)}</figcaption></figure>`).join('') + '</div>', 'sec-figures'));
  }

  // ---- provenance --------------------------------------------------------------
  const prov = md.dataset_provenance || {};
  const disc = md.provenance_disclosure || {};
  section('sec-prov', 'Provenance', card('Data provenance and scenario definitions',
    'Measured, trace-replayed and simulated components are kept separate.',
    kvGrid(disc, Object.keys(disc)) + '<h3>Datasets</h3>' +
    kvGrid(Object.fromEntries(Object.entries(prov).map(([k, v]) => [k, v && (v.mobility_source || JSON.stringify(v)) + (v && v.trace_sha256 ? ' · sha256 ' + v.trace_sha256.slice(0, 12) : '')])), Object.keys(prov)) +
    ((md.limitations && md.limitations.length) ? '<h3>Limitations recorded with this run</h3><ul>' + md.limitations.map((l) => `<li>${rEsc(l)}</li>`).join('') + '</ul>' : ''), 'sec-prov'));

  root.innerHTML = `<nav class="subnav">${nav.map(([id, l]) => `<a href="#${id}">${rEsc(l)}</a>`).join('')}</nav>` + parts.join('');
  root.querySelectorAll('.nav-a, .subnav a').forEach((a) => a.addEventListener('click', (e) => {
    e.preventDefault();
    const t = document.getElementById(a.getAttribute('href').slice(1));
    if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }));
  root.querySelectorAll('.figs img').forEach((img) => img.addEventListener('click', () => openLightbox(img.dataset.full, img.alt)));
  researchLoaded = true;
}

function openLightbox(src, alt) {
  let box = document.getElementById('lightbox');
  if (!box) {
    box = document.createElement('div');
    box.id = 'lightbox';
    box.className = 'lightbox';
    box.innerHTML = '<img alt=""><button class="btn" aria-label="Close">Close</button>';
    document.body.appendChild(box);
    const close = () => { box.style.display = 'none'; };
    box.addEventListener('click', close);
    window.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
  }
  const img = box.querySelector('img');
  img.src = src; img.alt = alt || '';
  box.style.display = 'flex';
}
