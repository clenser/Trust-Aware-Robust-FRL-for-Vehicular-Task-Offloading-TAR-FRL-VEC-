/* Layout controls for the console: collapsible side panels, a tabbed monitor,
 * accordion sections and a full-screen viewport.  Purely presentational: it
 * never touches simulation state, and every control degrades to the plain
 * layout when storage or the Fullscreen API is unavailable. */
(function () {
  'use strict';
  const store = {
    get(k, d) { try { const v = localStorage.getItem('vannet.' + k); return v === null ? d : v; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('vannet.' + k, v); } catch (e) { /* storage unavailable */ } },
  };
  const ws = document.getElementById('view-sim');
  if (!ws) return;

  function setCollapsed(side, collapsed) {
    ws.classList.toggle('collapse-' + side, collapsed);
    store.set('collapse-' + side, collapsed ? '1' : '0');
    const b = document.getElementById('btn-toggle-' + side);
    if (b) { b.classList.toggle('active', !collapsed); b.setAttribute('aria-pressed', String(!collapsed)); }
    // let the canvas renderers pick up the new width
    setTimeout(() => window.dispatchEvent(new Event('resize')), 60);
  }
  ['left', 'right'].forEach((side) => {
    const b = document.getElementById('btn-toggle-' + side);
    if (b) b.addEventListener('click', () => setCollapsed(side, !ws.classList.contains('collapse-' + side)));
    setCollapsed(side, store.get('collapse-' + side, side === 'right' ? '1' : '0') === '1');
  });

  // ----- accordion sections (click a heading to fold it)
  document.querySelectorAll('#panel-left > section, #panel-right > section').forEach((sec) => {
    const h = sec.querySelector(':scope > h2');
    if (!h) return;
    h.classList.add('fold-head');
    h.setAttribute('role', 'button'); h.setAttribute('tabindex', '0');
    const toggle = () => sec.classList.toggle('folded');
    h.addEventListener('click', toggle);
    h.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
  });

  // ----- tabbed monitor: show one group of sections at a time
  const right = document.getElementById('panel-right');
  if (right) {
    const secs = Array.from(right.querySelectorAll(':scope > section'));
    const groups = [
      ['Learning', [0, 1]],
      ['Trust', [2, 3]],
      ['History', [4, 5]],
    ];
    const bar = document.createElement('div');
    bar.className = 'subtabs'; bar.setAttribute('role', 'tablist');
    const apply = (idx) => {
      secs.forEach((s, i) => { s.hidden = !groups[idx][1].includes(i); });
      Array.from(bar.children).forEach((b, i) => b.classList.toggle('active', i === idx));
      store.set('monitor-tab', String(idx));
      setTimeout(() => window.dispatchEvent(new Event('resize')), 30);
    };
    groups.forEach(([label], i) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'subtab'; b.textContent = label; b.setAttribute('role', 'tab');
      b.addEventListener('click', () => apply(i));
      bar.appendChild(b);
    });
    right.insertBefore(bar, right.firstChild);
    apply(Math.min(groups.length - 1, parseInt(store.get('monitor-tab', '0'), 10) || 0));
  }

  // ----- viewport: legend toggle, full-screen, focus mode
  const wrap = document.querySelector('.viewport-wrap');
  const tools = document.querySelector('.viewport-tools');
  if (wrap && tools) {
    const legend = Array.from(tools.querySelectorAll('.legend-chip'));
    const lb = document.getElementById('btn-legend');
    const showLegend = (on) => { legend.forEach((c) => { c.hidden = !on; }); if (lb) lb.classList.toggle('active', on); store.set('legend', on ? '1' : '0'); };
    if (lb) lb.addEventListener('click', () => showLegend(legend[0] ? legend[0].hidden : true));
    showLegend(store.get('legend', '0') === '1');

    const fs = document.getElementById('btn-fullscreen');
    if (fs) fs.addEventListener('click', () => {
      try {
        if (document.fullscreenElement) document.exitFullscreen();
        else if (wrap.requestFullscreen) wrap.requestFullscreen();
        else wrap.classList.toggle('pseudo-fs');
      } catch (e) { wrap.classList.toggle('pseudo-fs'); }
    });
    document.addEventListener('fullscreenchange', () => setTimeout(() => window.dispatchEvent(new Event('resize')), 80));

    const focus = document.getElementById('btn-focus');
    if (focus) focus.addEventListener('click', () => {
      const on = !ws.classList.contains('focus-mode');
      ws.classList.toggle('focus-mode', on); focus.classList.toggle('active', on);
      setTimeout(() => window.dispatchEvent(new Event('resize')), 60);
    });
  }
})();
