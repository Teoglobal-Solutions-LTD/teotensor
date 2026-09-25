"""The studio page. One HTML document, no external site, no second network.

The browser asks Python to fit and to build frames. It only paints.
"""

from __future__ import annotations

PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>TeoTensor studio</title>
<style>
  :root {
    color-scheme: dark;
    --accent: #0d9488;
    --ink: #e7eef6;
    --muted: #93a4b8;
    --tone: #243044;
    --paper: #161e2b;
    --canvas: #0c121b;
    --well: inset 0 1px 2px rgba(0, 0, 0, 0.55);
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    font-family: "Segoe UI", sans-serif;
    background: var(--canvas);
    color: var(--ink);
    overflow-x: hidden;
  }
  .blobs {
    position: fixed;
    inset: 0;
    z-index: 0;
    overflow: hidden;
    pointer-events: none;
  }
  .blobs i {
    position: absolute;
    display: block;
    border-radius: 50%;
    filter: blur(56px);
    opacity: 0.72;
    background: var(--accent);
  }
  .blobs i:nth-child(1) {
    width: 480px;
    height: 480px;
    left: -140px;
    top: -120px;
    animation: blob-a 48s ease-in-out infinite;
  }
  .blobs i:nth-child(2) {
    width: 360px;
    height: 360px;
    right: -100px;
    top: 12vh;
    background: #14b8a6;
    animation: blob-b 62s ease-in-out infinite;
  }
  .blobs i:nth-child(3) {
    width: 300px;
    height: 300px;
    left: 28vw;
    bottom: -140px;
    background: #0f766e;
    animation: blob-c 40s ease-in-out infinite;
  }
  @keyframes blob-a {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(46vw, 42vh); }
  }
  @keyframes blob-b {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(-40vw, 32vh); }
  }
  @keyframes blob-c {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(22vw, -48vh); }
  }
  @media (prefers-reduced-motion: reduce) {
    .blobs i { animation: none; }
    label.switch .track::after { transition: none; }
    .spinner { animation: none; }
    #net-canvas.live { animation: none; }
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    padding: 18px 22px 8px;
  }
  .brand { display: flex; gap: 12px; align-items: center; }
  .mark {
    width: 12px; height: 28px; border-radius: 99px;
    background: var(--accent); flex: none;
  }
  header strong { display: block; font-size: 18px; letter-spacing: -0.02em; }
  header p { margin: 2px 0 0; color: var(--muted); font-size: 13px; }
  .top { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  header, main { position: relative; z-index: 1; }
  main {
    flex: 1;
    display: flex;
    flex-direction: column;
    width: min(980px, 100%);
    margin: 0 auto;
    padding: 12px 16px 28px;
  }
  .panel {
    flex: 1;
    background:
      linear-gradient(180deg, rgba(13, 148, 136, 0.16), transparent 96px),
      var(--paper);
    border: 0;
    border-radius: 22px;
    padding: 22px;
    box-shadow:
      inset 0 1px 0 rgba(255, 255, 255, 0.06),
      0 18px 40px rgba(0, 0, 0, 0.35);
  }
  h2 {
    margin: 18px 0 8px;
    font-size: 12px;
    font-weight: 650;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .panel > h2:first-child { margin-top: 0; }
  .lede { margin: 0 0 12px; color: var(--muted); font-size: 14px; }
  button, select, input, textarea { font: inherit; color: var(--ink); }
  button {
    background: var(--accent);
    color: white;
    border: 0;
    border-radius: 999px;
    padding: 8px 14px;
    cursor: pointer;
  }
  button:hover { background: #0f766e; }
  button.secondary { background: var(--tone); color: var(--ink); }
  button.secondary:hover { background: #314158; }
  .steps button, .tabs button {
    background: transparent;
    color: var(--muted);
    border-radius: 999px;
    padding: 6px 12px;
  }
  .steps button:hover, .tabs button:hover {
    background: var(--tone);
    color: var(--ink);
  }
  .steps button[aria-current="step"],
  .tabs button.active {
    background: var(--accent);
    color: white;
  }
  .tabs { display: flex; gap: 6px; flex-wrap: wrap; }
  #tabs { margin-top: 18px; }
  input, select, textarea {
    width: 100%;
    border: 0;
    border-radius: 12px;
    padding: 8px 10px;
    background: var(--canvas);
    color: var(--ink);
    box-shadow: var(--well);
  }
  textarea { min-height: 120px; resize: vertical; line-height: 1.45; }
  input:focus, select:focus, textarea:focus {
    outline: 2px solid rgba(13, 148, 136, 0.45);
  }
  .pick { position: relative; width: 100%; }
  .pager .pick { width: 72px; flex: none; }
  button.pick-face, button.pick-item {
    color: var(--ink);
    text-align: left;
    border-radius: 12px;
  }
  button.pick-face {
    position: relative;
    width: 100%;
    padding: 8px 32px 8px 10px;
    background: var(--canvas);
    box-shadow: var(--well);
  }
  button.pick-face::after {
    content: "";
    position: absolute;
    right: 12px;
    top: 50%;
    width: 6px;
    height: 6px;
    border-right: 1.5px solid var(--muted);
    border-bottom: 1.5px solid var(--muted);
    transform: translateY(-70%) rotate(45deg);
  }
  .pick.open button.pick-face::after {
    transform: translateY(-30%) rotate(225deg);
  }
  button.pick-face:hover { background: var(--canvas); }
  button.pick-face:focus {
    outline: 2px solid rgba(13, 148, 136, 0.45);
  }
  .pick.open { z-index: 40; }
  .pick-menu {
    position: absolute;
    z-index: 40;
    top: calc(100% + 6px);
    left: 0;
    min-width: 100%;
    max-height: 280px;
    overflow: auto;
    padding: 6px;
    background: #1c2838;
    border-radius: 14px;
    box-shadow:
      inset 0 1px 0 rgba(255, 255, 255, 0.06),
      0 16px 36px rgba(0, 0, 0, 0.45);
  }
  .pick.up .pick-menu { top: auto; bottom: calc(100% + 6px); }
  button.pick-item {
    display: block;
    width: 100%;
    margin: 0;
    padding: 8px 10px;
    background: transparent;
    box-shadow: none;
    border-radius: 10px;
  }
  button.pick-item:hover { background: var(--tone); }
  button.pick-item.current { background: rgba(13, 148, 136, 0.28); }
  select.native-select {
    position: absolute;
    width: 1px;
    height: 1px;
    margin: 0;
    padding: 0;
    opacity: 0;
    pointer-events: none;
    box-shadow: none;
  }
  input[type="checkbox"], input[type="radio"] {
    width: auto;
    box-shadow: none;
    accent-color: var(--accent);
  }
  label {
    display: block;
    margin: 10px 0 4px;
    font-size: 13px;
    color: var(--muted);
  }
  .field > label {
    color: var(--ink);
    font-size: 14px;
    font-weight: 600;
  }
  .segment { display: flex; gap: 8px; margin: 0; }
  .purpose {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-top: 8px;
  }
  .segment label {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0;
    padding: 8px 14px;
    border-radius: 999px;
    background: var(--canvas);
    color: var(--ink);
  }
  #stage-net {
    position: relative;
    flex: none;
    width: 100%;
    height: auto;
    aspect-ratio: 720 / 420;
    max-height: 68vh;
    margin-top: 8px;
    border-radius: 16px;
    overflow: hidden;
    background: var(--canvas);
    box-shadow: var(--well);
    cursor: grab;
    touch-action: none;
  }
  #stage-net.dragging { cursor: grabbing; }
  #stage-net svg, #stage-net canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }
  #stage-net [hidden] { display: none !important; }
  #net-zoom {
    position: absolute;
    z-index: 3;
    right: 6px;
    top: 28px;
    width: 18px;
    height: calc(100% - 44px);
    margin: 0;
    padding: 0;
    accent-color: #0d9488;
    writing-mode: vertical-lr;
    direction: rtl;
    background: transparent;
    box-shadow: none;
    border-radius: 0;
    cursor: ns-resize;
  }
  #net { background: transparent; border: 0; }
  #net-canvas.live { animation: mesh-breathe 1.4s ease-in-out infinite; }
  @keyframes mesh-breathe { 50% { opacity: 0.62; } }
  .row .segment label input { width: auto; }
  .scale-name { color: var(--muted); font-size: 13px; }
  .legend {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    font-size: 12px;
    color: var(--muted);
    margin: 8px 0;
  }
  .swatch {
    width: 12px; height: 12px; border-radius: 99px;
    display: inline-block; margin-right: 4px;
  }
  .row {
    display: flex; gap: 8px; flex-wrap: wrap;
    margin-top: 12px; align-items: center;
  }
  .spinner {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    border: 2px solid var(--tone);
    border-top-color: var(--accent);
    animation: spin 0.7s linear infinite;
    flex: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  button:disabled { opacity: 0.55; cursor: progress; }
  .row.end { justify-content: flex-end; }
  .row label { display: flex; align-items: center; gap: 8px; margin: 0; }
  .row label input { width: 88px; }
  a { text-decoration: none; }
  #notice { min-height: 18px; margin: 0 0 10px; color: #fb7185; font-size: 13px; }
  #data-note, #caption, .hint { color: var(--muted); font-size: 13px; }
  .hint { margin: 4px 0 0; }
  .grid {
    display: flex;
    max-height: 440px;
    margin-top: 12px;
    padding: 10px 10px 10px 0;
    overflow: hidden;
    border-radius: 16px;
    background: rgba(12, 18, 27, 0.72);
  }
  .grid-scroll {
    flex: 1 1 auto;
    min-width: 0;
    max-height: 420px;
    overflow: auto;
    scrollbar-width: thin;
    scrollbar-color: rgba(231, 238, 246, 0.35) transparent;
  }
  .grid-scroll::-webkit-scrollbar { width: 8px; height: 8px; }
  .grid-scroll::-webkit-scrollbar-track { background: transparent; }
  .grid-scroll::-webkit-scrollbar-thumb {
    background: rgba(231, 238, 246, 0.32);
    border-radius: 99px;
  }
  .grid-scroll::-webkit-scrollbar-thumb:hover {
    background: rgba(231, 238, 246, 0.5);
  }
  .grid-scroll::-webkit-scrollbar-corner { background: transparent; }
  .grid table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 13px;
  }
  .grid th {
    position: sticky;
    top: 0;
    background: #1c2838;
    color: var(--muted);
    font-size: 11px;
    font-weight: 650;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .grid th, .grid td {
    padding: 8px 10px;
    text-align: left;
    border-right: 1px solid rgba(231, 238, 246, 0.12);
    border-bottom: 1px solid rgba(231, 238, 246, 0.12);
  }
  .grid th:last-child, .grid td:last-child { border-right: 0; }
  .grid tr:last-child td { border-bottom: 0; }
  .grid tbody tr:hover td { background: rgba(13, 148, 136, 0.14); }
  .grid td.num { color: var(--muted); width: 72px; }
  .grid td.frame { width: 64px; }
  .grid td.frame canvas {
    display: block;
    height: 40px;
    width: auto;
    image-rendering: pixelated;
    background: #000;
    border-radius: 6px;
  }
  label.switch {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 14px 0 0;
    color: var(--ink);
  }
  label.switch input {
    position: absolute;
    width: 1px;
    height: 1px;
    margin: 0;
    opacity: 0;
  }
  label.switch .track {
    width: 36px;
    height: 20px;
    border-radius: 99px;
    background: var(--tone);
    box-shadow: var(--well);
    position: relative;
    flex: none;
  }
  label.switch .track::after {
    content: "";
    position: absolute;
    top: 3px;
    left: 3px;
    width: 14px;
    height: 14px;
    border-radius: 99px;
    background: var(--muted);
    transition: transform 0.16s ease;
  }
  label.switch input:checked + .track { background: var(--accent); }
  label.switch input:checked + .track::after {
    transform: translateX(16px);
    background: white;
  }
  label.switch input:focus-visible + .track {
    outline: 2px solid rgba(13, 148, 136, 0.45);
    outline-offset: 2px;
  }
  .field { position: relative; }
  .field:has(> .control > label.switch) { margin-top: 16px; }
  .layers-head { display: flex; align-items: center; gap: 8px; }
  .layers-head label {
    flex: 1;
    margin: 10px 0 4px;
    color: var(--ink);
    font-size: 14px;
    font-weight: 600;
  }
  .layer-list { display: flex; flex-direction: column; gap: 8px; }
  .layer-row { display: flex; align-items: center; gap: 8px; }
  .layer-no {
    width: 22px;
    height: 22px;
    border-radius: 99px;
    background: var(--tone);
    color: var(--muted);
    font-size: 12px;
    display: grid;
    place-items: center;
    flex: none;
  }
  .layer-row input { width: 96px; flex: none; }
  .layer-row .pick { flex: 1 1 auto; width: auto; min-width: 0; }
  .layer-remove { flex: none; }
  .layers > button.secondary { margin-top: 10px; }
  .field:hover, .field:focus-within { z-index: 20; }
  .control { display: flex; align-items: center; gap: 8px; }
  .control > input, .control > .pick {
    flex: 1 1 auto;
    width: auto;
    min-width: 0;
  }
  .control > label.switch { flex: 1 1 auto; margin: 0; }
  button.tip {
    position: relative;
    flex: none;
    width: 22px;
    height: 22px;
    padding: 0;
    border-radius: 99px;
    background: var(--tone);
    color: var(--muted);
    font-size: 12px;
    font-style: italic;
    font-weight: 650;
    line-height: 22px;
  }
  button.tip:hover, button.tip:focus {
    background: #314158;
    color: var(--ink);
    outline: none;
  }
  button.tip .bubble {
    display: none;
    position: absolute;
    right: 0;
    bottom: calc(100% + 8px);
    width: 260px;
    padding: 10px 12px;
    border-radius: 14px;
    background: #1c2838;
    color: var(--ink);
    font-style: normal;
    font-weight: 400;
    font-size: 13px;
    line-height: 1.45;
    text-align: left;
    box-shadow:
      inset 0 1px 0 rgba(255, 255, 255, 0.06),
      0 16px 36px rgba(0, 0, 0, 0.45);
    z-index: 2;
  }
  button.tip:hover .bubble, button.tip:focus .bubble { display: block; }
  button.tip.down .bubble { bottom: auto; top: calc(100% + 8px); }
  .sessions { display: flex; flex-direction: column; gap: 8px; margin-top: 16px; }
  .session {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
    padding: 12px 14px;
    border-radius: 14px;
    background: rgba(12, 18, 27, 0.72);
  }
  .session p { margin: 2px 0 0; color: var(--muted); font-size: 12px; }
  .split {
    display: grid;
    grid-template-columns: 1.35fr 0.75fr;
    gap: 22px;
    align-items: start;
  }
  body.stage .blobs { display: none; }
  body.stage main {
    width: 100%;
    padding: 8px 12px 12px;
    min-height: 0;
    overflow: auto;
  }
  body.stage #screen-run {
    display: flex;
    flex: none;
    flex-direction: column;
    min-height: calc(100vh - 96px);
    width: 100%;
  }
  #report-block { flex: none; margin-top: 12px; }
  #report-frame {
    display: block;
    width: 100%;
    height: 0;
    border: 0;
    border-radius: 16px;
    background: #0c121b;
    overflow: hidden;
  }
  .teach-status {
    color: var(--ink);
    font-size: 13px;
    font-variant-numeric: tabular-nums;
  }
  .layer-count { font-family: "Segoe UI", sans-serif; }
  body.stage #screen-run .split {
    flex: 1;
    min-height: 0;
    align-items: stretch;
  }
  body.stage #screen-run .split > div {
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
  }
  body.stage #state { flex: 1; min-height: 0; overflow: auto; }
  pre {
    white-space: pre-wrap;
    font-size: 12px;
    background: var(--canvas);
    border: 0;
    padding: 10px;
    border-radius: 14px;
    min-height: 96px;
    box-shadow: var(--well);
  }
  @media (max-width: 800px) {
    .split { grid-template-columns: 1fr; }
    header { align-items: flex-start; flex-direction: column; }
  }
</style>
</head>
<body>
<div class="blobs" aria-hidden="true"><i></i><i></i><i></i></div>
<header>
  <div class="brand">
    <span class="mark"></span>
    <div>
      <strong>TeoTensor studio</strong>
      <p id="session-label">Create a session, then walk the three screens.</p>
    </div>
  </div>
  <div class="top">
    <nav id="steps" class="steps" hidden>
      <button type="button" data-screen="data">Dataset</button>
      <button type="button" data-screen="model">Model</button>
      <button id="step-run" type="button" data-screen="run">Train</button>
    </nav>
    <button id="back-sessions" class="secondary" type="button" hidden>Sessions</button>
  </div>
</header>
<main>
  <p id="notice"></p>
  <section id="screen-home" class="panel">
    <h2>Session</h2>
    <p class="lede">
      A train session fits a model.
      An inference session loads weights and scores rows.
    </p>
    <label>Name</label>
    <input id="session-name" placeholder="xor"/>
    <div class="purpose">
      <div class="segment">
        <label><input type="radio" name="purpose" value="train" checked/> Train</label>
        <label><input type="radio" name="purpose" value="infer"/> Inference</label>
      </div>
      <button id="create" type="button">Create session</button>
    </div>
    <div id="session-list" class="sessions"></div>
  </section>
  <section id="screen-data" class="panel" hidden>
    <h2>Dataset</h2>
    <p class="lede">One source at a time. Steps then run on a copy of that table.</p>
    <div class="tabs" id="source-tabs">
      <button type="button" class="active" data-source="paste">Paste</button>
      <button type="button" data-source="file">File</button>
      <button type="button" data-source="url">URL</button>
    </div>
    <div data-source-pane="paste">
      <label>Rows. The first line names the columns.</label>
      <textarea id="csv">x1,x2,y
0,0,0
0,1,1
1,0,1
1,1,0</textarea>
      <label>Answer column. Empty when there is no answer.</label>
      <input id="target" value="y"/>
    </div>
    <div data-source-pane="file" hidden>
      <label>CSV, or a zip of IDX image files.</label>
      <input id="file" type="file" accept=".csv,.zip"/>
      <div id="file-target-box" hidden>
        <label>Answer column. Empty when there is no answer.</label>
        <input id="file-target" placeholder="y"/>
      </div>
    </div>
    <div data-source-pane="url" hidden>
      <label>http or https address of a CSV or a zip.</label>
      <input id="url" placeholder="https://"/>
      <label>Answer column for a CSV. A zip ignores this.</label>
      <input id="url-target" placeholder="y"/>
    </div>
    <div class="row"><button id="load" type="button">Load table</button></div>
    <h2>Steps</h2>
    <label>Divide every stored number by</label>
    <input id="divisor" inputmode="decimal" placeholder="empty keeps the numbers"/>
    <p class="hint">
      255 turns a pixel of 255 into 1. Empty means the numbers are already fine.
    </p>
    <label>Share of training rows kept back for a check</label>
    <input id="holdout" inputmode="decimal" value="0"/>
    <p class="hint">
      0.1 sets the last tenth aside and does not teach on it.
      0 teaches on every training row. Test rows stay test.
    </p>
    <label class="switch">
      <input id="center" type="checkbox"/>
      <span class="track"></span>
      <span>Center each column on the training rows</span>
    </label>
    <p class="hint">
      Subtract the training average, then divide by the spread.
      Check and test rows use that same average.
    </p>
    <div class="row"><button id="apply" type="button">Apply steps</button></div>
    <p id="data-note"></p>
    <div id="preview" class="grid"></div>
    <div class="row pager">
      <button id="prev" class="secondary" type="button">Previous</button>
      <span id="page-label"></span>
      <button id="next" class="secondary" type="button">Next</button>
      <label>Rows
        <select id="page-size">
          <option value="8" selected>8</option>
          <option value="20">20</option>
          <option value="50">50</option>
        </select>
      </label>
    </div>
    <div class="row end">
      <button id="to-model" type="button">Continue to model</button>
    </div>
  </section>
  <section id="screen-model" class="panel" hidden>
    <div id="train-model">
      <h2>Model</h2>
      <p class="lede">Blank components keep the documented default until fit.</p>
      <select id="models"></select>
      <div id="tabs" class="tabs"></div>
      <form id="fields"></form>
    </div>
    <div id="infer-model" hidden>
      <h2>Weights</h2>
      <p class="lede">A .ttw file names its model class and carries the numbers.</p>
      <input id="weights" type="file" accept=".ttw"/>
      <div class="row">
        <button id="load-weights" type="button">Load weights</button>
      </div>
      <p id="weight-note"></p>
    </div>
    <div class="row">
      <button id="to-run" type="button">Continue</button>
    </div>
  </section>
  <section id="screen-run" class="panel" hidden>
    <div class="split">
      <div>
        <div id="train-run">
          <h2>Train</h2>
          <div class="row">
            <button id="fit" type="button">Teach until it settles</button>
            <button id="step" class="secondary" type="button">One more pass</button>
            <span class="spinner" hidden></span>
            <span id="teach-status" class="teach-status" hidden></span>
            <button id="pause" class="secondary" type="button" hidden>Pause</button>
          </div>
        </div>
        <div id="infer-run" hidden>
          <h2>Inference</h2>
          <p class="lede">
            Every row is scored, including rows held out of training.
          </p>
          <div class="row">
            <button id="score" type="button">Score this table</button>
            <span class="spinner" hidden></span>
          </div>
          <div id="pred" class="grid"></div>
        </div>
        <h2>Signal</h2>
        <div class="legend">
          <span><i class="swatch" style="background:#0d9488"></i>forward +</span>
          <span><i class="swatch" style="background:#2563eb"></i>forward -</span>
          <span><i class="swatch" style="background:#d97706"></i>backward +</span>
          <span><i class="swatch" style="background:#e11d48"></i>backward -</span>
        </div>
        <div id="stage-net">
          <canvas id="net-canvas" hidden></canvas>
          <svg id="net" viewBox="0 0 720 420"
            preserveAspectRatio="xMidYMid meet"></svg>
          <input id="net-zoom" type="range" min="1" max="5" step="0.05"
            value="1" aria-label="Zoom"/>
        </div>
        <p id="caption">Load a model that can play a film.</p>
        <div class="row">
          <button id="play" class="secondary" type="button">Play</button>
          <button id="back" class="secondary" type="button">
            Show this row forward and back
          </button>
          <label>Row
            <input id="row" type="number" value="0" min="0"/>
          </label>
          <span class="scale-name">Scale</span>
          <div class="segment">
            <label>
              <input type="radio" name="scale" value="approx" checked/>
              Approximate
            </label>
            <label>
              <input type="radio" name="scale" value="real"/>
              Real
            </label>
          </div>
        </div>
        <div id="check-box" hidden>
          <h2>Held-out rows</h2>
          <p class="lede">
            Test images when the file has them, otherwise the rows kept
            back from teaching.
          </p>
          <div class="row">
            <button id="check" class="secondary" type="button">
              Score these rows
            </button>
            <span id="check-score"></span>
          </div>
          <div id="check-table" class="grid"></div>
          <div class="row pager">
            <button id="check-prev" class="secondary" type="button">Previous</button>
            <span id="check-page"></span>
            <button id="check-next" class="secondary" type="button">Next</button>
          </div>
        </div>
      </div>
      <div>
        <h2>What the model reports</h2>
        <pre id="state">Nothing fitted yet.</pre>
        <div class="row">
          <a href="/api/download/report">
            <button class="secondary" type="button">Report</button>
          </a>
          <a href="/api/download/model">
            <button class="secondary" type="button">Whole model</button>
          </a>
          <a href="/api/download/weights">
            <button class="secondary" type="button">Weights (.ttw)</button>
          </a>
        </div>
      </div>
    </div>
  </section>
  <section id="report-block" class="panel" hidden>
    <h2>Report</h2>
    <iframe id="report-frame" title="Report"></iframe>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
let playback = null;
let frame = 0;
let timer = null;
let layoutSizes = null;
let pulseFrame = 0;
let teachPaused = false;
let teachTick = 0;
let sweep = null;
let sweepQueue = [];
let sweepHold = null;
let playedEval = false;
let scaleMode = "approx";
let netZoom = 1;
let netPanX = 0;
let netPanY = 0;
let netDrag = null;
let wireSheet = null;
let wireKey = "";
let progressTimer = 0;
let checkOffset = 0;
let active = null;
let settingsTab = "Network";
let session = null;
let pageOffset = 0;
let sourceKind = "paste";

const FIELD_COPY = {
  layers: {label: "Hidden layers"},
  hidden_layer_sizes: {label: "Neurons per layer"},
  activation: {label: "Activation"},
  hidden_activations: {label: "Hidden activations"},
  bias: {label: "Learned bias"},
  dropout: {label: "Dropout"},
  weight_init: {label: "Weight fill"},
  bias_init: {label: "Bias fill"},
  solver: {label: "Solver"},
  schedule: {label: "Schedule"},
  penalty: {label: "Penalty"},
  loss: {label: "Loss"},
  max_grad_norm: {label: "Gradient cap"},
  class_weight: {label: "Class weight"},
  max_iter: {label: "Epochs"},
  batch_size: {label: "Batch size"},
  shuffle: {label: "Shuffle rows"},
  tol: {label: "Tolerance"},
  n_iter_no_change: {label: "Quiet epochs"},
  early_stopping: {label: "Early stopping"},
  validation_fraction: {label: "Check fraction"},
  warm_start: {label: "Keep weights"},
  random_state: {label: "Seed"},
  inspect_every: {label: "Film every"},
  "activation__negative_slope": {label: "Negative slope"},
  "activation__alpha": {label: "ELU alpha"},
  "solver__lr": {label: "Step length"},
  "solver__beta_1": {label: "Gradient memory"},
  "solver__beta_2": {label: "Squared-gradient memory"},
  "solver__epsilon": {label: "Epsilon"},
  "solver__momentum": {label: "Momentum"},
  "solver__nesterov": {label: "Nesterov"},
  "solver__alpha": {label: "Squared-gradient memory"},
  "schedule__power_t": {label: "Decay power"},
  "schedule__lr_shrink": {label: "Step shrink"},
  "schedule__n_iter_no_change": {label: "Epochs before shrink"},
  "schedule__t_max": {label: "Cosine length"},
  "schedule__step_size": {label: "Decay every"},
  "schedule__gamma": {label: "Decay factor"},
  "penalty__alpha": {label: "Penalty strength"},
  "penalty__l1_ratio": {label: "L1 share"},
  "loss__delta": {label: "Huber width"},
  n_components: {label: "Components"},
  copy: {label: "Edit a copy"},
  svd_solver: {label: "SVD"}
};

const FIELD_TIPS = {
  layers:
    "Each row is one hidden layer: a count of neurons, then its activation. " +
    "Input width is the feature columns. Output width is the class count, " +
    "or 1 for one target column. The output has no activation. " +
    "Remove every row to connect the input straight to the output.",
  hidden_layer_sizes:
    "Neurons, not layers. 100 is one layer of 100 neurons. " +
    "32,16 is two layers.",
  activation:
    "Name of the function after each hidden layer. Blank keeps ReLU. " +
    "The output layer stays linear.",
  hidden_activations:
    "One function name per hidden layer, in order. Not a neuron count. " +
    "Blank uses the single activation above.",
  bias: "On or off. A learned offset added before the activation.",
  dropout:
    "Fraction from 0 to 1, not a percent. 0.2 silences 20% of hidden " +
    "neurons on a training row. 0 keeps every neuron.",
  weight_init:
    "Name of the fill for weight matrices. " +
    "Blank picks a fill from the activation.",
  bias_init: "Name of the fill for bias vectors. Blank starts them at zero.",
  solver: "Name of the update rule. Blank is Adam, step length 0.001.",
  schedule: "Name of how the step length changes. Blank keeps it constant.",
  penalty:
    "Name of the extra cost on large weights. Biases are not penalized. " +
    "Blank is a small L2.",
  loss:
    "Name of what training minimizes. Blank is cross-entropy for a " +
    "classifier and squared error for a regressor.",
  max_grad_norm:
    "Cap on gradient length, in gradient units. Empty means no cap.",
  class_weight:
    "The word balanced, or empty. balanced makes rare classes count more.",
  max_iter:
    "Count of epochs. One epoch is one pass over the training rows.",
  batch_size:
    "Count of rows in one weight update. auto picks a count from the table.",
  shuffle: "On or off. Reorder training rows at the start of each epoch.",
  tol: "Smallest loss improvement that still counts, in loss units.",
  n_iter_no_change: "Count of quiet epochs allowed before the stop.",
  early_stopping:
    "On or off. Watch a held-out slice and stop when it stops improving.",
  validation_fraction:
    "Fraction from 0 to 1, not a percent. 0.1 holds back the last 10% " +
    "of training rows.",
  warm_start: "On or off. On keeps the weights. Off starts over.",
  random_state: "Integer seed for shuffling and dropout. Empty draws a new one.",
  inspect_every:
    "Count of epochs between stored film frames. 1 stores every epoch.",
  "activation__negative_slope":
    "Unitless slope on the negative side of leaky ReLU. 0.01 is a small leak.",
  "activation__alpha": "Unitless scale of the negative side of ELU.",
  "solver__lr":
    "Unitless step length. How far one update moves the weights.",
  "solver__beta_1":
    "Fraction from 0 to 1. How much Adam keeps of the past gradient.",
  "solver__beta_2":
    "Fraction from 0 to 1. How much Adam keeps of past squared gradients.",
  "solver__epsilon":
    "Tiny unitless number under the square root, so the step never " +
    "divides by zero.",
  "solver__momentum":
    "Fraction from 0 to 1. Share of the previous update added to this one.",
  "solver__nesterov":
    "On or off. Look ahead along the momentum before the gradient.",
  "solver__alpha":
    "Fraction from 0 to 1. How much RMSprop keeps of old squared gradients.",
  "schedule__power_t":
    "Unitless exponent. The step is the epoch raised to minus this.",
  "schedule__lr_shrink":
    "Fraction. The adaptive schedule multiplies the step by this.",
  "schedule__n_iter_no_change":
    "Count of quiet epochs before the adaptive schedule shrinks the step.",
  "schedule__t_max": "Count of epochs. The cosine step reaches 0 here.",
  "schedule__step_size": "Count of epochs between step-decay drops.",
  "schedule__gamma":
    "Fraction. Step decay multiplies by it at each drop. " +
    "Exponential multiplies by it every epoch.",
  "penalty__alpha": "Unitless strength of the weight penalty. 0 turns it off.",
  "penalty__l1_ratio":
    "Fraction from 0 to 1, not a percent. 0 is pure squared, 1 is pure " +
    "absolute.",
  "loss__delta":
    "Distance in target units. Past it, Huber uses absolute error.",
  n_components:
    "Count of directions, or a fraction from 0 to 1 of the variance. " +
    "Empty keeps the smaller side of the table.",
  copy: "On or off. Off edits the array you passed in.",
  svd_solver: "Name of the decomposition. full is the dense SVD."
};

const FIELD_GROUPS = [
  ["Network", [
    "layers", "hidden_layer_sizes", "activation", "hidden_activations",
    "bias", "dropout"
  ]],
  ["Weights", ["weight_init", "bias_init"]],
  ["Learning", [
    "solver", "schedule", "penalty", "loss", "max_grad_norm", "class_weight"
  ]],
  ["Loop", [
    "max_iter", "batch_size", "shuffle", "tol", "n_iter_no_change",
    "early_stopping", "validation_fraction", "warm_start",
    "random_state", "inspect_every"
  ]]
];

function groupOf(key) {
  const root = key.split("__")[0];
  for (const [name, keys] of FIELD_GROUPS) {
    if (keys.includes(root)) return name;
  }
  return "Other";
}

function color(phase, value) {
  if (phase === "backward") return value >= 0 ? "#d97706" : "#e11d48";
  return value >= 0 ? "#0d9488" : "#2563eb";
}

function outputGlow(beat, nLayers) {
  if (!beat || beat.kind !== "neurons" || beat.phase !== "forward") return null;
  if (beat.layer !== nLayers - 1) return null;
  const values = beat.values || [];
  if (!values.length) return null;
  let max = values[0];
  let winner = 0;
  for (let i = 1; i < values.length; i++) {
    if (values[i] > max) {
      max = values[i];
      winner = i;
    }
  }
  let sum = 0;
  const weights = values.map((value) => {
    const raised = Math.exp(Math.max(-40, value - max));
    sum += raised;
    return raised;
  });
  return {winner: winner, weights: weights, sum: sum};
}

async function api(path, body) {
  const response = await fetch(path, body ? {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  } : undefined);
  const data = await response.json();
  if (!response.ok) {
    $("notice").textContent = data.error || "request failed";
    throw new Error(data.error || "request failed");
  }
  $("notice").textContent = "";
  return data;
}

function showScreen(name) {
  ["home", "data", "model", "run"].forEach((id) => {
    $("screen-" + id).hidden = id !== name;
  });
  const home = name === "home";
  document.body.classList.toggle("stage", name === "run");
  if (name === "run") showLayout();
  else $("report-block").hidden = true;
  $("steps").hidden = home;
  $("back-sessions").hidden = home;
  document.querySelectorAll("#steps button").forEach((button) => {
    const on = button.dataset.screen === name;
    if (on) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
}

function purposeName(purpose) {
  return purpose === "infer" ? "Inference" : "Train";
}

function applySession() {
  const infer = session.purpose === "infer";
  $("session-label").textContent = session.name + " · " + purposeName(session.purpose);
  $("step-run").textContent = purposeName(session.purpose);
  $("divisor").value = session.scale_divisor == null ? "" : session.scale_divisor;
  $("center").checked = !!session.standardize;
  $("holdout").value = session.val_fraction || 0;
  $("train-model").hidden = infer;
  $("infer-model").hidden = !infer;
  $("train-run").hidden = infer;
  $("infer-run").hidden = !infer;
  showHeldOut();
  if (session.model) $("weight-note").textContent = session.model;
}

function showHeldOut() {
  const rows = session ? (session.val || 0) + (session.test || 0) : 0;
  const infer = session && session.purpose === "infer";
  $("check-box").hidden = !session || infer || rows === 0;
}

function shownCounts(sizes) {
  return sizes.map((count) => Math.min(16, count));
}

function showNet(real) {
  const svg = $("net");
  svg.toggleAttribute("hidden", real);
  $("net-canvas").toggleAttribute("hidden", !real);
  if (real) svg.replaceChildren();
}

function dotRadius(count) {
  if (count <= 1) return 11;
  const gap = 340 / (count - 1);
  return Math.max(0.45, Math.min(11, gap * 0.42));
}

function beatMax(beat) {
  let maxAbs = 1e-9;
  if (!beat) return maxAbs;
  beat.values.forEach((value) => {
    maxAbs = Math.max(maxAbs, Math.abs(value));
  });
  beat.edges.forEach((edge) => {
    maxAbs = Math.max(maxAbs, Math.abs(edge[2]));
  });
  return maxAbs;
}

function netFrame() {
  const wrap = $("stage-net");
  const width = wrap.clientWidth || 720;
  const height = wrap.clientHeight || 420;
  const fit = Math.min(width / 720, height / 420) || 1;
  return {width: width, height: height, fit: fit};
}

function clampPan() {
  const frame = netFrame();
  const span = Math.max(0, netZoom - 1);
  const maxX = 720 * frame.fit * span / 2;
  const maxY = 420 * frame.fit * span / 2;
  netPanX = Math.max(-maxX, Math.min(maxX, netPanX));
  netPanY = Math.max(-maxY, Math.min(maxY, netPanY));
}

function viewOrigin() {
  clampPan();
  const frame = netFrame();
  const viewW = 720 / netZoom;
  const viewH = 420 / netZoom;
  return {
    x: (720 - viewW) / 2 - netPanX / frame.fit,
    y: (420 - viewH) / 2 - netPanY / frame.fit,
    w: viewW,
    h: viewH,
    frame: frame
  };
}

function placeNet(ctx, dpr, width, height) {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  const fit = Math.min(width / 720, height / 420) || 1;
  const viewW = 720 / netZoom;
  const viewH = 420 / netZoom;
  const scale = dpr * fit * netZoom;
  const viewX = (720 - viewW) / 2 - netPanX / fit;
  const viewY = (420 - viewH) / 2 - netPanY / fit;
  ctx.setTransform(
    scale, 0, 0, scale,
    dpr * (width - viewW * fit * netZoom) / 2 - viewX * scale,
    dpr * (height - viewH * fit * netZoom) / 2 - viewY * scale
  );
}

function applyNetView() {
  const box = viewOrigin();
  $("net").setAttribute(
    "viewBox",
    box.x + " " + box.y + " " + box.w + " " + box.h
  );
  if (scaleMode !== "real" || pulseFrame) return;
  if (playback && playback.frames && playback.frames.length) {
    paintReal(playback.layer_sizes, playback.frames[frame]);
  } else if (layoutSizes && layoutSizes.length) {
    paintReal(layoutSizes, null);
  }
}

function wireSheetFor(sizes, pos) {
  const key = sizes.join(",");
  if (wireSheet && wireKey === key) return wireSheet;
  const sheet = document.createElement("canvas");
  const dpr = 2;
  sheet.width = 720 * dpr;
  sheet.height = 420 * dpr;
  const ctx = sheet.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  for (let layer = 0; layer < pos.length - 1; layer++) {
    const src = pos[layer];
    const dst = pos[layer + 1];
    const count = src.length * dst.length;
    ctx.beginPath();
    ctx.strokeStyle = count > 4000 ? "#314158" : "#93a4b8";
    ctx.lineWidth = count > 4000 ? 0.4 : 1.2;
    ctx.globalAlpha = count > 4000 ? 0.05 : 0.8;
    for (let i = 0; i < src.length; i++) {
      for (let j = 0; j < dst.length; j++) {
        ctx.moveTo(src[i][0], src[i][1]);
        ctx.lineTo(dst[j][0], dst[j][1]);
      }
    }
    ctx.stroke();
  }
  wireSheet = sheet;
  wireKey = key;
  return sheet;
}

function paintReal(sizes, beat) {
  const canvas = $("net-canvas");
  const wrap = $("stage-net");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = wrap.clientWidth || 720;
  const height = wrap.clientHeight || 420;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  placeNet(ctx, dpr, width, height);
  const pos = layerPositions(sizes);
  ctx.drawImage(wireSheetFor(sizes, pos), 0, 0, 720, 420);
  const maxAbs = beatMax(beat);
  if (beat && beat.kind === "wires") {
    beat.edges.forEach(([src, dst, value]) => {
      const start = pos[beat.layer - 1][src];
      const end = pos[beat.layer][dst];
      if (!start || !end) return;
      ctx.globalAlpha = 1;
      ctx.strokeStyle = color(beat.phase, value);
      ctx.lineWidth = 1 + 3 * Math.abs(value) / maxAbs;
      ctx.beginPath();
      ctx.moveTo(start[0], start[1]);
      ctx.lineTo(end[0], end[1]);
      ctx.stroke();
    });
  }
  const glow = outputGlow(beat, pos.length);
  pos.forEach((points, layer) => {
    const radius = dotRadius(points.length);
    points.forEach(([x, y], index) => {
      let fill = "#475569";
      let dot = radius;
      ctx.globalAlpha = 1;
      if (glow && beat.layer === layer && index === glow.winner) {
        ctx.fillStyle = "#f8fafc";
        ctx.beginPath();
        ctx.arc(x, y, Math.max(radius * 2.8, 8), 0, Math.PI * 2);
        ctx.fill();
        fill = "#0d9488";
        dot = Math.max(radius * 1.7, 5);
      } else if (glow && beat.layer === layer) {
        const weight = glow.weights[index] || 0;
        const share = glow.sum ? weight / glow.sum : 0;
        fill = color(beat.phase, beat.values[index] || 0);
        ctx.globalAlpha = 0.15 + 0.5 * share;
      } else if (beat && beat.kind === "neurons" && beat.layer === layer) {
        const value = beat.values[index] || 0;
        fill = color(beat.phase, value);
        ctx.globalAlpha = 0.35 + 0.65 * Math.abs(value) / maxAbs;
      }
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.arc(x, y, dot, 0, Math.PI * 2);
      ctx.fill();
    });
  });
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#93a4b8";
  ctx.font = "13px Segoe UI, sans-serif";
  ctx.textAlign = "center";
  sizes.forEach((count, layer) => {
    if (!pos[layer].length) return;
    ctx.fillText(String(count), pos[layer][0][0], 22);
  });
}

function layerPositions(counts) {
  const width = 720;
  const height = 420;
  const gap = width / (counts.length + 1);
  return counts.map((count, layer) => {
    const points = [];
    for (let i = 0; i < count; i++) {
      const y = 40 + (height - 80) * (count === 1 ? 0.5 : i / (count - 1));
      points.push([gap * (layer + 1), y]);
    }
    return points;
  });
}

function drawStructure(sizes) {
  if (scaleMode === "real") {
    showNet(true);
    paintReal(sizes, null);
    return;
  }
  showNet(false);
  const svg = $("net");
  svg.innerHTML = "";
  const pos = layerPositions(shownCounts(sizes));
  for (let layer = 0; layer < pos.length - 1; layer++) {
    pos[layer].forEach((start) => {
      pos[layer + 1].forEach((end) => {
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", start[0]);
        line.setAttribute("y1", start[1]);
        line.setAttribute("x2", end[0]);
        line.setAttribute("y2", end[1]);
        line.setAttribute("stroke", "#314158");
        line.setAttribute("stroke-width", "1.25");
        line.dataset.wire = "1";
        svg.appendChild(line);
      });
    });
  }
  pos.forEach((points) => {
    points.forEach(([x, y]) => {
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", x);
      circle.setAttribute("cy", y);
      circle.setAttribute("r", "11");
      circle.setAttribute("fill", "#475569");
      svg.appendChild(circle);
    });
  });
  drawCounts(svg, sizes, pos);
}

function drawCounts(svg, sizes, pos) {
  sizes.forEach((count, layer) => {
    if (!pos[layer] || !pos[layer].length) return;
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", pos[layer][0][0]);
    text.setAttribute("y", "22");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("fill", "#93a4b8");
    text.setAttribute("font-size", "13");
    text.setAttribute("class", "layer-count");
    text.textContent = String(count);
    svg.appendChild(text);
  });
}

function pulseColor(back, y) {
  const positive = y < 210;
  if (back) return positive ? "#d97706" : "#e11d48";
  return positive ? "#0d9488" : "#2563eb";
}

function sweepHot(along, cursor) {
  const lead = (along - cursor + 1) % 1;
  return lead < 0.22;
}

function appendPhase(phase) {
  const queued = sweepQueue.length ? sweepQueue[sweepQueue.length - 1] : "";
  const tail = queued || (sweep ? sweep.phase : "");
  if (phase === "backpropagation" && tail !== "feedforward") {
    sweepQueue.push("feedforward");
  }
  const last = sweepQueue.length ? sweepQueue[sweepQueue.length - 1] : tail;
  if (last !== phase) sweepQueue.push(phase);
  while (sweepQueue.length > 2) sweepQueue.shift();
  if (!sweep) beginNext();
}

function beginNext() {
  const phase = sweepQueue.shift();
  if (!phase) {
    sweep = null;
    return;
  }
  sweepHold = null;
  sweep = {phase: phase, start: performance.now(), duration: 520};
}

function notePhase(tick, phase) {
  if (phase === "evaluate") {
    if (playedEval) return;
    playedEval = true;
    appendPhase("feedforward");
    return;
  }
  playedEval = false;
  if (phase !== "feedforward" && phase !== "backpropagation") return;
  if (tick === teachTick) return;
  teachTick = tick;
  appendPhase(phase);
}

function sweepMotion(now) {
  if (!sweep) return sweepHold;
  const t = (now - sweep.start) / sweep.duration;
  if (t >= 1) {
    sweepHold = {
      phase: sweep.phase,
      at: sweep.phase === "backpropagation" ? 0 : 1
    };
    beginNext();
    return sweepMotion(now);
  }
  const at = sweep.phase === "backpropagation" ? 1 - t : t;
  return {phase: sweep.phase, at: at};
}

function sweepWires(now) {
  const motion = sweepMotion(now);
  const back = !!motion && motion.phase === "backpropagation";
  const cursor = motion ? motion.at : -1;
  $("net").querySelectorAll("line[data-wire]").forEach((line) => {
    const x = Number(line.getAttribute(back ? "x2" : "x1"));
    const y1 = Number(line.getAttribute("y1"));
    const y2 = Number(line.getAttribute("y2"));
    const hot = motion && sweepHot(x / 720, cursor);
    const mid = (y1 + y2) / 2;
    line.setAttribute("stroke", hot ? pulseColor(back, mid) : "#314158");
    line.setAttribute("stroke-width", hot ? "2.4" : "1.25");
  });
  $("net").querySelectorAll("circle").forEach((circle) => {
    const along = Number(circle.getAttribute("cx")) / 720;
    const y = Number(circle.getAttribute("cy"));
    const hot = motion && sweepHot(along, cursor);
    circle.setAttribute("fill", hot ? pulseColor(back, y) : "#475569");
    circle.setAttribute("r", hot ? "13" : "11");
  });
}

function paintColumn(ctx, points, back) {
  if (!points.length) return;
  const x = points[0][0];
  const y0 = points[0][1];
  const y1 = points[points.length - 1][1];
  const mid = (y0 + y1) / 2;
  ctx.globalAlpha = 1;
  ctx.lineWidth = 7;
  ctx.lineCap = "round";
  ctx.strokeStyle = pulseColor(back, y0);
  ctx.beginPath();
  ctx.moveTo(x, y0);
  ctx.lineTo(x, mid);
  ctx.stroke();
  ctx.strokeStyle = pulseColor(back, y1);
  ctx.beginPath();
  ctx.moveTo(x, mid);
  ctx.lineTo(x, y1);
  ctx.stroke();
  const radius = Math.max(dotRadius(points.length) * 1.8, 2.6);
  points.forEach(([px, py]) => {
    ctx.globalAlpha = 1;
    ctx.fillStyle = pulseColor(back, py);
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fill();
  });
}

function sweepReal(now) {
  if (!layoutSizes || !layoutSizes.length) return;
  const canvas = $("net-canvas");
  if (canvas.hidden || !canvas.width) return;
  const wrap = $("stage-net");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = wrap.clientWidth || 720;
  const height = wrap.clientHeight || 420;
  const ctx = canvas.getContext("2d");
  placeNet(ctx, dpr, width, height);
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
  const pos = layerPositions(layoutSizes);
  ctx.drawImage(wireSheetFor(layoutSizes, pos), 0, 0, 720, 420);
  const motion = sweepMotion(now);
  const back = !!motion && motion.phase === "backpropagation";
  pos.forEach((points) => {
    const along = points.length ? points[0][0] / 720 : 0;
    const hot = motion && sweepHot(along, motion.at);
    if (hot) {
      paintColumn(ctx, points, back);
      return;
    }
    const radius = dotRadius(points.length);
    points.forEach(([x, y]) => {
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#475569";
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    });
  });
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#93a4b8";
  ctx.font = "13px Segoe UI, sans-serif";
  ctx.textAlign = "center";
  layoutSizes.forEach((count, layer) => {
    if (!pos[layer].length) return;
    ctx.fillText(String(count), pos[layer][0][0], 22);
  });
}

function startPulse() {
  stopPulse();
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  let last = 0;
  const tick = (now) => {
    if (teachPaused && sweep && last) sweep.start += now - last;
    else if (!teachPaused) {
      if (scaleMode === "real") sweepReal(now);
      else sweepWires(now);
    }
    last = now;
    pulseFrame = requestAnimationFrame(tick);
  };
  pulseFrame = requestAnimationFrame(tick);
}

function stopPulse() {
  $("net-canvas").classList.remove("live");
  if (!pulseFrame) return;
  cancelAnimationFrame(pulseFrame);
  pulseFrame = 0;
}

function watchProgress() {
  stopWatch();
  teachPaused = false;
  teachTick = 0;
  sweep = null;
  sweepQueue = [];
  sweepHold = null;
  playedEval = false;
  const node = $("teach-status");
  node.hidden = false;
  node.textContent = "Epoch 0 · feedforward";
  $("pause").hidden = false;
  $("pause").textContent = "Pause";
  progressTimer = setInterval(async () => {
    const ticket = progressTimer;
    try {
      const response = await fetch("/api/progress");
      if (progressTimer !== ticket) return;
      const data = await response.json();
      const phase = data.phase || "feedforward";
      notePhase(Number(data.tick) || 0, phase);
      teachPaused = !!data.paused;
      const pause = $("pause");
      pause.hidden = !data.running;
      pause.textContent = data.paused ? "Resume" : "Pause";
      const spinner = document.querySelector("#train-run .spinner");
      if (spinner) spinner.hidden = !data.running || !!data.paused;
      node.textContent = data.paused
        ? ("Paused · epoch " + data.epoch + " / " + data.epochs)
        : ("Epoch " + data.epoch + " / " + data.epochs + " · " + phase);
    } catch (err) {
      /* The fit request is the one that reports a failure. */
    }
  }, 150);
}

function stopWatch() {
  teachPaused = false;
  if (progressTimer) clearInterval(progressTimer);
  progressTimer = 0;
  $("teach-status").hidden = true;
  $("pause").hidden = true;
}

function setBusy(on) {
  document.querySelectorAll(".spinner").forEach((node) => {
    node.hidden = !on;
  });
  ["fit", "step", "score"].forEach((id) => {
    $(id).disabled = on;
  });
}

async function showLayout() {
  if (!session) return;
  let data;
  try {
    data = await api("/api/layout");
  } catch (err) {
    return;
  }
  layoutSizes = data.sizes || [];
  if (playback && playback.frames && playback.frames.length) {
    draw();
  } else if (layoutSizes.length) {
    playback = null;
    drawStructure(layoutSizes);
    const widths = layoutSizes.join(" → ");
    const scaleNote = scaleMode === "real"
      ? "Every neuron is drawn at this scale."
      : "The first 16 of a wide layer are drawn.";
    $("caption").textContent = "Configured network, " + widths
      + " neurons. " + scaleNote;
  } else {
    $("net").innerHTML = "";
    $("caption").textContent = data.note || "This model has no neuron film.";
  }
  await refreshState();
}

async function loadFilm(backward) {
  playback = await api("/api/playback", {
    sample_index: Number($("row").value || 0),
    with_backward: backward
  });
  frame = 0;
  draw();
}

function startPlay(once) {
  if (timer) clearInterval(timer);
  if (!playback || !playback.frames.length) return;
  timer = setInterval(() => {
    if (once && frame >= playback.frames.length - 1) {
      clearInterval(timer);
      timer = null;
      return;
    }
    frame = (frame + 1) % playback.frames.length;
    draw();
  }, 700);
}

async function playForward(index) {
  if (timer) { clearInterval(timer); timer = null; }
  $("row").value = index;
  await loadFilm(false);
  startPlay(true);
}

async function teach(mode) {
  setBusy(true);
  watchProgress();
  try {
    if (!layoutSizes || !layoutSizes.length) await showLayout();
    startPulse();
    $("caption").textContent = "Teaching. The signal plays when this pass finishes.";
    await api("/api/params", {fields: collectFields()});
    await api("/api/fit", {mode: mode});
    stopPulse();
    await refreshState();
    try {
      await loadFilm(true);
      startPlay();
    } catch (err) {
      playback = null;
      if (layoutSizes && layoutSizes.length) drawStructure(layoutSizes);
    }
  } catch (err) {
    /* The request already wrote the reason into the notice. */
  } finally {
    stopPulse();
    stopWatch();
    setBusy(false);
  }
}

function draw() {
  if (!playback || !playback.frames.length) {
    if (layoutSizes && layoutSizes.length) drawStructure(layoutSizes);
    return;
  }
  const beat = playback.frames[frame];
  $("caption").textContent = beat.caption
    + " (" + (frame + 1) + "/" + playback.frames.length + ")";
  if (scaleMode === "real") {
    showNet(true);
    paintReal(playback.layer_sizes, beat);
    return;
  }
  showNet(false);
  const svg = $("net");
  svg.innerHTML = "";
  const pos = layerPositions(shownCounts(playback.layer_sizes));
  const maxAbs = Math.max(
    1e-9,
    ...beat.values.slice(0, 16).map(Math.abs),
    ...beat.edges.map((edge) => Math.abs(edge[2]))
  );
  if (beat.kind === "wires") {
    beat.edges.forEach(([src, dst, value]) => {
      const a = pos[beat.layer - 1][src];
      const b = pos[beat.layer][dst];
      if (!a || !b) return;
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", a[0]); line.setAttribute("y1", a[1]);
      line.setAttribute("x2", b[0]); line.setAttribute("y2", b[1]);
      line.setAttribute("stroke", color(beat.phase, value));
      line.setAttribute("stroke-width", String(1 + 4 * Math.abs(value) / maxAbs));
      svg.appendChild(line);
    });
  }
  const glow = outputGlow(beat, playback.layer_sizes.length);
  pos.forEach((points, layer) => {
    points.forEach(([x, y], index) => {
      const won = glow && beat.layer === layer && index === glow.winner;
      if (won) {
        const halo = document.createElementNS(
          "http://www.w3.org/2000/svg", "circle"
        );
        halo.setAttribute("cx", x);
        halo.setAttribute("cy", y);
        halo.setAttribute("r", "18");
        halo.setAttribute("fill", "#f8fafc");
        svg.appendChild(halo);
      }
      const circle = document.createElementNS(
        "http://www.w3.org/2000/svg", "circle"
      );
      circle.setAttribute("cx", x);
      circle.setAttribute("cy", y);
      circle.setAttribute("r", won ? "13" : "11");
      let fill = "#475569";
      if (won) {
        fill = "#0d9488";
        circle.setAttribute("fill-opacity", "1");
      } else if (glow && beat.layer === layer) {
        const weight = glow.weights[index] || 0;
        const share = glow.sum ? weight / glow.sum : 0;
        fill = color(beat.phase, beat.values[index] || 0);
        circle.setAttribute("fill-opacity", String(0.15 + 0.5 * share));
      } else if (beat.kind === "neurons" && beat.layer === layer) {
        const value = beat.values[index] || 0;
        fill = color(beat.phase, value);
        circle.setAttribute(
          "fill-opacity",
          String(0.35 + 0.65 * Math.abs(value) / maxAbs)
        );
      }
      circle.setAttribute("fill", fill);
      svg.appendChild(circle);
    });
  });
  drawCounts(svg, playback.layer_sizes, pos);
}

function renderFields(fields) {
  const form = $("fields");
  const tabs = $("tabs");
  form.innerHTML = "";
  tabs.innerHTML = "";
  const buckets = new Map();
  fields.forEach((field) => {
    const name = groupOf(field.key);
    if (!buckets.has(name)) buckets.set(name, []);
    buckets.get(name).push(field);
  });
  const rank = new Map();
  FIELD_GROUPS.forEach((group) => {
    group[1].forEach((key, index) => rank.set(key, index));
  });
  const order = FIELD_GROUPS.map((group) => group[0])
    .concat(["Other"])
    .filter((name) => buckets.has(name));
  if (!order.includes(settingsTab)) settingsTab = order[0] || "Network";
  if (order.length > 1) {
    order.forEach((name) => {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.className = "tab" + (name === settingsTab ? " active" : "");
      tab.dataset.tab = name;
      tab.textContent = name;
      tab.onclick = () => showTab(name);
      tabs.appendChild(tab);
    });
  }
  order.forEach((name) => {
    const pane = document.createElement("div");
    pane.dataset.pane = name;
    pane.hidden = order.length > 1 && name !== settingsTab;
    const ranked = buckets.get(name).slice().sort((left, right) =>
      (rank.get(left.key) ?? 99) - (rank.get(right.key) ?? 99)
    );
    ranked.forEach((field) => {
      const node = field.kind === "layers" ? layerEditor(field) : fieldControl(field);
      pane.appendChild(node);
    });
    form.appendChild(pane);
  });
}

function layerEditor(field) {
  const wrap = document.createElement("div");
  wrap.className = "field layers";
  const head = document.createElement("div");
  head.className = "layers-head";
  const label = document.createElement("label");
  label.textContent = "Hidden layers";
  head.append(label, tipButton("layers"));
  wrap.appendChild(head);
  const list = document.createElement("div");
  list.className = "layer-list";
  wrap.appendChild(list);
  let empty = null;
  function paint() {
    [...list.querySelectorAll(".layer-row")].forEach((row, index) => {
      row.querySelector(".layer-no").textContent = String(index + 1);
    });
    if (list.children.length) {
      if (empty) { empty.remove(); empty = null; }
      return;
    }
    if (empty) return;
    empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "No hidden layer. The input connects straight to the output.";
    list.after(empty);
  }
  function addRow(neurons, activation) {
    const row = document.createElement("div");
    row.className = "layer-row";
    const number = document.createElement("span");
    number.className = "layer-no";
    const count = document.createElement("input");
    count.type = "number";
    count.min = "1";
    count.step = "1";
    count.value = String(neurons);
    count.setAttribute("aria-label", "Neurons");
    const select = document.createElement("select");
    select.setAttribute("aria-label", "Activation");
    (field.options || []).forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      if (name === activation) option.selected = true;
      select.appendChild(option);
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "secondary layer-remove";
    remove.textContent = "Remove";
    remove.onclick = () => { row.remove(); paint(); };
    row.append(number, count, select, remove);
    list.appendChild(row);
    dress(select);
    paint();
  }
  (field.value || []).forEach((row) => addRow(row.neurons, row.activation));
  paint();
  const add = document.createElement("button");
  add.type = "button";
  add.className = "secondary";
  add.textContent = "Add hidden layer";
  add.onclick = () => {
    const last = list.querySelector(".layer-row:last-child select");
    addRow(32, last ? last.value : "ReLU");
  };
  wrap.appendChild(add);
  return wrap;
}

function fieldCopy(field) {
  if (FIELD_COPY[field.key]) return FIELD_COPY[field.key];
  const tail = field.key.split("__").pop().replaceAll("_", " ");
  return {label: tail.charAt(0).toUpperCase() + tail.slice(1)};
}

function fieldTip(key) {
  return FIELD_TIPS[key] || ("Setting " + key + ".");
}

function tipButton(key) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "tip";
  button.textContent = "i";
  button.setAttribute("aria-label", key);
  const bubble = document.createElement("span");
  bubble.className = "bubble";
  bubble.textContent = fieldTip(key);
  button.appendChild(bubble);
  button.addEventListener("mouseenter", () => {
    const room = button.getBoundingClientRect().top;
    button.classList.toggle("down", room < 140);
  });
  return button;
}

function fieldControl(field) {
  const wrap = document.createElement("div");
  const copy = fieldCopy(field);
  const label = document.createElement("label");
  let input;
  if (field.kind === "bool") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = !!field.value;
    const track = document.createElement("span");
    track.className = "track";
    const caption = document.createElement("span");
    caption.textContent = copy.label;
    label.className = "switch";
    label.append(input, track, caption);
  } else if (field.kind === "component") {
    input = document.createElement("select");
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = field.placeholder
      ? "default (" + field.placeholder + ")"
      : "default";
    input.appendChild(blank);
    field.options.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      if (name === field.value) option.selected = true;
      input.appendChild(option);
    });
    input.addEventListener("change", () => refreshComponent(input.dataset.key));
  } else if (field.kind === "ints") {
    input = document.createElement("input");
    input.value = (field.value || []).join(",");
  } else {
    input = document.createElement("input");
    input.value = field.value === null ? "" : field.value;
  }
  input.dataset.key = field.key;
  input.dataset.kind = field.kind;
  const control = document.createElement("div");
  control.className = "control";
  if (field.kind === "bool") {
    control.appendChild(label);
  } else {
    label.textContent = copy.label;
    wrap.appendChild(label);
    control.appendChild(input);
  }
  control.appendChild(tipButton(field.key));
  wrap.className = "field";
  wrap.appendChild(control);
  if (input.tagName === "SELECT") dress(input);
  return wrap;
}

function showTab(name) {
  settingsTab = name;
  document.querySelectorAll("#tabs button").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
  document.querySelectorAll("#fields [data-pane]").forEach((pane) => {
    pane.hidden = pane.dataset.pane !== name;
  });
}

function collectFields() {
  const fields = [...$("fields").querySelectorAll("input, select")]
    .filter((input) => !input.closest(".layers"))
    .map((input) => ({
      key: input.dataset.key,
      kind: input.dataset.kind,
      value: input.dataset.kind === "bool" ? input.checked : input.value
    }));
  const editor = $("fields").querySelector(".layers");
  if (!editor) return fields;
  const rows = [...editor.querySelectorAll(".layer-row")].map((row) => ({
    neurons: Number(row.querySelector("input").value),
    activation: row.querySelector("select").value
  }));
  fields.push({key: "layers", kind: "layers", value: rows});
  return fields;
}

async function refreshComponent(key) {
  const prefix = key + "__";
  [...$("fields").querySelectorAll("input, select")].forEach((el) => {
    if (el.dataset.key && el.dataset.key.startsWith(prefix)) el.remove();
  });
  const schema = await api("/api/params", {fields: collectFields()});
  renderFields(schema.fields);
}

async function refreshState() {
  const state = await api("/api/state");
  const lines = [];
  if (state.summary) lines.push(state.summary);
  (state.diagnostics || []).forEach((item) => {
    lines.push(item.severity + " " + item.code + ": " + item.message);
  });
  $("state").textContent = lines.join("\\n") || "Nothing fitted yet.";
  const block = $("report-block");
  if (!state.fitted || $("screen-run").hidden) {
    block.hidden = true;
    return;
  }
  block.hidden = false;
  $("report-frame").src = "/api/report?t=" + Date.now();
}

function fitReportFrame() {
  const frame = $("report-frame");
  const doc = frame.contentDocument;
  if (!doc || !doc.documentElement) return;
  frame.style.height = doc.documentElement.scrollHeight + "px";
}

function note(data) {
  if (session) {
    session.val = data.val || 0;
    session.test = data.test || 0;
    showHeldOut();
  }
  const bits = [
    data.rows + " rows",
    data.features + " features",
    "train " + data.train,
    "val " + data.val,
    "test " + data.test
  ];
  if (data.scale_divisor != null) bits.push("divided by " + data.scale_divisor);
  if (data.standardize) bits.push("centered");
  $("data-note").textContent = bits.join(", ") + ".";
}

function pageLimit() {
  const picked = Number($("page-size").value);
  return picked > 0 ? picked : 8;
}

async function showPage() {
  const query = "/api/rows?offset=" + pageOffset + "&limit=" + pageLimit();
  const page = await api(query);
  if (pageOffset > 0 && pageOffset >= page.rows) {
    pageOffset = Math.max(0, page.rows - pageLimit());
    return showPage();
  }
  paintTable($("preview"), page, false);
  const start = page.records.length ? page.offset + 1 : 0;
  const end = page.offset + page.records.length;
  $("page-label").textContent = start + "-" + end + " of " + page.rows;
}

function paintTable(box, page, scored) {
  box.innerHTML = "";
  if (!page.records.length) return;
  const scroller = document.createElement("div");
  scroller.className = "grid-scroll";
  const first = page.records[0];
  const shape = page.image_shape;
  const table = document.createElement("table");
  const head = document.createElement("tr");
  const titles = ["#"];
  if (first.pixels) titles.push("frame");
  else titles.push.apply(titles, first.names || []);
  if (page.target || first.target !== undefined) titles.push(page.target || "target");
  if (scored) titles.push("prediction");
  if (scored && first.match !== undefined) titles.push("match");
  titles.forEach((name) => {
    const cell = document.createElement("th");
    cell.textContent = name;
    head.appendChild(cell);
  });
  table.appendChild(head);
  page.records.forEach((record) => {
    const row = document.createElement("tr");
    const index = document.createElement("td");
    index.className = "num";
    index.textContent = String(record.index);
    row.appendChild(index);
    if (record.pixels) {
      const frame = document.createElement("td");
      frame.className = "frame";
      frame.appendChild(paintFrame(record, shape));
      row.appendChild(frame);
    } else {
      (record.values || []).forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = String(value);
        row.appendChild(cell);
      });
    }
    if (record.target !== undefined) {
      const cell = document.createElement("td");
      cell.textContent = String(record.target);
      row.appendChild(cell);
    }
    if (scored) {
      const cell = document.createElement("td");
      cell.textContent = String(shown(record.prediction));
      row.appendChild(cell);
    }
    if (scored && record.match !== undefined) {
      const mark = document.createElement("td");
      mark.textContent = record.match ? "yes" : "no";
      row.appendChild(mark);
    }
    row.onclick = () => {
      $("row").value = record.index;
      if (scored) playForward(record.index);
    };
    table.appendChild(row);
  });
  scroller.appendChild(table);
  box.appendChild(scroller);
}

function shown(value) {
  if (Array.isArray(value)) return value.map((item) => String(item)).join(", ");
  return value;
}

function paintFrame(record, shape) {
  const canvas = document.createElement("canvas");
  const height = shape ? shape[0] : 1;
  const width = shape ? shape[1] : record.pixels.length;
  canvas.width = Math.max(1, width);
  canvas.height = Math.max(1, height);
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(canvas.width, canvas.height);
  const count = Math.min(record.pixels.length, canvas.width * canvas.height);
  for (let i = 0; i < count; i++) {
    const gray = 255 - record.pixels[i];
    image.data[i * 4] = gray;
    image.data[i * 4 + 1] = gray;
    image.data[i * 4 + 2] = gray;
    image.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

async function refreshSessions() {
  const data = await api("/api/sessions");
  const box = $("session-list");
  box.innerHTML = "";
  if (!data.sessions.length) {
    box.textContent = "No sessions yet.";
    return;
  }
  data.sessions.forEach((item) => box.appendChild(sessionRow(item)));
}

function sessionRow(item) {
  const row = document.createElement("div");
  row.className = "session";
  const text = document.createElement("div");
  const title = document.createElement("strong");
  title.textContent = item.name;
  const meta = document.createElement("p");
  meta.textContent = purposeName(item.purpose)
    + (item.has_data ? " · " + item.rows + " rows" : "")
    + (item.model ? " · " + item.model : "");
  text.appendChild(title);
  text.appendChild(meta);
  const open = document.createElement("button");
  open.type = "button";
  open.textContent = "Open";
  open.onclick = () => openSession(item.id);
  const drop = document.createElement("button");
  drop.type = "button";
  drop.className = "secondary";
  drop.textContent = "Drop";
  drop.onclick = () => dropSession(item.id);
  const actions = document.createElement("div");
  actions.className = "row";
  actions.appendChild(open);
  actions.appendChild(drop);
  row.appendChild(text);
  row.appendChild(actions);
  return row;
}

async function openSession(id) {
  session = await api("/api/sessions/open", {id: id});
  applySession();
  showScreen("data");
  if (session.has_data) {
    pageOffset = 0;
    await showPage();
  }
}

async function dropSession(id) {
  await api("/api/sessions/drop", {id: id});
  if (session && session.id === id) {
    session = null;
    showScreen("home");
  }
  await refreshSessions();
}

async function boot() {
  const data = await api("/api/models");
  const box = $("models");
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = "Choose a model";
  box.appendChild(blank);
  data.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = model.title;
    option.title = model.description;
    box.appendChild(option);
  });
  box.onchange = async () => {
    active = box.value;
    if (!active) {
      $("tabs").innerHTML = "";
      $("fields").innerHTML = "";
      return;
    }
    const schema = await api("/api/select", {model: active});
    if (session) session.model = schema.model;
    renderFields(schema.fields);
  };
  document.querySelectorAll("#steps button").forEach((button) => {
    button.onclick = () => showScreen(button.dataset.screen);
  });
  await refreshSessions();
  showScreen("home");
}

$("create").onclick = async () => {
  const picked = document.querySelector("input[name=purpose]:checked");
  session = await api("/api/sessions", {
    name: $("session-name").value,
    purpose: picked ? picked.value : "train"
  });
  applySession();
  showScreen("data");
};

$("back-sessions").onclick = async () => {
  showScreen("home");
  await refreshSessions();
};

document.querySelectorAll("#source-tabs button").forEach((button) => {
  button.onclick = () => {
    sourceKind = button.dataset.source;
    document.querySelectorAll("#source-tabs button").forEach((tab) => {
      tab.classList.toggle("active", tab === button);
    });
    document.querySelectorAll("[data-source-pane]").forEach((pane) => {
      pane.hidden = pane.dataset.sourcePane !== sourceKind;
    });
  };
});

$("file").onchange = () => {
  const chosen = $("file").files && $("file").files[0];
  const name = chosen ? chosen.name.toLowerCase() : "";
  $("file-target-box").hidden = !name.endsWith(".csv");
};

$("load").onclick = async () => {
  pageOffset = 0;
  let data;
  if (sourceKind === "file") {
    const file = $("file").files && $("file").files[0];
    if (!file) {
      $("notice").textContent = "Choose a file.";
      return;
    }
    const response = await fetch("/api/upload", {
      method: "POST",
      headers: {
        "X-Filename": file.name,
        "X-Target": $("file-target").value
      },
      body: file
    });
    data = await response.json();
    if (!response.ok) {
      $("notice").textContent = data.error || "upload failed";
      return;
    }
    $("notice").textContent = "";
  } else if (sourceKind === "url") {
    data = await api("/api/url", {
      url: $("url").value,
      target: $("url-target").value
    });
  } else {
    data = await api("/api/data", {
      csv: $("csv").value,
      target: $("target").value
    });
  }
  if (session) session.has_data = true;
  note(data);
  await showPage();
};

$("apply").onclick = async () => {
  const data = await api("/api/prepare", {
    scale_divisor: $("divisor").value,
    standardize: $("center").checked,
    val_fraction: $("holdout").value
  });
  if (session) {
    session.scale_divisor = data.scale_divisor;
    session.standardize = data.standardize;
    session.val_fraction = data.val_fraction;
  }
  pageOffset = 0;
  note(data);
  await showPage();
};

$("prev").onclick = async () => {
  pageOffset = Math.max(0, pageOffset - pageLimit());
  await showPage();
};

$("next").onclick = async () => {
  pageOffset += pageLimit();
  await showPage();
};

$("page-size").onchange = async () => {
  pageOffset = 0;
  if ($("data-note").textContent) await showPage();
};

$("to-model").onclick = () => showScreen("model");

$("to-run").onclick = async () => {
  const ready = $("fields").querySelector("input, select");
  if (session && session.purpose === "train" && ready) {
    await api("/api/params", {fields: collectFields()});
  }
  showScreen("run");
};

$("load-weights").onclick = async () => {
  const file = $("weights").files && $("weights").files[0];
  if (!file) {
    $("notice").textContent = "Choose a .ttw file.";
    return;
  }
  const response = await fetch("/api/weights", {
    method: "POST",
    headers: {"X-Filename": file.name},
    body: file
  });
  const data = await response.json();
  if (!response.ok) {
    $("notice").textContent = data.error || "upload failed";
    return;
  }
  session = data;
  applySession();
};

async function showCheck() {
  const page = await api("/api/check?offset=" + checkOffset + "&limit=8");
  const note = $("check-score");
  if (!page.scored) {
    note.textContent = page.note || "";
    $("check-table").innerHTML = "";
    $("check-page").textContent = "";
    return page;
  }
  const kind = page.split === "test"
    ? "test rows"
    : "rows held out of teaching";
  note.textContent = page.correct + " / " + page.scored
    + " correct on the " + kind + ".";
  paintTable($("check-table"), page, true);
  const start = page.records.length ? page.offset + 1 : 0;
  const end = page.offset + page.records.length;
  $("check-page").textContent = start + "-" + end + " of " + page.scored;
  return page;
}

$("check").onclick = async () => {
  checkOffset = 0;
  const page = await showCheck();
  const first = page.records && page.records[0];
  if (first) await playForward(first.index);
};

$("check-prev").onclick = () => {
  checkOffset = Math.max(0, checkOffset - 8);
  return showCheck();
};

$("check-next").onclick = () => {
  checkOffset += 8;
  return showCheck();
};

$("pause").onclick = () => api("/api/control", {
  paused: $("pause").textContent === "Pause"
});

$("fit").onclick = () => teach("full");

$("step").onclick = () => teach("step");

$("score").onclick = async () => {
  setBusy(true);
  try {
    const page = await api("/api/predict?offset=0&limit=" + pageLimit());
    paintTable($("pred"), page, true);
    try {
      await loadFilm(false);
      startPlay();
    } catch (err) {
      playback = null;
    }
  } finally {
    setBusy(false);
  }
};

$("back").onclick = async () => {
  if (timer) { clearInterval(timer); timer = null; }
  await loadFilm(true);
};

document.querySelectorAll("input[name=scale]").forEach((input) => {
  input.addEventListener("change", () => {
    if (!input.checked) return;
    scaleMode = input.value;
    if (playback && playback.frames && playback.frames.length) draw();
    else if (layoutSizes && layoutSizes.length) {
      drawStructure(layoutSizes);
      const scaleNote = scaleMode === "real"
        ? "Every neuron is drawn at this scale."
        : "The first 16 of a wide layer are drawn.";
      $("caption").textContent = "Configured network, "
        + layoutSizes.join(" → ") + " neurons. " + scaleNote;
    }
  });
});

$("net-zoom").oninput = () => {
  netZoom = Number($("net-zoom").value) || 1;
  applyNetView();
};

const netStage = $("stage-net");
netStage.addEventListener("pointerdown", (event) => {
  if (event.target.id === "net-zoom") return;
  netDrag = {
    x: event.clientX,
    y: event.clientY,
    px: netPanX,
    py: netPanY
  };
  netStage.classList.add("dragging");
  netStage.setPointerCapture(event.pointerId);
});
netStage.addEventListener("pointermove", (event) => {
  if (!netDrag) return;
  netPanX = netDrag.px + event.clientX - netDrag.x;
  netPanY = netDrag.py + event.clientY - netDrag.y;
  applyNetView();
});
function endNetDrag() {
  netDrag = null;
  $("stage-net").classList.remove("dragging");
}
netStage.addEventListener("pointerup", endNetDrag);
netStage.addEventListener("pointercancel", endNetDrag);

$("play").onclick = () => {
  if (!playback) return;
  if (timer) { clearInterval(timer); timer = null; return; }
  startPlay();
};

function closePicks(except) {
  document.querySelectorAll(".pick.open").forEach((pick) => {
    if (pick === except) return;
    pick.classList.remove("open");
    const menu = pick.querySelector(".pick-menu");
    if (menu) menu.hidden = true;
  });
}

function dress(select) {
  if (!select || select.dataset.dressed) return;
  select.dataset.dressed = "1";
  select.tabIndex = -1;
  select.setAttribute("aria-hidden", "true");
  select.classList.add("native-select");
  const pick = document.createElement("div");
  pick.className = "pick";
  const face = document.createElement("button");
  face.type = "button";
  face.className = "pick-face";
  const menu = document.createElement("div");
  menu.className = "pick-menu";
  menu.hidden = true;
  select.parentNode.insertBefore(pick, select);
  pick.append(face, menu, select);
  const paint = () => {
    const chosen = select.selectedOptions[0];
    face.textContent = chosen ? chosen.textContent : "";
  };
  const close = () => {
    menu.hidden = true;
    pick.classList.remove("open");
  };
  const open = () => {
    closePicks(pick);
    menu.innerHTML = "";
    [...select.options].forEach((option) => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "pick-item";
      if (option.value === select.value) item.classList.add("current");
      item.textContent = option.textContent;
      if (option.title) item.title = option.title;
      item.onclick = (event) => {
        event.stopPropagation();
        select.value = option.value;
        paint();
        close();
        select.dispatchEvent(new Event("change"));
      };
      menu.appendChild(item);
    });
    menu.hidden = false;
    pick.classList.add("open");
    const box = face.getBoundingClientRect();
    const above = Math.max(0, box.top - 8);
    const below = Math.max(0, window.innerHeight - box.bottom - 8);
    const up = below < above;
    pick.classList.toggle("up", up);
    menu.style.maxHeight = Math.max(1, up ? above : below) + "px";
  };
  face.onclick = (event) => {
    event.stopPropagation();
    if (menu.hidden) open();
    else close();
  };
  select.addEventListener("change", paint);
  new MutationObserver(paint).observe(select, {childList: true});
  paint();
}

document.addEventListener("click", () => closePicks(null));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closePicks(null);
});

$("report-frame").addEventListener("load", () => {
  fitReportFrame();
  const doc = $("report-frame").contentDocument;
  if (doc && doc.fonts) doc.fonts.ready.then(fitReportFrame);
});

dress($("page-size"));
dress($("models"));
boot();
</script>
</body>
</html>
"""
