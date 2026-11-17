"""Self-contained, offline, keyboard-accessible evidence report."""
from __future__ import annotations
import html
import json
from . import study_view


def _number(value):
    return "undefined" if value is None else f"{value:.6g}"


def render_report(report: dict) -> str:
    """Render a schema-1 analysis or schema-2 study; never embed raw matrices."""
    bundle = report
    is_study = bundle.get('report_type') == 'validation_study'
    study_summary = study_view.summary(bundle) if is_study else ''
    study_detail = study_view.DETAIL if is_study else ''
    if is_study:
        report = bundle['cohorts']['discovery']
    payload = json.dumps(bundle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for char, escape in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                         ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(char, escape)
    title = html.escape(report["title"])
    sample = html.escape(report["sample_label"])
    rows = []
    for cfg in report["configurations"]:
        m = cfg["metrics"]
        rows.append("<tr><th scope=\"row\">" + html.escape(cfg["id"]) + "</th>" +
                    "".join(f"<td>{_number(m[k])}</td>" for k in
                            ("weight_mse", "output_mse", "relative_output_l2", "scale_count", "clipped_count")) + "</tr>")
    warnings = "".join("<li>" + html.escape(w) + "</li>" for w in report["warnings"])
    shape = report["shapes"]
    dimensions = f"W {shape['weights'][0]} × {shape['weights'][1]} · X {shape['inputs'][0]} × {shape['inputs'][1]}"
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><title>{title} · RoundoffLab</title>
<style>{STYLE}{study_view.STYLE if is_study else ''}</style></head><body>
<a class="skip" href="#comparison">Skip to comparison</a>
<header><a class="brand" href="#"><span class="mark" aria-hidden="true">R<span>·</span></span>Roundoff<span>Lab</span></a>
<span class="header-note">WEIGHT-ONLY QUANTIZATION / LOCAL EVIDENCE</span><button id="download-json" type="button">Download JSON <span aria-hidden="true">↓</span></button></header>
<main>{study_summary}<section class="intro"><div><p class="eyebrow">A matrix is not its output.</p><h1>{title}</h1>
<p class="lede">Smaller weight error does not always mean smaller output error. Compare both before choosing a quantization policy.</p>
<div class="tags"><span id="shape-label">{html.escape(dimensions)}</span><span>float64 reference</span><span id="sample-label">{sample}</span></div>
<p class="boundary">Single linear layer · {'external candidates and/or simulations' if is_study else 'symmetric fake quantization'} · no task-accuracy or performance claims</p></div>
<div class="rank-card"><div class="panel-label">Two rankings. Same weights.</div><p class="muted">Each line connects one policy's rank. Lower error ranks first.</p>
<div id="rank-chart" aria-label="Weight and output error ranking comparison"></div><p id="reversal-note" class="reversal-note" aria-live="polite"></p></div></section>
<section id="comparison" class="panel"><div class="section-head"><div><p class="eyebrow">Policy comparison</p><h2>The numbers behind the rankings</h2></div>
<label class="sort-label">Sort by <select id="sort-select"><option value="weight_mse">Weight MSE</option><option value="output_mse">Output MSE</option><option value="id">Policy name</option></select></label></div>
<div class="table-wrap" role="region" aria-label="Quantization policy results" tabindex="0"><table><thead><tr><th scope="col">Policy</th><th scope="col">Weight MSE</th><th scope="col">Output MSE</th><th scope="col">Relative output L2</th><th scope="col">Scales</th><th scope="col">Clipped weights</th></tr></thead><tbody id="comparison-body">{''.join(rows)}</tbody></table></div>
<p class="footnote">MSE values have different units and are not directly comparable across the two columns. Scales are counted, not a memory estimate. A zero reference norm makes relative L2 undefined.</p></section>
<section class="inspection"><div class="section-head"><div><p class="eyebrow">Inspect a policy</p><h2>Where does the error land?</h2></div><label>Selected policy <select id="config-select"></select></label></div>
<div class="metrics" aria-live="polite"><div><span>Weight MSE</span><strong id="weight-mse">—</strong></div><div><span>Output MSE</span><strong id="output-mse">—</strong></div><div><span>Relative output L2</span><strong id="relative-l2">—</strong></div><div><span>Clipped weights</span><strong id="clipped">—</strong></div></div>
<div class="detail-grid"><article class="panel map-panel"><div class="section-head"><div><h3>Weight error map</h3><p class="muted">Mean absolute error per tile. Select a tile to inspect its bounds.</p></div></div>
<div class="tile-controls"><label>Output tile <select id="row-select"></select></label><label>Input tile <select id="col-select"></select></label></div>
<div id="heatmap" class="heatmap" role="group" aria-label="Weight error tiles; use arrow keys to move between tiles"></div><div class="legend"><span>Zero</span><span class="ramp"></span><span id="heat-max">—</span></div><p id="tile-detail" class="footnote" aria-live="polite">Rows = output channels. Columns = input channels.</p></article>
<article class="panel"><h3>Output error by channel</h3><p class="muted">MSE on the supplied inputs. Bar scales update with the selected policy.</p><div id="channels" class="channels" aria-label="Per-channel output errors"></div></article></div>{study_detail}</section>
<section class="method panel"><div><p class="eyebrow">What this report establishes</p><h2>Evidence, with boundaries.</h2><ul>{warnings}</ul></div><div class="formula"><span>Reference</span><code>Y = XWᵀ</code><span>Quantized comparison</span><code>{'Ŷ = XWqᵀ' if is_study else 'Ŷ = XQ(W)ᵀ'}</code><p>No activations are quantized. {'External Wq is used as supplied, without re-quantization.' if is_study else 'Percentile clipping is computed from weights, not from calibration inputs.'}</p></div></section>
<details class="provenance"><summary>Numerical policy and input fingerprints</summary><pre id="provenance"></pre><p>Fingerprints include dtype, shape and C-order values. They do not certify provenance or prevent sensitive information in summary statistics.</p></details>
<noscript><p class="notice">JavaScript is disabled. The complete comparison table and scope notes remain readable; interactive inspection requires JavaScript.</p></noscript>
</main><footer><span>RoundoffLab <span class="muted">/ offline by design</span></span><a href="https://github.com/gedouluhui/roundoff-lab">Source, method & reproducible examples ↗</a></footer>
<script id="report-data" type="application/json">{payload}</script><script>{SCRIPT}</script></body></html>'''


STYLE = r'''
:root{--ink:#182e49;--muted:#52657a;--blue:#246889;--orange:#b74b22;--line:#d6e0e8;--paper:#f0f5f9;--white:#fff;--mono:Consolas,"Liberation Mono",monospace}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 "Segoe UI",system-ui,sans-serif}button,select{font:inherit}button,select,a{touch-action:manipulation}button,select{color:var(--ink);border:1px solid #aabccc;background:white;border-radius:7px;padding:9px 13px;min-height:42px}button{cursor:pointer}button:hover{border-color:var(--blue);background:#f4f8fb}a{color:var(--blue)}:focus-visible{outline:3px solid #087bac;outline-offset:3px}.skip{position:absolute;left:16px;top:-80px;background:white;padding:12px;z-index:10}.skip:focus{top:10px}header,footer{max-width:1400px;margin:auto;padding:24px 44px;display:flex;align-items:center;justify-content:space-between;gap:20px}header{border-bottom:1px solid var(--line)}.brand{display:flex;align-items:center;color:var(--ink);text-decoration:none;font-weight:750;font-size:21px;letter-spacing:-.7px;gap:0}.brand>span:last-child{font-weight:350}.mark{background:var(--ink);color:white;padding:0 8px;margin-right:10px;border-radius:5px;line-height:32px;font:700 25px/32px var(--mono)}.mark span{color:#f4af7f}.header-note{font:10px var(--mono);letter-spacing:1.2px;color:var(--muted)}main{max-width:1312px;margin:auto;padding:42px 0}.intro{display:grid;grid-template-columns:1fr 1fr;gap:48px;align-items:center;margin:0 0 36px}.eyebrow{margin:0 0 10px;color:var(--blue);font-size:12px;font-weight:700;letter-spacing:.1em;text-transform:uppercase}h1{font-size:clamp(30px,3.4vw,48px);line-height:1.12;letter-spacing:-1.6px;margin:0 0 20px;overflow-wrap:anywhere}h2{font-size:23px;line-height:1.25;letter-spacing:-.6px;margin:0}h3{font-size:17px;margin:0 0 6px}.lede{font-size:18px;line-height:1.6;color:var(--muted);max-width:540px}.tags{display:flex;gap:7px;flex-wrap:wrap;margin-top:24px}.tags span{border:1px solid #cedbe5;background:#e7eff5;border-radius:4px;padding:5px 8px;font:11px/1.5 var(--mono);overflow-wrap:anywhere;max-width:100%}.boundary{font-size:12px;color:var(--muted);margin-top:17px}.rank-card{background:#e4edf4;border:1px solid #cedce7;border-radius:12px;padding:24px}.panel-label{font-size:16px;font-weight:700}.muted{color:var(--muted);font-size:13px;margin:5px 0 14px}.rank-card svg{display:block;width:100%;height:auto}.reversal-note{font-size:12px;margin:12px 0 0;padding-top:12px;border-top:1px solid #c9d8e4;overflow-wrap:anywhere}.panel{border:1px solid var(--line);border-radius:12px;background:var(--white);padding:26px}.section-head{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:20px}.section-head label{font-size:12px;color:var(--muted);display:flex;align-items:center;gap:10px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;white-space:nowrap;text-align:right;font:12px/1.5 var(--mono)}th,td{padding:13px 12px;border-bottom:1px solid #e6edf2}thead th{color:var(--muted);font-size:11px;font-weight:500}th:first-child{text-align:left}tbody th{font-weight:600}tbody tr.selected{background:#eef5fa}tbody tr:hover{background:#f3f7fa}.footnote{font-size:12px;color:var(--muted);margin:14px 0 0}.inspection{margin-top:40px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-bottom:20px}.metrics>div{padding:20px 24px;background:#fff}.metrics span{display:block;color:var(--muted);font-size:12px}.metrics strong{display:block;margin-top:7px;font:24px/1.2 var(--mono);overflow-wrap:anywhere}.detail-grid{display:grid;grid-template-columns:1.35fr 1fr;gap:20px}.detail-grid>article{min-width:0}.heatmap{display:grid;gap:1px;background:#dce5ed;padding:3px;border-radius:6px;overflow:auto;max-height:310px}.tile{display:block;border:0;border-radius:0;min-height:10px;min-width:0;padding:0;aspect-ratio:1.5;position:relative}.tile:focus-visible{z-index:1}.tile.active{outline:2px solid var(--ink);outline-offset:-2px}.tile-controls{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}.tile-controls label{font-size:12px;color:var(--muted);display:flex;align-items:center;gap:8px}.tile-controls select{max-width:135px;min-height:44px}.legend{display:flex;align-items:center;gap:10px;font:10px var(--mono);color:var(--muted);margin-top:10px}.ramp{display:inline-block;width:100px;height:7px;background:linear-gradient(90deg,#edf3f8,#b74b22)}.channels{max-height:280px;overflow:auto;padding-right:8px}.channel{display:grid;grid-template-columns:35px 1fr 86px;gap:10px;align-items:center;margin:12px 0;font:11px var(--mono)}.track{height:9px;background:#e6eef4;border-radius:2px;overflow:hidden}.bar{height:100%;background:var(--blue);border-radius:2px}.channel-value{text-align:right}.method{display:grid;grid-template-columns:1.5fr 1fr;gap:55px;margin-top:36px}.method ul{padding-left:18px;color:var(--muted);font-size:13px}.method li{margin-bottom:9px}.formula{border-left:2px solid #d3e1ec;padding-left:26px}.formula>span{display:block;font-size:11px;color:var(--muted)}.formula code{display:block;font:26px var(--mono);margin:5px 0 16px}.formula p{font-size:12px;color:var(--muted);margin-bottom:0}.provenance{font-size:12px;margin:22px 2px}.provenance summary{cursor:pointer;color:var(--blue)}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.65 var(--mono);padding:18px;background:#e5edf4;border-radius:6px}footer{border-top:1px solid var(--line);font-size:12px;padding-top:20px;padding-bottom:28px}.notice{padding:15px;background:#fff4e9;border:1px solid #ddc1a3}
@media(min-width:1500px){main{padding-top:52px}}@media(max-width:1400px){main{margin:0 36px}header,footer{padding-left:36px;padding-right:36px}.intro{gap:30px}}
@media(max-width:950px){.header-note{display:none}.intro{grid-template-columns:1fr}.intro>.rank-card{max-width:700px}.detail-grid{grid-template-columns:1fr}.method{gap:25px}.metrics strong{font-size:19px}}
@media(max-width:600px){header,footer{padding:18px}header{gap:8px}.brand{font-size:18px}#download-json{font-size:11px;padding:7px}main{margin:0 16px;padding-top:30px}.panel,.rank-card{padding:18px}.metrics{grid-template-columns:repeat(2,1fr)}.metrics>div{padding:16px}.section-head{align-items:flex-start;flex-direction:column;gap:14px}.method{grid-template-columns:1fr}.formula{padding-left:18px}footer{flex-direction:column;align-items:flex-start}.lede{font-size:16px}h1{letter-spacing:-1px}.heatmap{gap:1px}.tile{min-height:9px;aspect-ratio:auto}.tags span{font-size:10px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*,*::before,*::after{animation:none!important;transition:none!important}}
'''

SCRIPT = r'''
'use strict';
const bundle = JSON.parse(document.getElementById('report-data').textContent);
const isStudy = bundle.report_type === 'validation_study';
let report = isStudy ? bundle.cohorts.discovery : bundle;
let configs = report.configurations;
const $ = id => document.getElementById(id);
const num = v => v === null ? 'undefined' : v === 0 ? '0' : Math.abs(v) < .001 || Math.abs(v) >= 1e5 ? v.toExponential(3) : Number(v.toPrecision(5)).toString();
const el = (tag, text, cls) => { const n=document.createElement(tag); if(text!==undefined)n.textContent=text; if(cls)n.className=cls; return n; };
const colors=['#246889','#b74b22','#476b45','#775b9a','#996021','#426e79'];
let selected=isStudy?bundle.selection.candidate_id:configs[0].id;
configs.forEach(c=>{const option=el('option',c.id);option.value=c.id;$('config-select').append(option);});
$('config-select').value=selected;
function renderTable(){
 const key=$('sort-select').value;
 const ordered=[...configs].sort((a,b)=> key==='id' ? a.id.localeCompare(b.id) : a.metrics[key]-b.metrics[key] || a.id.localeCompare(b.id));
 $('comparison-body').replaceChildren();
 ordered.forEach(c=>{const row=el('tr');if(c.id===selected)row.className='selected';const th=el('th',c.id);th.scope='row';row.append(th);
 ['weight_mse','output_mse','relative_output_l2','scale_count','clipped_count'].forEach(k=>row.append(el('td',c.metrics[k]===null&&(k==='scale_count'||k==='clipped_count')?'unknown':num(c.metrics[k]))));$('comparison-body').append(row);});
}
function drawRanks(){
 const ns='http://www.w3.org/2000/svg', svg=document.createElementNS(ns,'svg');
 const add=(tag,attrs,text)=>{const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;svg.append(n);return n;};
 const height=50+configs.length*30;svg.setAttribute('viewBox',`0 0 500 ${height}`);svg.setAttribute('role','img');svg.setAttribute('aria-label','Policy ranks for weight MSE and output MSE; exact values are in the comparison table');
 const sorted=key=>[...configs].sort((a,b)=>a.metrics[key]-b.metrics[key]||a.id.localeCompare(b.id));
 const left=sorted('weight_mse'),right=sorted('output_mse');
 add('text',{x:0,y:14,fill:'#52657a','font-size':11},'Weight MSE ↓');add('text',{x:500,y:14,fill:'#52657a','font-size':11,'text-anchor':'end'},'Output MSE ↓');
 configs.forEach((c,i)=>{const l=left.indexOf(c),r=right.indexOf(c),y1=42+l*30,y2=42+r*30,col=colors[i%colors.length];
 add('path',{d:`M 156 ${y1} C 230 ${y1},270 ${y2},344 ${y2}`,fill:'none',stroke:col,'stroke-width':c.id===selected?3:1.5,opacity:c.id===selected?1:.5});
 add('circle',{cx:156,cy:y1,r:3,fill:col});add('circle',{cx:344,cy:y2,r:3,fill:col});
 add('text',{x:0,y:y1+4,fill:col,'font-size':11,'font-family':'monospace'},c.id);add('text',{x:500,y:y2+4,fill:col,'font-size':11,'font-family':'monospace','text-anchor':'end'},c.id);});
 $('rank-chart').replaceChildren(svg);
 const r=report.ranking_reversals;
 $('reversal-note').textContent=r.length ? `${r.length} strict ranking reversal${r.length===1?'':'s'} observed. For example, ${r[0].lower_weight_error} has lower weight MSE, but ${r[0].lower_output_error} has lower output MSE on these inputs.` : 'No strict ranking reversal observed on these inputs. Equal errors are ordered by policy name, not treated as a difference.';
}
function inspect(){
 selected=$('config-select').value;const c=configs.find(v=>v.id===selected),m=c.metrics;
 $('weight-mse').textContent=num(m.weight_mse);$('output-mse').textContent=num(m.output_mse);$('relative-l2').textContent=num(m.relative_output_l2);$('clipped').textContent=m.clipped_count===null?'unknown':m.clipped_count.toLocaleString('en-US');
 const h=c.heatmap,max=Math.max(0,...h.values.flat());$('heatmap').replaceChildren();$('heatmap').style.gridTemplateColumns=`repeat(${h.col_bounds.length}, minmax(0, 1fr))`;
 h.values.forEach((row,r)=>row.forEach((v,k)=>{const b=el('button',undefined,'tile');b.type='button';b.tabIndex=r===0&&k===0?0:-1;const a=h.row_bounds[r],d=h.col_bounds[k];const text=`Output channels [${a[0]}, ${a[1]}), input channels [${d[0]}, ${d[1]}): mean absolute error ${num(v)}`;
 const t=max?v/max:0;const start=[237,243,248],end=[183,75,34];b.style.backgroundColor=`rgb(${start.map((n,i)=>Math.round(n+(end[i]-n)*t)).join(',')})`;b.title=text;b.setAttribute('aria-label',text);b.addEventListener('click',()=>chooseTile(r,k,false));b.addEventListener('keydown',event=>{const steps={ArrowRight:[0,1],ArrowLeft:[0,-1],ArrowDown:[1,0],ArrowUp:[-1,0]};if(steps[event.key]){event.preventDefault();const [dr,dc]=steps[event.key];chooseTile(Math.max(0,Math.min(h.row_bounds.length-1,r+dr)),Math.max(0,Math.min(h.col_bounds.length-1,k+dc)),true);}});$('heatmap').append(b);}));
 for(const [id,bounds] of [['row-select',h.row_bounds],['col-select',h.col_bounds]]){const select=$(id);select.replaceChildren();bounds.forEach((b,i)=>{const option=el('option',`[${b[0]}, ${b[1]})`);option.value=i;select.append(option);});select.onchange=()=>chooseTile(Number($('row-select').value),Number($('col-select').value),false);}
 function chooseTile(r,k,focus){const tiles=[...$('heatmap').children],chosen=tiles[r*h.col_bounds.length+k];tiles.forEach(t=>{t.tabIndex=t===chosen?0:-1;t.classList.toggle('active',t===chosen);});$('row-select').value=r;$('col-select').value=k;$('tile-detail').textContent=chosen.getAttribute('aria-label');if(focus)chosen.focus();}
 $('heat-max').textContent=num(max);chooseTile(0,0,false);
 const largest=Math.max(0,...c.channels.map(v=>v.output_mse));$('channels').replaceChildren();
 c.channels.forEach(v=>{const row=el('div',undefined,'channel');row.append(el('span',`O${String(v.index).padStart(2,'0')}`));const track=el('div',undefined,'track'),bar=el('div',undefined,'bar');bar.style.width=`${largest?(v.output_mse/largest)*100:0}%`;track.append(bar);row.append(track,el('span',num(v.output_mse),'channel-value'));row.title=`Channel ${v.index}; output MSE ${num(v.output_mse)}; reference RMS ${num(v.reference_rms)}`;$('channels').append(row);});
 $('provenance').textContent=JSON.stringify(report.provenance,null,2);
 $('sample-label').textContent=report.sample_label;
 const shapes=report.shapes;$('shape-label').textContent=`W ${shapes.weights[0]} × ${shapes.weights[1]} · X ${shapes.inputs[0]} × ${shapes.inputs[1]}`;
 if(isStudy){
  $('propagated-mse').textContent=num(m.propagated_mse);
  $('rounding-mask').textContent=c.rounding_mask_detected?'Forward rounding masks a nonzero propagated error. Observed MSE is zero; do not read it as zero weight-error effect.':'Both measurements are reported separately. No zero-observed/nonzero-propagated masking detected for this candidate on this cohort.';
  $('candidate-origin').textContent=`Origin: ${c.origin==='external_dequantized'?'External, user-declared; scale and clipping counts unknown':'RoundoffLab simulation'}. Candidate SHA256: ${c.candidate_sha256}`;
  $('worst-samples').replaceChildren();
  if(!c.worst_samples.length)$('worst-samples').append(el('p','No nonzero propagated sample errors.','footnote'));
  c.worst_samples.forEach(v=>$('worst-samples').append(el('div',`Sample ${v.sample_index} · output channel ${v.output_channel} · max propagated absolute error ${num(v.max_abs_error)}`,'witness')));
 }
 renderTable();drawRanks();
}
$('config-select').addEventListener('change',inspect);$('sort-select').addEventListener('change',renderTable);
$('download-json').addEventListener('click',()=>{const blob=new Blob([JSON.stringify(bundle,null,2)+'\n'],{type:'application/json'});const url=URL.createObjectURL(blob),a=el('a');a.href=url;a.download='roundoff-report.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
if(isStudy)$('cohort-select').addEventListener('change',()=>{report=bundle.cohorts[$('cohort-select').value];configs=report.configurations;inspect();});
inspect();
'''
