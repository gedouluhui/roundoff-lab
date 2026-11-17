"""Small static fragments for a schema-2 study. No executable user markup."""
from html import escape


def summary(bundle: dict) -> str:
    choice = bundle['selection']
    metric = choice['metric']
    validation = {c['id']: c for c in bundle['cohorts']['validation']['configurations']}
    rows = []
    for candidate in bundle['cohorts']['discovery']['configurations']:
        other = validation[candidate['id']]
        source = 'External (user-declared)' if candidate['origin'] == 'external_dequantized' else 'RoundoffLab simulation'
        rows.append('<tr><th scope="row">' + escape(candidate['id']) + '</th><td>' + source +
                    f'</td><td>{candidate["metrics"][metric]:.6g}</td><td>{other["metrics"][metric]:.6g}</td></tr>')
    warnings = ''.join('<li>' + escape(w) + '</li>' for w in bundle['warnings'])
    return f'''<section class="panel study-summary" aria-labelledby="study-title">
<div class="section-head"><div><p class="eyebrow">Candidate validation study</p><h2 id="study-title">Choose on discovery. Inspect on validation.</h2></div>
<label>Inspect cohort <select id="cohort-select"><option value="discovery">Discovery</option><option value="validation">Validation</option></select></label></div>
<div class="selection-banner"><div><span>Fixed candidate · {escape(choice['method'])}</span><strong id="fixed-selection">{escape(choice['candidate_id'])}</strong></div>
<div><span>Selection metric</span><strong>{escape(metric)}</strong></div><div><span>Excess validation metric</span><strong>{choice['excess_validation_metric']:.6g}</strong></div></div>
<p class="footnote">The fixed candidate never changes when you switch the view. Excess compares with the best supplied candidate on validation; it is not statistical significance.</p>
<div class="table-wrap" tabindex="0" role="region" aria-label="Both cohorts and candidate origins"><table><thead><tr><th scope="col">Candidate</th><th scope="col">Origin</th><th scope="col">Discovery · {escape(metric)}</th><th scope="col">Validation · {escape(metric)}</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<details><summary>Study boundaries and input independence</summary><ul>{warnings}</ul></details></section>'''


DETAIL = '''<section class="panel study-detail" aria-label="Propagation and worst-sample evidence">
<div class="section-head"><div><p class="eyebrow">Two numerical measurements</p><h3>Small error can disappear in a rounded forward pass</h3></div><div class="propagation-value"><span>Propagated MSE</span><strong id="propagated-mse">—</strong></div></div>
<p class="footnote">Observed: XWqᵀ − XWᵀ, from two float64 forwards. Propagated: X(Wq − W)ᵀ, directly from the weight difference. Neither is exact arithmetic or a deployed-kernel measurement.</p>
<p id="rounding-mask" class="notice" aria-live="polite"></p><p id="candidate-origin" class="footnote">External counts and bit widths are unknown unless a simulation actually measures them.</p>
<h3>Worst samples by propagated absolute error</h3><p class="footnote">Zero-based row and output-channel coordinates; no raw activations are embedded.</p>
<div id="worst-samples" aria-live="polite"></div></section>'''

STYLE = '''
#candidate-origin{overflow-wrap:anywhere}
.study-summary{margin-bottom:32px;border-top:4px solid var(--blue)}.study-summary details{font-size:12px;color:var(--muted);margin-top:18px}.study-summary summary{cursor:pointer}.selection-banner{display:grid;grid-template-columns:1.2fr 1fr .8fr;gap:20px;background:#eaf2f8;padding:18px;border-radius:8px}.selection-banner span,.propagation-value span{display:block;font-size:11px;color:var(--muted)}.selection-banner strong,.propagation-value strong{display:block;font:17px/1.6 var(--mono);overflow-wrap:anywhere}.study-detail{margin-top:20px}.study-detail h3{margin-top:12px}.study-detail .notice{font-size:13px}.witness{font:12px/1.6 var(--mono);border-bottom:1px solid var(--line);padding:8px 0;overflow-wrap:anywhere}.study-summary select,#config-select{max-width:min(390px,75vw)}
@media(max-width:600px){.selection-banner{grid-template-columns:1fr;gap:12px}.study-detail .section-head{gap:8px}.study-summary .section-head{gap:16px}}
'''
