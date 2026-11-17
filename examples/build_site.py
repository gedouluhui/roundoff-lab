"""Generate the public, synthetic-only demo and an evidence-based SVG preview."""
from __future__ import annotations
import argparse
from html import escape
import json
from pathlib import Path

from roundofflab import analyze, render_report
from roundofflab.demo import make_demo
from roundofflab.io import write_artifacts


def preview(quiet: dict, active: dict) -> str:
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="610" viewBox="0 0 1200 610" role="img" aria-labelledby="title desc">',
             '<title id="title">RoundoffLab: the same weights, two input conditions</title>',
             '<desc id="desc">Measured output errors for two quantization policies on synthetic inputs. See the adjacent report for full data.</desc>',
             '<rect width="1200" height="610" rx="16" fill="#edf3f8"/>',
             '<style>text{font-family:Segoe UI,Arial,sans-serif}.mono{font-family:Consolas,monospace}</style>']
    def text(x, y, value, size=16, color='#182e49', cls=''):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" class="{cls}">{escape(str(value))}</text>')
    text(40, 52, 'RoundoffLab', 26)
    text(40, 105, 'The same weights. A different answer.', 35)
    text(40, 142, 'Weight-only fake quantization • synthetic example • float64 linear-layer reference', 16, '#52657a')
    for left, report, heading in ((40, quiet, 'Outlier direction quiet'), (625, active, 'Outlier direction active')):
        parts.append(f'<rect x="{left}" y="180" width="535" height="320" rx="10" fill="white" stroke="#d6e0e8"/>')
        text(left + 24, 218, heading, 23)
        text(left + 24, 250, 'Policy                          Weight MSE     Output MSE', 13, '#52657a', 'mono')
        for y, name, label, color in ((304, 'int4-channel', 'Absmax', '#246889'),
                                     (397, 'int4-channel-p95', 'Clip p95', '#b74b22')):
            config = next(c for c in report['configurations'] if c['id'] == name)
            text(left + 24, y, label + ' / 4-bit', 19, color)
            text(left + 280, y, format(config['metrics']['weight_mse'], '.4g'), 19, '#182e49', 'mono')
            text(left + 416, y, format(config['metrics']['output_mse'], '.4g'), 19, '#182e49', 'mono')
            text(left + 24, y + 27, 'Per-output-channel symmetric quantization', 13, '#52657a')
        text(left + 24, 470, 'Both columns are errors, not task accuracy.', 13, '#52657a')
    text(40, 550, 'Matrices in. Inspectable offline evidence out.', 23)
    text(40, 583, 'No GPU · No model conversion · No uploads · No speed or accuracy claims', 14, '#52657a')
    parts.append('</svg>')
    return ''.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'docs')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    reports = {}
    for scenario in ('quiet-outlier', 'active-outlier'):
        reports[scenario] = analyze(*make_demo(scenario=scenario),
            title='When the outlier goes quiet' if scenario == 'quiet-outlier' else 'When the outlier wakes up',
            sample_label=f'synthetic / {scenario} (seed 7)')
    navigation = '<nav aria-label="Synthetic scenarios" style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:28px;font-size:14px"><a href="index.html">Outlier quiet</a><a href="active.html">Outlier active</a><span>Same weights, different inputs. Synthetic examples.</span></nav>'
    payloads = {}
    for name, stem in (('quiet-outlier', 'quiet'), ('active-outlier', 'active')):
        report = reports[name]
        page = 'index.html' if stem == 'quiet' else 'active.html'
        payloads[page] = render_report(report).replace('<main>', '<main>' + navigation, 1).encode('utf-8')
        payloads[stem + '.json'] = (json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8')
    payloads['preview.svg'] = preview(reports['quiet-outlier'], reports['active-outlier']).encode('utf-8')
    payloads['.nojekyll'] = b''
    write_artifacts(args.out, payloads, force=args.force)
    print(f'Generated synthetic demo in {args.out}')


if __name__ == '__main__':
    main()
