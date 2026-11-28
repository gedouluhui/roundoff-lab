# RoundoffLab: numerical and product contract

Status: v0.1 implementation contract, 2026-11-18. This is an offline diagnostic, not a new quantization algorithm.

## Job and non-goals

Given one real-valued weight matrix W[out, in] and a representative input matrix X[samples, in], compare symmetric uniform weight-only fake quantization policies. Show errors in W and in Y = X @ W.T separately, expose ranking reversals, and export a self-contained interactive report. Everything runs locally on a CPU with NumPy. User data is never uploaded by the program.

No model downloads, checkpoint parsing, activation quantization, bias, training, graph conversion, packed deployment weights, automatic tuning, inference kernels, task-accuracy claims, or speed/VRAM claims. The synthetic demo is explicitly labeled synthetic and is not a model benchmark. Results on supplied inputs do not generalize to unseen inputs. Large weights are not automatically causal explanations.

## Quantization

`Config(bits=4, mode="channel", group_size=32, percentile=100.0)` is immutable. Bits must be an integer in [2, 8], not bool. Mode is tensor, channel, or group. group_size is a positive integer (not bool); only group mode uses it. Percentile is a finite real in (0, 100]. Unknown fields are rejected by the CLI.

- qmax = 2**(bits-1) - 1; integer code interval is [-qmax, qmax], e.g. [-7, 7] for 4 bits. This is a narrow symmetric range, not all signed integer codes.
- Each scale unit uses t = percentile(abs(W_unit), percentile, method="linear"); p100 is absmax. Tensor mode has one unit, channel mode one per output row, group mode contiguous groups along each row's input axis. A final short group is included.
- Scale s = t / qmax when t > 0; otherwise s = 1. Clip weights to [-t,t] first, then `rint(clipped / s)` (nearest, ties to even), clip integer codes to [-qmax,qmax], dequantize by multiplying s. All-zero units dequantize to zero. A zero percentile threshold clips a nonzero unit entirely to zero, not to its original values.
- clipped_count counts `abs(original) > t`, not values equal to a representable endpoint.
- Computation is float64. Input arrays must be finite, nonempty 2D float32/float64 arrays. Complex, object, bool, integer, empty or mismatched arrays are rejected. Do not mutate input arrays.
- Nonfinite intermediates/results or a positive threshold whose scale underflows to zero raise ValueError rather than silently producing a successful report. No exact bitwise reproducibility guarantee across BLAS/platforms.

## Metrics and limits

Weight MSE and maximum absolute error compare dequantized weights to W. Output metrics compare the independently evaluated `X @ dequantized.T` against `X @ W.T`. Relative output L2 is `norm(delta_Y)/norm(Y)`; when norm(Y) is zero it is null, including zero/zero. JSON must not contain NaN or Infinity. Per-output-channel weight MSE, output MSE and reference RMS are included. Metrics must not be mislabeled accuracy, perplexity or speed.

A ranking reversal is a strict pair: A has smaller weight MSE than B, but larger output MSE. Ties produce no reversal. No automatic winner is asserted across bit widths, scale counts or deployment costs. The heatmap uses mean absolute weight error per rectangular tile, at most 24 output-row tiles by 64 input-column tiles, with explicit half-open index bounds. It is an aggregation, not an exact per-weight view when tiled.

Default aggregate input limit: 4,000,000 elements across W and X. Default output limit: 4,000,000 elements. Up to 24 distinct configurations and 4096 output channels (to bound report size). Limits are checked before large float64 copies/matmul, but are work-size guards, not process-RSS guarantees. Configuration IDs are unique and deterministic. Float64 arithmetic overflow is rejected. Reports include NumPy version, exact policy string, logical-array fingerprints (dtype/shape/C-order values), shapes and sample label; no raw matrices or absolute file paths.

## Python interface and report schema

`quantize(weights, config) -> Quantized` exposes `.weights`, `.scales`, `.clipped_count`, `.scale_count`.

`analyze(weights, inputs, configs=None, *, title="RoundoffLab report", sample_label="user-provided", max_elements=4_000_000, max_output_elements=4_000_000) -> dict`.

`default_configs() -> list[Config]`: 4-bit tensor, 4-bit channel, 4-bit group32, 8-bit channel, 4-bit channel p99, 4-bit channel p95. ID: `int{bits}-{mode}` plus `-g{group_size}` for group and `-p{percentile}` for p != 100. Percentiles are normalized to float64. The historical `:g` spelling is retained only when it round-trips exactly; otherwise `float.hex()` encodes the value losslessly. Thus released IDs such as `int4-channel-p99`, `int4-channel-p95`, and `int4-channel-p75` remain unchanged, while close values such as 99.1234567 and 99.1234568 cannot collide.

`make_demo(seed=7, scenario="quiet-outlier") -> (W, X)`. Scenarios quiet-outlier and active-outlier use identical synthetic W, with different excitation of its outlier input channel. Use deterministic RNG; demonstrate rather than assume at least one ranking reversal in quiet-outlier.

Report keys:

```json
{
  "schema_version": 1,
  "title": "RoundoffLab report",
  "sample_label": "user-provided",
  "shapes": {"weights": [12, 64], "inputs": [96, 64], "output": [96, 12]},
  "provenance": {"weight_sha256": "hex", "input_sha256": "hex", "numpy_version": "version", "numerical_policy": "float64; symmetric narrow-range; ties-to-even; linear percentile"},
  "configurations": [{
    "id": "int4-channel", "bits": 4, "mode": "channel", "group_size": null, "percentile": 100.0,
    "metrics": {"weight_mse": 0.1, "weight_max_abs_error": 0.2, "output_mse": 0.3, "output_max_abs_error": 0.4, "relative_output_l2": null, "clipped_count": 0, "scale_count": 12},
    "channels": [{"index": 0, "weight_mse": 0.1, "output_mse": 0.3, "reference_rms": 0.0}],
    "heatmap": {"metric": "mean_absolute_weight_error", "values": [[0.1]], "row_bounds": [[0, 12]], "col_bounds": [[0, 64]]}
  }],
  "ranking_reversals": [{"lower_weight_error": "id-A", "lower_output_error": "id-B"}],
  "warnings": ["Single-layer fake quantization is not a model-accuracy or performance benchmark."]
}
```

Numbers above illustrate the schema; they are not evidence or an analysis fixture.

## Local report and CLI

`render_report(report: dict) -> str` returns a standalone HTML document with embedded CSS/JS/data and no network dependencies. Selected configuration updates metrics, aggregated heatmap and channel bars. A complete native HTML comparison table is visible even without JS. A sort control changes table order; a JSON download button exports the same report. All user strings are text/escaped data, never executable HTML. Embedded JSON must escape `<`, `>`, `&`, U+2028/U+2029. No file picker or upload service.

Visual direction: an instrument notebook, slate-blue/navy with cool white and restrained orange error marks. System sans typography for interface, monospace for numbers. Signature: weight-error vs output-error comparison with a measured reversal explanation. No decorative neon, stock images, external fonts or generic landing-page hero. Keyboard focus, contrast, mobile layout and reduced motion are required.

Commands: `roundofflab demo --out DIR [--scenario quiet-outlier|active-outlier] [--force]` generates public synthetic .npy inputs plus report.json/report.html. `roundofflab analyze --weights W.npy --inputs X.npy --out DIR [--configs C.json] [--title TEXT] [--force]` analyzes user inputs. `.npy` only, mmap read-only and allow_pickle=False; never execute model code. Outputs are not overwritten without --force. Analysis refuses outputs aliasing an input. Writes stage new bytes and back up all old artifacts before publishing. Ordinary failures roll back; rollback failures retain recoverable backups and report their location. NTFS/POSIX hard-link publication is required. This is not a crash-atomic multi-file transaction: process termination/power loss and concurrent writers remain outside the guarantee. Validation failures exit 2 with actionable stderr and create no success artifact. Successful diagnostics exit 0; errors are measurements, not a universal pass/fail gate. stdout is a concise artifact summary, not tensor data.

## Validation and positioning

Use hand-calculated quantization including ties, zero units, clipping and tail groups; an independent scalar reference; output identity X dW.T up to float64 rounding; an isotropic-input ranking control; same-W/different-X reversal; invalid and oversized inputs; input immutability; actual CLI round trips; script-injection strings; offline browser interaction and clean wheel installation. Test failure injections must alter the real production path and be detected, not just prove a fixture is wrong.

ORT already matches weights and activations between original and quantized ONNX graphs: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html . LiteRT has a quantization debugger: https://developers.google.com/edge/litert/models/quantization_debugger . GPTQ uses layer-output reconstruction: https://arxiv.org/abs/2210.17323 . RoundoffLab's narrower workflow is direct matrix-in, local shareable evidence-out without conversion or a whole model. It is not a replacement for these tools or an original quantization method.
