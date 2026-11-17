# Candidate validation studies (v0.2 contract)

## Purpose

Compare actual, already-dequantized weight candidates and/or RoundoffLab simulations on two explicitly provided input cohorts. Select using discovery only, then expose the fixed candidate's validation outcome. This adds a real matrix-to-evidence workflow, not a new quantizer or a statistical evaluation platform.

The v0.1 `analyze`, `demo`, report schema 1 and numerical conventions remain unchanged. New `study` produces schema 2 with `report_type="validation_study"`. No checkpoint code, packed weights, inference service, network requests, optional-framework imports at runtime or GPU claims.

## Inputs and APIs

```python
study(weights, discovery, validation, *, candidates=None, configs=None,
      selected=None, selection_metric="propagated_mse", batch_rows=256, max_elements=4_000_000,
      max_output_elements=4_000_000, title="Candidate validation study") -> dict
```

W and candidate matrices use `[out,in]`; discovery and validation use `[samples,in]`. All inputs are finite, nonempty float32/64 matrices. External candidates are a mapping of ASCII names (1–64 letters/digits/dot/underscore/hyphen) to arrays. They must already include their quantization scale/zero-point and use the exact same coordinate order as W; matching shape cannot establish semantic alignment. No second quantization is applied. Report IDs are `external:<name>`; origin is `external_dequantized`, with `bits`, `scale_count` and `clipped_count` null (not inferred). The candidate fingerprint is recorded; the label is user-declared, not a verified quantizer identity.

`configs` is an explicit list of Config objects. When candidates is absent/empty and configs is None, use the six existing defaults. When external candidates are supplied and configs is None, compare only those external candidates. Explicit empty combined candidate sets and duplicate IDs are rejected. Maximum 24 candidates, maximum 4096 output channels. Aggregate logical input elements count W, both cohorts, and all external matrices; check this budget before float64 copies. Each cohort's logical output element count is separately limited. These are configurable work-size guards, not peak-RSS or runtime promises.

`batch_rows` is an integer 1–4096, not bool. Outputs are evaluated in row batches; no full N×O candidate/reference outputs are retained. Per-output-channel scaled sums of squares yield RMS/MSE. Nonfinite arithmetic and a representable nonzero RMS/squared error underflowing to zero are rejected, never silently called zero. The study canonicalizes base/candidate and each input batch to C-contiguous float64, then uses the same row-wise matrix-vector contraction and ordered per-row statistics irrespective of batch size; this prevents BLAS GEMM/GEMV dispatch changes from changing selection merely when the storage batch size changes. This reference-oriented path may be slower than a bulk matrix multiply. Floating-point/BLAS differences across environments remain possible; this is not an arbitrary-precision oracle. Cohort provenance describes this evaluation policy; quantization policy is per simulated candidate only and null for external candidates.

## Selection and evidence

Two explicitly different errors are measured: `output_mse = mean((X@Q.T - X@W.T)**2)` from separately rounded forward evaluations (the v0.1 definition), and `propagated_mse = mean((X@(Q-W).T)**2)` from direct float64 weight-error propagation. They need not match when forward rounding masks a small difference; neither is arbitrary-precision truth or actual deployment-kernel error. A diagnostic records when observed error is zero while propagated error is nonzero.

Default selection: smallest discovery `propagated_mse`, ties broken by candidate ID. `selection_metric` may instead be explicitly set to `output_mse`; reject other values. With `selected`, use exactly that known report ID; never select/reselect using validation. The selection section records metric, method, fixed ID, discovery and validation values, exact validation-best IDs, and nonnegative excess validation metric against the best supplied candidate. This is descriptive within this candidate set, not significance, accuracy, fairness or a generalization guarantee.

Each cohort contains a schema-1-shaped comparison view with origin metadata, existing metrics, channel metrics and heatmap. New evidence contains at most five worst samples, their row index, maximum absolute error and output-channel coordinate; it contains no raw tensor values. Fingerprints include dtype/shape/C-order bytes, so different encodings can differ for mathematically identical values. The report always states that independent sampling/leakage cannot be established. Identical numeric cohorts (including dtype changes) additionally trigger an explicit reuse warning; partial overlap is not certified absent.

The selection panel stays fixed while the reader switches cohorts. Table, ranking chart, metrics, tile/channel views, worst-sample evidence and provenance must all switch together. The JSON download always exports the complete study, not just the current view. Static/no-JS output includes both-cohort summary values and the frozen selection.

## CLI

`roundofflab study --weights W.npy --discovery D.npy --validation V.npy --out DIR [--candidates candidates.json] [--configs policies.json] [--selected ID] [--batch-rows N] [--force]`.

Candidate manifest: a JSON object mapping candidate names to `.npy` paths. Relative paths resolve from the manifest's directory, not current working directory. Reject duplicate keys, non-string paths, empty objects, excess candidates and files over 1 MB. Inputs are read-only mmap with allow_pickle=False. All candidate/config/cohort paths join the output-alias protection list. The old CLI is unchanged. Reports use the existing staged publication/rollback behavior. API/CLI failures do not produce a successful study artifact.

## Sources, examples and acceptance

PyTorch quantized tensors expose `dequantize()` (https://docs.pytorch.org/docs/stable/generated/torch.Tensor.dequantize.html). ONNX `DequantizeLinear` uses `(x - zero_point) * scale` with axis/block semantics (https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html). RoundoffLab does not implement or certify those formats. Users export float32/64 matrices after applying the real framework's dequantization and any layout restoration. Passing integer codes as weights is rejected by dtype but float-cast codes cannot be detected as semantically wrong.

Required evidence: hand-computed two-candidate selection reversal, validation changes cannot alter selected ID, explicit selection honored, identical-candidate/zero-reference controls, non-square axis oracle, external arrays are not re-quantized, same-number/different-dtype reuse warning, bad shape/dtype/nonfinite/budget/manifest cases, input bytes unchanged, block-size equivalence, stable large-number arithmetic, full CLI round trip and output alias protection. Test the real PyTorch export path as an optional, separately reported integration with no runtime dependency. Never train/select on validation in the example. Use a synthetic selection counterexample plus an optional real PyTorch quantized-tensor export (including a deliberately omitted-zero-point counterexample); both use synthetic matrices. No training or pretrained-model benchmark is implied.

Browser acceptance: complete cohort switch consistency, fixed selection, full JSON download, worst coordinates, 375px overflow, long IDs, keyboard navigation, escaping/no unexpected network and visual rendering where tools permit. Large heatmap selections remain accessible through row/column selectors.

## Compatibility and release status

This is the 0.2.0a1 alpha contract. Schema-1 analyses remain supported; validation studies use explicit schema version 2. The stable v0.1 hosted example does not imply this alpha has been published. No model-quality, deployment-performance or external-adoption result is claimed.
