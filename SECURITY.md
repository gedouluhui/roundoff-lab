# Security

## Supported versions

Only the current `world-class/v1` source line for **RoundoffLab 0.2.0a1** is considered for fixes. Older snapshots are unsupported. This is a best-effort project with no SLA, production support, or guarantee that every reported issue will receive a fix.

## Reporting a vulnerability

Do not include credentials, private data, proprietary inputs, personal paths, or unreduced production artifacts in public issues. Report vulnerabilities privately through [GitHub Security Advisories](https://github.com/gedouluhui/roundoff-lab/security/advisories/new). Include the affected version or commit, operating system, runtime/dependency versions, a minimal sanitized reproduction, and the security impact you observed.

## Security boundaries

Inputs are read-only NumPy files loaded with allow_pickle=False; no model code is executed or uploaded. Reports and fingerprints may still reveal sensitive matrix characteristics.

Hashes and fingerprints identify bytes or normalized inputs; they are not signatures or proof of trust. Review generated reports before sharing because metadata, names, prompts, queries, or previews may still be sensitive.
