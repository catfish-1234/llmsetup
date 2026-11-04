# Security policy

## Supported code

Security fixes target the current `main` branch and the 0.1.x development line.
There is no maintained historical-release support policy or guaranteed response
time. Use a supported Python interpreter and current inference engines.

## Report a vulnerability privately

Use [GitHub private vulnerability reporting](https://github.com/catfish-1234/llmsetup/security/advisories/new)
if the repository offers it. If that control is unavailable, open an issue
requesting a private reporting channel **without vulnerability details**, and
wait for a maintainer to provide one. Do not put an exploit, access token,
private model URL, or sensitive report in a public issue.

Include the affected commit or version, operating system and Python version,
a minimal reproduction using synthetic data, expected and actual behavior,
and the likely impact. A proposed patch or regression test is useful. Review
logs and reports before attaching them; they can contain local paths, hardware
details, endpoint URLs, custom configuration, and server output.

## Trust boundaries

- Catalog YAML, model metadata, GGUF files, and runtime responses are inputs,
  not trusted code. Use trusted publishers and review custom catalog overlays.
  `safe_load` prevents YAML object construction; it does not prove data is safe.
- Automatic catalog updates contact the configured source. Disable with
  `llmsetup catalog auto-update --disable` or `LLMSETUP_AUTO_UPDATE=0`; an
  `offline_only` profile skips that refresh. Real setup can still download
  weights through an inference engine.
- An `LLMSETUP_CATALOG_URL` or `HF_ENDPOINT` override is a trust decision. Never
  embed credentials in URLs. Set `HF_TOKEN` only for a publisher or mirror you
  trust, and grant the minimum read access needed.
- Engines run as your user. Keep them updated, bind endpoints to loopback,
  and do not expose the unauthenticated local UI to a network.
- Generated registrations write into application model directories. Use
  `--no-register` when that is unwanted. Keep model stores and configuration
  directories writable only by users you trust.

Current checks do not provide cryptographic model provenance, a sandbox for
inference engines, or a comprehensive secret audit. Runtime dependencies must
be audited separately from development tools; neither a passing audit nor a
passing smoke test guarantees the absence of vulnerabilities.
