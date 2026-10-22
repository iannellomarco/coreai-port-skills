---
name: coreai-model-release
description: Prepare and verify a public Hugging Face Core AI model release with a portable package layout, resolvable pinned requirements, upstream notices, checksums, privacy scans, honest model cards, charts, tags, and fresh-download smoke tests. Use whenever an agent is about to publish or update a `.aimodel` package.
---

# Purpose

Make the download useful to a stranger on a clean machine. Treat packaging, licensing, dependency resolution, privacy, and claims as release gates.

## Use this when

- A Core AI model or runtime is ready for public distribution.
- A release needs Python and Swift entry points, assets, tests, or charts.
- Requirements or attribution were changed after a previous release.

## Do not use this when

- You are only making a private local experiment.
- Publication authorization is missing. Prepare locally, but do not upload.

## Prerequisites

- A release directory staged outside the working export area.
- Final model assets and runtime sources.
- Upstream licenses, notices, model card links, and adapted-file provenance.
- A smoke test that does not download hidden dependencies at runtime.

## Procedure

1. Use a stable package layout.

   ```text
   package/
     README.md  LICENSE  NOTICE  CHECKSUMS.sha256
     models/    tokenizer/  python/  swift/  tests/  assets/
   ```

   Keep model assets, tokenizer/processor files, runtime code, tests, and charts in obvious directories. Include metadata describing functions, shapes, precision, source revision, and hardware used for measurements.

2. Preserve every applicable license.

   Keep the base model license and attribution for derived weights/tokenizer material. Keep BSD/Apache notices for adapted runtime files. State which files were adapted and under which original terms. Do not relicense upstream material by copying a repository-wide license into it.

3. Pin only versions that resolve together.

   Write a requirements file from a fresh environment, not from a machine that already has packages. Recreate the environment with Python 3.12 and install the file. A pin that looks precise but yields `ResolutionImpossible`, or omits a runtime dependency, is a release blocker.

4. Run the smoke test before checksums.

   ```bash
   python3.12 -m venv /tmp/coreai-release-venv
   /tmp/coreai-release-venv/bin/pip install -r python/requirements.txt
   /tmp/coreai-release-venv/bin/python tests/smoke_test.py
   ```

   Run Swift build/smoke separately with the supported Xcode SDK. Verify the default compute request, a forced GPU control where useful, tokenizer loading, audio/image preprocessing, and model load.

5. Scan for private material.

   ```bash
   Run the repository privacy checker for personal paths, usernames, private app names, machine identifiers, and credentials. Keep the scan itself out of the public package if it would repeat the sensitive terms it is designed to find.
   ```

   Replace personal paths with `<lab>/`, `~/models/`, or generic placeholders. Remove machine names, IP addresses, private app details, internal plans, and credentials. Public artifact names are allowed only where they are intentional and documented.

6. Write an honest model card.

   Include architecture, source model and license, modifications, supported OS/SDK, package units, hardware, first-load and warm-load behavior, process and neural memory, compute placement, evaluation data and normalizers, biggest regressions, and unmeasured languages/domains. Say when results are one machine or one run.

7. Generate charts from checked-in data.

   Keep the table or JSON used to make each chart. Label axes, units, p50/p95, warm/cold, and whether the measurement is end-to-end or graph-only. Do not make a chart imply that an unmeasured component was shipped.

8. Create checksums after all files are final.

   ```bash
   find . -type f ! -name CHECKSUMS.sha256 -print0 | sort -z | xargs -0 shasum -a 256 > CHECKSUMS.sha256
   ```

   Use a path format your verifier understands and test a deliberate mismatch.

9. Verify from a fresh copy.

   Use [`scripts/release_check.py`](scripts/release_check.py) for required files, checksums, and privacy. Copy the staged package to a fresh temporary directory, then run the documented smoke command from that copy. If the public host is used, download the exact uploaded revision into a new directory and repeat the same checks.

   ```bash
   python scripts/release_check.py . --fresh-copy \
     --smoke-cmd python tests/smoke_test.py
   ```

10. Version the claims, not just the files.

   Add a changelog entry for asset bytes, runtime behavior, source revisions, supported languages, and known regressions. Use tags or immutable revisions for reproducible downloads.

11. Stop before external publication unless authorized.

   Uploading, moving a repository, or publishing a public model is an outward-facing action. The local package can be fully verified without doing it.

## Decision points

- If a dependency cannot be installed in a fresh environment, fix the pin or mark the runtime unsupported.
- If a model-derived file has a stronger upstream license, retain that term in the package.
- If the Swift runtime is not numerically identical to Python, publish the measured delta and keep Python as the reference.
- If an asset was not included in the measured pipeline, do not combine its size with the benchmark result.

## Verification gates

- Fresh Python install and smoke pass.
- Swift build and smoke pass where supported.
- Privacy scan is clean.
- Checksums validate.
- Model card includes regressions, units, hardware, and unmeasured claims.
- Fresh-copy and fresh-download verification pass.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Fresh install fails | Conflicting pins or missing dependency | Rebuild from an empty venv and update the lock. |
| Smoke works locally only | Hidden cache, path, or downloaded file | Use local-only assets and a fresh copy. |
| Checksum changes after release | Files were edited after hashing | Re-run smoke, privacy, then checksums. |
| Model card overclaims | Graph-only numbers presented as shipped pipeline | Separate measured components and state limits. |
| Notice is missing | Adapted third-party file was copied without terms | Restore original notice and file-level provenance. |

## References and scripts

- [`references/release-layout.md`](references/release-layout.md) has a card outline and attribution checklist.
- [`scripts/release_check.py`](scripts/release_check.py) is a local, read-only package checker.

## Evidence notes

The checker is intentionally local and read-only. It does not upload, move, or publish a repository. A real fresh-download check must be performed against the exact public revision after authorization, using a new directory and a new environment. Keep that result separate from the local package check.

Do not delete a failed release directory until the failure, command, and dependency-resolution output are copied into the release notes. The most useful release fixes are often missing-dependency or attribution fixes, not model changes.

Required release artifacts:

```text
model card
license and notices
requirements
smoke fixtures
checksum file
privacy result
fresh-copy result
```

## Done when

- A stranger with the documented OS/toolchain can install, smoke-test, and run the package from a fresh directory.
- Every public number and license statement has an evidence trail.
- External publication, if desired, is a separate authorized step.

