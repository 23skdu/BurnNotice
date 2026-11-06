#!/usr/bin/env bash
# Generates a reproducible release evidence bundle for BurnNotice (Roadmap Item 20).
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

OUT_DIR="${1:-dist/evidence}"
mkdir -p "$OUT_DIR"

COMMIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
SHORT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
TIMESTAMP="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
GO_VER="$(go version)"
OS_ARCH="$(uname -srm)"
CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo unknown)"

DATASET_HASH="none"
if [ -d datasets ]; then
	DATASET_HASH="$(find datasets -type f -print0 2>/dev/null | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | awk '{print $1}')"
fi

cat > "$OUT_DIR/manifest.json" <<EOF
{
  "project": "BurnNotice",
  "commit_sha": "$COMMIT_SHA",
  "short_sha": "$SHORT_SHA",
  "timestamp_utc": "$TIMESTAMP",
  "go_version": "$GO_VER",
  "os_arch": "$OS_ARCH",
  "cpu_model": "$CPU_MODEL",
  "dataset_sha256": "$DATASET_HASH"
}
EOF

echo "==> Running unit, accuracy, and oracle gates for evidence bundle..."
go test ./... > "$OUT_DIR/unit_tests.txt"
go test -run='Accuracy|Invariant|Cosine|Recall|Rank|Threshold|Calibration' ./... > "$OUT_DIR/accuracy_gate.txt"
go test -count=2 -run='Oracle|Golden|Determin' ./... > "$OUT_DIR/oracle_gate.txt"

if [ -f benchmark_results.txt ]; then
	cp benchmark_results.txt "$OUT_DIR/benchmark_results.txt"
fi

(
	cd "$OUT_DIR"
	sha256sum manifest.json unit_tests.txt accuracy_gate.txt oracle_gate.txt > SHA256SUMS
)

echo "Release evidence bundle written to $OUT_DIR"
