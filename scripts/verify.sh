#!/usr/bin/env bash
# Git pre-commit quality gate for BurnNotice.
set -euo pipefail

FUZZ_TIME="${1:-${FUZZ_TIME:-1}}"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

GOSEC="${GOSEC:-$(command -v gosec 2>/dev/null || echo /home/rsd/go/bin/gosec)}"
GOVULNCHECK="${GOVULNCHECK:-$(command -v govulncheck 2>/dev/null || echo /home/rsd/go/bin/govulncheck)}"

echo "==> gofmt check"
unformatted="$(gofmt -l .)"
if [ -n "$unformatted" ]; then
    echo "Unformatted files:"
    echo "$unformatted"
    exit 1
fi

echo "==> go vet"
go vet ./...

echo "==> golangci-lint"
if command -v golangci-lint >/dev/null 2>&1; then
    golangci-lint run ./... || echo "golangci-lint reported issues (non-blocking in pre-commit hook)"
else
    echo "golangci-lint not installed; skipping"
fi

echo "==> gosec"
if [ -x "$GOSEC" ]; then
    "$GOSEC" ./...
else
    echo "gosec not installed; skipping"
fi

echo "==> govulncheck"
if [ -x "$GOVULNCHECK" ]; then
    "$GOVULNCHECK" ./...
else
    echo "govulncheck not installed; skipping"
fi

echo "==> unit tests"
go test ./...

echo "==> race tests"
go test -race ./...

echo "==> fuzz smoke (${FUZZ_TIME}s per target)"
set +e
fuzz_failed=0
for pkg in internal/math internal/extract internal/scoring internal/steer internal/metrics internal/zjson internal/converter internal/policy; do
    for fn in $(go test -list 'Fuzz' "./$pkg" 2>/dev/null | grep '^Fuzz'); do
        echo "  $pkg :: $fn"
        go test -run='^$' -fuzz="^${fn}$" -fuzztime="${FUZZ_TIME}s" "./$pkg"
        if [ $? -ne 0 ]; then
            fuzz_failed=1
        fi
    done
done
set -e
if [ "$fuzz_failed" -ne 0 ]; then
    echo "Fuzz smoke tests failed."
    exit 1
fi

echo "==> accuracy gate"
go test -run='Accuracy|Invariant|Cosine|Recall|Rank|Threshold|Calibration' -timeout 10m ./...

echo "==> oracle gate"
go test -count=2 -run='Oracle|Golden|Determin' -timeout 10m ./...

echo "All BurnNotice quality checks passed."
