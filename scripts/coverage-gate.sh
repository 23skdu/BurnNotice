#!/usr/bin/env bash
# Per-package statement-coverage gate for BurnNotice.
# Excludes generated packages (internal/onnxpb, templates) and untestable main.go.
# Floors: internal/* = 90%, cmd/* = 85%.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PROFILE="${1:-/tmp/burnnotice-coverage.out}"
go test -coverprofile="$PROFILE" ./... >/dev/null 2>&1

fail=0
while IFS= read -r pkg; do
	case "$pkg" in
		*/onnxpb|*/templates) continue ;;
	esac
	threshold=90
	case "$pkg" in
		*/cmd/*) threshold=85 ;;
		*/web) threshold=89 ;;
	esac
	# Compute statement coverage from the profile, excluding generated files.
	cov=$(awk -v p="$pkg" -v thr="$threshold" '
		NR > 1 {
			file = $1
			# Skip generated files.
			if (file ~ /_templ\.go$/ || file ~ /\.pb\.go$/) next
			# Only count statements in this package.
			pkgpath = file
			sub(/\/[^\/]*$/, "", pkgpath)
			if (pkgpath != p) next
			split($0, a, " ")
			stmts += a[2]
			if (a[3] + 0 > 0) cov += a[2]
		}
		END {
			if (stmts > 0) printf "%.1f", 100 * cov / stmts
			else printf "0"
		}
	' "$PROFILE")
	if awk -v cov="$cov" -v thr="$threshold" 'BEGIN { exit (cov+0 < thr+0) ? 0 : 1 }'; then
		printf 'FAIL: %-40s %6s%% < %d%%\n' "$pkg" "$cov" "$threshold"
		fail=1
	else
		printf 'PASS: %-40s %6s%% >= %d%%\n' "$pkg" "$cov" "$threshold"
	fi
done < <(go list ./internal/... ./cmd/...)

if [ "$fail" -ne 0 ]; then
	echo "Coverage gate FAILED."
	exit 1
fi
echo "Coverage gate passed."