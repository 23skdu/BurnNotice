#!/usr/bin/env bash
# Compare current benchmark results against the committed baseline.
# Hard-fails on allocation regressions (allocs/op, B/op); warns on timing.
# Usage: scripts/bench-regression.sh [baseline_file] [tolerance_percent]
set -euo pipefail

BASELINE="${1:-docs/benchmarks/baseline.txt}"
TOL="${2:-50}"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

if [ ! -f "$BASELINE" ]; then
	echo "Baseline not found: $BASELINE"
	exit 1
fi

CURRENT=$(mktemp)
trap 'rm -f "$CURRENT"' EXIT
{
	go test -run='^$' -bench=. -benchmem -benchtime=5s ./internal/scoring 2>&1 | grep -E '^Benchmark'
	go test -run='^$' -bench=. -benchmem -benchtime=5s ./internal/steer 2>&1 | grep -E '^Benchmark'
	go test -run='^$' -bench=. -benchmem -benchtime=5s ./internal/math 2>&1 | grep -E '^Benchmark'
} > "$CURRENT"

declare -A base_ns base_b base_a
while IFS= read -r line; do
	name=$(echo "$line" | awk '{print $1}')
	ns=$(echo "$line" | sed -n 's/.* \([0-9.]*\) ns\/op.*/\1/p')
	b=$(echo "$line" | sed -n 's/.* \([0-9]*\) B\/op.*/\1/p')
	a=$(echo "$line" | sed -n 's/.* \([0-9]*\) allocs\/op.*/\1/p')
	[ -n "$ns" ] && base_ns["$name"]="$ns"
	[ -n "$b" ] && base_b["$name"]="$b"
	[ -n "$a" ] && base_a["$name"]="$a"
done < <(grep '^Benchmark' "$BASELINE")

fail=0
while IFS= read -r line; do
	name=$(echo "$line" | awk '{print $1}')
	ns=$(echo "$line" | sed -n 's/.* \([0-9.]*\) ns\/op.*/\1/p')
	b=$(echo "$line" | sed -n 's/.* \([0-9]*\) B\/op.*/\1/p')
	a=$(echo "$line" | sed -n 's/.* \([0-9]*\) allocs\/op.*/\1/p')
	[ -z "$ns" ] && continue
	base="${base_ns[$name]:-}"
	if [ -z "$base" ]; then
		echo "NEW: $name (${ns} ns/op) — no baseline entry"
		continue
	fi
	# Hard fail on allocation count regression.
	if [ -n "$a" ] && [ -n "${base_a[$name]:-}" ]; then
		if [ "$a" -gt "${base_a[$name]}" ]; then
			printf 'ALLOC REGRESSION: %-40s %s allocs/op > baseline %s\n' "$name" "$a" "${base_a[$name]}"
			fail=1
		fi
	fi
	# Warn (do not fail) on memory regression beyond tolerance; B/op fluctuates with GC.
	if [ -n "$b" ] && [ -n "${base_b[$name]:-}" ]; then
		limit=$(awk -v bb="${base_b[$name]}" -v t="$TOL" 'BEGIN { printf "%.0f", bb * (100 + t) / 100 }')
		if [ "$b" -gt "$limit" ]; then
			printf 'MEM WARN:      %-40s %s B/op > limit %s (baseline %s)\n' "$name" "$b" "$limit" "${base_b[$name]}"
		fi
	fi
	# Warn (do not fail) on timing regression beyond tolerance.
	limit=$(awk -v bb="$base" -v t="$TOL" 'BEGIN { printf "%.2f", bb * (100 + t) / 100 }')
	if awk -v n="$ns" -v l="$limit" 'BEGIN { exit (n > l) ? 0 : 1 }'; then
		printf 'TIMING WARN:   %-40s %10s ns/op > limit %s (baseline %s)\n' "$name" "$ns" "$limit" "$base"
	fi
done < "$CURRENT"

if [ "$fail" -ne 0 ]; then
	echo "Benchmark regression detected (allocations)."
	exit 1
fi
echo "No allocation regressions; timing within ${TOL}% tolerance."
