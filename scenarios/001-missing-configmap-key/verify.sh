#!/usr/bin/env bash
set -uo pipefail

for _ in $(seq 1 30); do
  reasons=$(kubectl -n shop get pods -l app=checkout \
    -o jsonpath='{.items[*].status.containerStatuses[*].state.waiting.reason}' 2>/dev/null)
  if echo "$reasons" | grep -q CreateContainerConfigError; then
    echo "fault confirmed: checkout pod in CreateContainerConfigError"
    exit 0
  fi
  sleep 2
done

echo "fault NOT confirmed after 60s. Observed waiting reasons: ${reasons:-none}" >&2
exit 1
