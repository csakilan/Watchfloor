#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

kubectl apply -f "$DIR/manifest.yaml"
kubectl -n shop rollout status deployment/checkout --timeout=90s

kubectl -n shop patch configmap app-config \
  --type=json -p='[{"op": "remove", "path": "/data/DB_HOST"}]'
kubectl -n shop rollout restart deployment/checkout
