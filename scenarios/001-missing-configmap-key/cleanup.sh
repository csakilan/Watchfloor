#!/usr/bin/env bash
set -euo pipefail
kubectl delete namespace shop --ignore-not-found --wait=true
