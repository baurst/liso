#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

docker compose build base
