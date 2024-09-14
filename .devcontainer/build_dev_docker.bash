#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

script_dir="${BASH_SOURCE[0]:-$0}"
script_dir="$( dirname -- "$script_dir" )"

pushd ${script_dir}

UID_GID="$(id -u):$(id -g)" docker compose build liso_dev

popd
