#!/bin/bash
set -e
set -u
set -o pipefail

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate liso

whoami
which python
which pip


nvidia-smi
which nvcc

echo $(pwd)

pushd config_helper
pip install --user -e .
popd

pushd nuscenes-devkit/python-sdk
pip install --user -e .
popd

pushd mmdetection3d
pip install --user -e .
popd

pushd iou3d_nms
python setup.py install --user --prefix=
popd

pip install --user -e .

echo "Installed packages:"
uname -a
python --version
pip freeze
conda list

echo "Installation finished!"