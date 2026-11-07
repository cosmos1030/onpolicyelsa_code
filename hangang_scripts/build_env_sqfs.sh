#!/bin/bash
# Build the SCOUT aarch64 env on the login node's local /tmp (outside the bs0229 /scratch file-count quota),
# then pack it into ONE squashfs file on /scratch. Jobs mount it back at the same path (/tmp/scout_env).
set -x
B=/tmp/doyoon_build; E=/tmp/scout_env; OUT=/scratch/t512a04/users/doyoon
mkdir -p $B $OUT
export CONDA_PKGS_DIRS=$B/pkgs CONDA_ENVS_PATH=$B/envs PIP_CACHE_DIR=$B/pip UV_CACHE_DIR=$B/uv PYTHONNOUSERSITE=1
module load miniforge/26.1.1-3
conda create -y -p $E python=3.10 --no-default-packages || exit 1
P=$E/bin/python
$P -m pip install -q -U pip uv || exit 1
$P -m uv pip install --python $P vllm==0.10.2 || exit 1
$P -m uv pip install --python $P -r $B/req_rest.txt 2>&1 | tail -20
$P -m uv pip install --python $P vllm==0.10.2 2>&1 | tail -3
# PyPI aarch64 torch is CPU-only; the cu129 index has the CUDA aarch64 build of the same version
$P -m uv pip install --python $P --reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129 || exit 1
$P -c "import torch, vllm, transformers, lighteval; assert torch.version.cuda; print('IMPORT_OK', torch.__version__, torch.version.cuda, vllm.__version__, transformers.__version__)" || exit 1
find $E -name "__pycache__" -prune -o -type f -print | wc -l
mksquashfs $E $OUT/scout_env.sqfs -noappend -processors 8 -quiet || exit 1
ls -la $OUT/scout_env.sqfs
rm -rf $B $E
sed -i "\#^/tmp/scout_env\$#d" ~/.conda/environments.txt
echo BUILD_DONE
