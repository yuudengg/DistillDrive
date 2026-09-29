ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace/DistillDrive \
    TORCH_CUDA_ARCH_LIST=12.1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        git \
        libglib2.0-0 \
        libgl1 \
        ninja-build \
        python3-lib2to3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/DistillDrive

COPY docker/requirements-gb10.txt /tmp/requirements-gb10.txt
COPY docker/mmcv_ext_stub.py /tmp/mmcv_ext_stub.py
COPY docker/patch_mmcv_focal_loss.py /tmp/patch_mmcv_focal_loss.py
COPY docker/patch_mmcv_distributed.py /tmp/patch_mmcv_distributed.py

# NGC images set a global constraints file. DistillDrive deliberately needs an
# older OpenMMLab API generation, so only this installation opts out of it.
# Torch and flash-attn remain the Blackwell-compatible builds from the base.
RUN PIP_CONSTRAINT= python -m pip install --no-cache-dir \
        -r /tmp/requirements-gb10.txt \
    && PIP_CONSTRAINT= python -m pip install --no-cache-dir \
        --no-build-isolation --no-deps mmcv==1.7.1 \
    && PIP_CONSTRAINT= python -m pip install --no-cache-dir --no-deps \
        mmdet==2.28.2 nuscenes-devkit==1.1.11 \
    && cp /tmp/mmcv_ext_stub.py \
        /usr/local/lib/python3.12/dist-packages/mmcv/_ext.py \
    && sed -i "s/plt.style.use('seaborn-whitegrid')/plt.style.use('seaborn-v0_8-whitegrid')/" \
        /usr/local/lib/python3.12/dist-packages/nuscenes/map_expansion/map_api.py \
    && sed -i \
        "s/streams = \[_get_stream(device) for device in target_gpus\]/streams = [_get_stream(torch.device(f'cuda:{device}')) for device in target_gpus]/" \
        /usr/local/lib/python3.12/dist-packages/mmcv/parallel/_functions.py \
    && cat /tmp/patch_mmcv_focal_loss.py >> \
        /usr/local/lib/python3.12/dist-packages/mmcv/ops/focal_loss.py \
    && cat /tmp/patch_mmcv_distributed.py >> \
        /usr/local/lib/python3.12/dist-packages/mmcv/parallel/distributed.py \
    && sed -i \
        's/torch\.load(\([^)]*\), map_location=map_location)/torch.load(\1, map_location=map_location, weights_only=False)/g' \
        /usr/local/lib/python3.12/dist-packages/mmcv/runner/checkpoint.py \
    && sed -i \
        's/from collections import OrderedDict, Iterable/from collections import OrderedDict\nfrom collections.abc import Iterable/' \
        /usr/local/lib/python3.12/dist-packages/motmetrics/metrics.py \
    && sed -i \
        's/inspect\.getargspec/inspect.getfullargspec/g' \
        /usr/local/lib/python3.12/dist-packages/motmetrics/metrics.py

COPY . .

RUN cd projects/mmdet3d_plugin/ops \
    && FORCE_CUDA=1 python setup.py build_ext --inplace \
    && find . -maxdepth 2 -type d -name build -prune -exec rm -rf {} +

CMD ["bash"]
