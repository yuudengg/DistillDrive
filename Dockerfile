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
        /usr/local/lib/python3.12/dist-packages/mmcv/_ext.py

COPY . .

RUN cd projects/mmdet3d_plugin/ops \
    && FORCE_CUDA=1 python setup.py build_ext --inplace \
    && find . -maxdepth 2 -type d -name build -prune -exec rm -rf {} +

CMD ["bash"]
