# To build an image run:
#   docker build --tag otr .
# Run a shell inside the docker container with
#   docker run --runtime=nvidia --rm -it -v `pwd`:/workdir/otr --workdir=/workdir/otr otr:latest
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},graphics
ENV LANG=C.UTF-8

# Needed by nvidia-container-runtime, if used
ENV NVIDIA_VISIBLE_DEVICES \
        ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
        ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

# Install system dependencies
# Does a few things
# * Fetch the latest GPG key for CUDA
# * Install rendering dependencies for dm_control
# * Enable EGL support via libglvnd, see
#   https://gitlab.com/nvidia/container-images/opengl/blob/ubuntu20.04/glvnd/runtime/Dockerfile
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
        wget git unzip \
        patchelf \
        libglfw3 \
        libglew2.1 \
        libgl1-mesa-glx libosmesa6 libosmesa6-dev \
        libglvnd0 libgl1 libglx0 libegl1 libgles2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s $(which python3) /usr/local/bin/python
# See
#   https://gitlab.com/nvidia/container-images/opengl/-/blob/ubuntu20.04/glvnd/runtime/Dockerfile
# and
#   https://github.com/apptainer/singularity/issues/1635
#
RUN wget -q https://gitlab.com/nvidia/container-images/opengl/-/raw/ubuntu20.04/glvnd/runtime/10_nvidia.json \
    -O /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Install Miniconda
# TODO: we don't need conda
ARG CONDA_VERSION=py38_4.10.3
ENV PATH /opt/conda/bin:$PATH
# Update LD_LIBRARY_PATH to expose libpython3.8.so to Reverb and Launchpad
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/conda/lib
RUN set -x && \
    mkdir -p /opt && \
    wget -q https://repo.continuum.io/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

ENV MUJOCO_DIR=/opt/mujoco
ENV MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210
RUN mkdir -p ${MUJOCO_DIR} && \
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
    tar -C ${MUJOCO_DIR} -xvzf mujoco210-linux-x86_64.tar.gz

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MUJOCO_DIR}/mujoco210/bin:/usr/lib/nvidia
RUN mkdir -p /workdir
WORKDIR /workdir
ADD ./requirements /tmp/requirements
RUN python3 -m pip --no-cache-dir install -U pip wheel setuptools && \
    pip install --no-cache-dir -r /tmp/requirements/dev.txt && \
    rm -r /tmp/requirements && \
    # This forces compilation of the mujoco_py native extensions
    python3 -c "import mujoco_py"
# Apply a patch to mujoco_py to fix issues with locking on a read-only filesystem
ADD patches/mujoco_py.patch /tmp/mujoco_py.patch
RUN patch -u /opt/conda/lib/python3.8/site-packages/mujoco_py/builder.py -i /tmp/mujoco_py.patch

# Set up a user for running the application
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user
