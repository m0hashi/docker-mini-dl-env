FROM rapidsai/rapidsai-dev:21.08-cuda11.2-devel-ubuntu20.04-py3.8
# FROM rapidsai/rapidsai:21.08-cuda11.2-runtime-ubuntu20.04-py3.8
ARG USERNAME=mluser
ARG GROUPNAME=mluser
ARG UPASSWD=password
ARG GID=1000
ARG UID=1000
ARG CONDAENV=rapids

# We need to redefine ARG TENSORFLOW_VERSION etc... here to get the default ARG value defined above the FROM instruction.
# See: https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ENV _USERNAME=${USERNAME}

# Add sudo command
RUN apt-get update && \
    apt-get install sudo && \
    apt-get autoremove -y && \
    apt-get clean


# Add user
RUN groupadd ${GROUPNAME} -g ${GID} && \
    useradd -u ${UID} -g ${GROUPNAME} -m ${USERNAME} --create-home --shell /bin/bash && \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    echo "Set disable_coredump false" >> /etc/sudo.conf && \
    echo ${USERNAME}:${UPASSWD} | chpasswd && \
    find /opt/conda -type f,d -print0 | xargs -0 -P 4096 -L 100 chown -R ${UID}:${GID} 


COPY --chown=${USERNAME}:${USERNAME} clean-layer.sh  /tmp/clean-layer.sh
COPY --chown=${USERNAME}:${USERNAME} ./install-list/apt-install-list.txt /tmp/apt-install-list.txt

ENV TZ Asia/Tokyo
RUN apt-get update \
 && echo "${TZ}" > /etc/timezone \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
 && rm /etc/localtime \
 && ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
 && rm -rf /var/lib/apt/lists/* \
 && dpkg-reconfigure -f noninteractive tzdata \
 && apt-get clean
RUN apt-get update && \
    apt-get install -y $(grep -vE "^\s*#" /tmp/apt-install-list.txt  | tr "\n" " ") && \
    apt-get autoremove -y && \
    apt-get clean

# Install OpenCL & libboost (required by LightGBM GPU version)


# # Make sure the dynamic linker finds the right libstdc++
ENV LD_LIBRARY_PATH=/opt/conda/envs/rapids/lib
# b/128333087: Set PROJ_LIB to points to the proj4 cartographic library.
ENV CUDA_MAJOR_VERSION=11
ENV CUDA_MINOR_VERSION=2
ENV CUDA_VERSION=$CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/bin:${PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda
RUN apt-get install -y  ocl-icd-libopencl1 clinfo libboost-all-dev && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    apt-get autoremove -y && \
    apt-get clean

USER ${USERNAME}
SHELL ["conda", "run", "--no-capture-output", "-n", "rapids", "/bin/bash", "-c"]

# Install LightGBM with GPU
RUN sudo mkdir /usr/local/src/LightGBM && \
    sudo chown ${USERNAME}:${USERNAME} /usr/local/src/LightGBM
RUN cd /usr/local/src && \
    git clone --recursive https://github.com/microsoft/LightGBM && \
    cd LightGBM && \
    git checkout tags/v3.2.1 && \
    mkdir build && cd build && \
    cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ .. && \
    make -j$(nproc) && \
    cd /usr/local/src/LightGBM/python-package && \
    python setup.py install --precompile && \
    sudo mkdir -p /etc/OpenCL/vendors && \
    sudo sh -c "echo 'libnvidia-opencl.so.1' > /etc/OpenCL/vendors/nvidia.icd" && \
    /tmp/clean-layer.sh

RUN conda install  mkl -c conda-forge -c default && \
    /tmp/clean-layer.sh

ARG TORCH_CUDA_MINOR_VERSION=1
RUN pip install torch==1.9.0+cu${CUDA_MAJOR_VERSION}${TORCH_CUDA_MINOR_VERSION} torchvision==0.10.0+cu${CUDA_MAJOR_VERSION}${TORCH_CUDA_MINOR_VERSION} torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html



# Install JAX (Keep JAX version in sync with CPU image)
ARG JAX_CUDA_MINOR_VERSION=1
RUN pip install jax==0.2.16 jaxlib==0.1.68+cuda$CUDA_MAJOR_VERSION$JAX_CUDA_MINOR_VERSION -f https://storage.googleapis.com/jax-releases/jax_releases.html && \
    /tmp/clean-layer.sh

# Reinstall packages with a separate version for GPU support.
RUN pip install tensorflow-gpu==2.6.0 && \
    pip install tensorflow-gcs-config==2.6.0 && \
    pip install tensorflow-addons==0.14.0 && \
    pip install mxnet-cu$CUDA_MAJOR_VERSION$CUDA_MINOR_VERSION && \
    /tmp/clean-layer.sh

RUN pip install mlflow && \
    /tmp/clean-layer.sh
# RUN pip install jupyterlab==3.0 jupyterlab-lsp==3.8.0 jedi-language-server && \
RUN pip install jupyterlab==3.0 jupyterlab-lsp==3.8.0 && \
    /tmp/clean-layer.sh
RUN pip install jupyterlab_vim && \
    /tmp/clean-layer.sh

WORKDIR /home/${USERNAME}

RUN mkdir -p ~/.local && \
    mkdir -p ~/workspace && \
    mkdir -p ~/.ssh && \
    mkdir -p ~/.kaggle && \
    mkdir -p ~/.aws


COPY --chown=${USERNAME}:${USERNAME} docker-entrypoint.sh /home/${USERNAME}/.local/bin/docker-entrypoint.sh
COPY --chown=${USERNAME}:${USERNAME} run-jupyter.sh /home/${USERNAME}/.local/bin/run-jupyter.sh
COPY --chown=${USERNAME}:${USERNAME} run-mlflow.sh /home/${USERNAME}/.local/bin/run-mlflow.sh
RUN chmod 777 /home/${USERNAME}/.local/bin/docker-entrypoint.sh
RUN echo "PATH=~/.local/bin/:$PATH" >> ~/.bashrc
RUN conda init
RUN echo "conda activate ${CONDAENV}" >> ~/.bashrc

USER root
RUN mkdir /var/run/sshd