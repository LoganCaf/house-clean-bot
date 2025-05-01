## dockerfile that build the environment to run the sripts
## Author: Logan Caffey

FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

WORKDIR /workspace

ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DISPLAY=$DISPLAY

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 x11-apps && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


