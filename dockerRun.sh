##dockerRun.sh
docker run  --gpus all  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)":/workspace -w /workspace  nvcr.io/nvidia/tensorflow:25.02-tf2-py3 bash
