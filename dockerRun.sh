##dockerRun.sh
## Author: Logan Caffey

docker run  --gpus all  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(pwd)":/workspace housecleanbot:latest bash
