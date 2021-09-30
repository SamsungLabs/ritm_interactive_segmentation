xhost +
docker run --rm -it \
-v "$PWD/src:/work" \
-v "$PWD/models:/models:ro" \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-p 33766:22 \
ritm_interactive \
bash
