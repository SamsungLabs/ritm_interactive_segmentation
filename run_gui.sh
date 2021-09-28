xhost +
docker run --rm -it \
-v "$PWD/src:/work" \
-v "$PWD/modeles:/models:ro"
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
ritm_interactive \
python3 demo.py --checkpoint /models/coco_lvis_h32_itermask.pth --cpu
