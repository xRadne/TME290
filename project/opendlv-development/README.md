# opendlv-development

Microservice for development/debugging.

Run using:

docker build -t opendlv-development .

docker run --rm -ti --ipc=host --net=host -v /tmp:/tmp -e DISPLAY=$DISPLAY  opendlv-development --cid=111 --name=img.argb --width=1280 --height=720 --verbose
