# sample run command:
# docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e "DISPLAY=unix:0" -v /home/james:/work/output jmarca/pdptw_scratch bash
#
# build this with
# docker build -t jmarca/pdptw_scratch .
#

FROM python:3.6

RUN pip install --upgrade ortools numpy matplotlib

RUN mkdir -p /work

COPY *.py /work/

WORKDIR /work
