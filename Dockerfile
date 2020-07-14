FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN pip install --upgrade pip

COPY . .