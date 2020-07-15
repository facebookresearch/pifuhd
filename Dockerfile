FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN apt-get install -y libsm6 libxrender-dev libxext6 libglib2.0-0 freeglut3-dev gcc g++
RUN pip install --upgrade pip

COPY requirements.txt .

#install python package
RUN pip install -r requirements.txt

COPY . .