FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y libgl1-mesa-glx
