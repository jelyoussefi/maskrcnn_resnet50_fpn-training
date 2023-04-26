FROM openvino/ubuntu20_dev:latest
ARG DEBIAN_FRONTEND=noninteractive

USER root
RUN apt update -y
RUN apt install -y  libgl1 libglib2.0-0  libx11-dev  python3-tk python3-opencv 
RUN pip3 install  onnxruntime-openvino
RUN pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html	
RUN apt install -y python3-dev
RUN pip3 install pycocotools albumentations
RUN mkdir -p  /workspace/
COPY train.py model.py /workspace/
WORKDIR /workspace 
