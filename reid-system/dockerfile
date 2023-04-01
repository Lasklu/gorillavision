FROM nvcr.io/nvidia/pytorch:22.02-py3
WORKDIR /workspace
RUN apt-get update
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get install -y libglu1
WORKDIR /gorilla-reidentification/reid-system

