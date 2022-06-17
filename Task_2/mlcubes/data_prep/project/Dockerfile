FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    python3-dev \
    curl && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa -y && apt-get update

RUN apt-get install python3 -y

RUN apt-get install python3-pip -y

COPY ./requirements.txt project/requirements.txt 

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir -r project/requirements.txt

# Set the locale
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

COPY . /project

WORKDIR /project

ENTRYPOINT ["python3", "/project/mlcube.py"]