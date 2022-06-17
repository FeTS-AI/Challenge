# Please use one of the following base images for your container.
# This makes sure it can be run successfully in the federated evaluation.
FROM nvcr.io/nvidia/pytorch:20.08-py3
# FROM nvcr.io/nvidia/tensorflow:20.08-tf2-py3
# FROM nvcr.io/nvidia/tensorflow:20.08-tf1-py3

# fill in your info here
LABEL author="chuck@norris.org"
LABEL team="A-team"
LABEL application="your application name"
LABEL maintainer="chuck@norris.org"
LABEL version="0.0.1"
LABEL status="beta"

# basic
RUN apt-get -y update && apt -y full-upgrade && apt-get -y install apt-utils wget git tar build-essential curl nano

# install all python requirements
WORKDIR /mlcube_project
COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

# copy all files
COPY ./ /mlcube_project

# NOTE: to be able to run this with singularity, an absolute path is required here.
ENTRYPOINT [ "python3", "/mlcube_project/mlcube.py"]
