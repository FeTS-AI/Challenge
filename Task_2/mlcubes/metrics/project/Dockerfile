FROM cbica/captk:release-1.8.1

RUN yum install -y xz-devel

RUN cd /work/CaPTk/bin/ && \
    curl https://captk.projects.nitrc.org/Hausdorff95_linux.zip --output Hausdorff95_linux.zip && \
    unzip -o Hausdorff95_linux.zip && \
    chmod a+x Hausdorff95 && \
    rm Hausdorff95_linux.zip

# install all python requirements
RUN yum install python3 python3-pip -y

WORKDIR /project
COPY ./requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# copy all files
COPY ./ /project

# Set the locale
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# these produce problems with singularity
ENV CMAKE_PREFIX_PATH=
ENV DCMTK_DIR=

ENTRYPOINT ["python3", "/project/mlcube.py"]
