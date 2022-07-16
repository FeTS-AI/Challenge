# Official base image
FROM ubuntu:18.04

# Install/Update Ubuntu packages, virtual environment, python dependencies and UTF-8 locale support.
RUN : \
    && apt-get update \
    && apt-get install -y build-essential python3.7 python3.7-venv python3-venv git locales \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 \
    && :

# Init virtual environment and add to PATH
RUN python3.7 -m venv /fetsenv
ENV PATH=/fetsenv/bin:$PATH

# Locale environment variable to support/render certain text
ENV LANG en_US.utf8

# Add UTF-8 support
ENV PYTHONUTF8 1

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Mount Point
VOLUME [ "/data" ]

# Upgrade pip and setup tools
RUN pip install --upgrade pip setuptools

# Application base directory
WORKDIR /workspace

# Install pip package directly
# pip install -y torch==1.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
# OR 
# through requirements file.
ADD requirements.txt /workspace
RUN python -m pip install -r requirements.txt

# Copy over application files and install packaged/distributed python modules. 
ADD Task_1/ /workspace
RUN pip install .

# Publish the exposed port 80 and map to the higher order ports
EXPOSE 80

# Command to execute when container is run.
CMD ["python", "FeTS_Challenge.py"]
