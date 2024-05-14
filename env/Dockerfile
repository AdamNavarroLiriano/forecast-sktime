FROM python:3.8-slim

RUN apt-get -y update
RUN apt-get -y install git libgomp1

# set working directory
WORKDIR /workspace

# install dependencies
COPY ./requirements.txt /workspace
RUN pip install --no-cache-dir -r requirements.txt

# copy everything to folder
COPY . /workspace

# Install jupyterlab
RUN pip install --upgrade pip
RUN pip install --no-cache-dir jupyterlab

EXPOSE 8888

