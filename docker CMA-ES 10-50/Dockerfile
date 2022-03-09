FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends build-essential r-base r-cran-randomforest python3.6 python3-pip python3-setuptools python3-dev git vim ffmpeg libsm6 libxext6

WORKDIR /

COPY requirements.txt /requirements.txt

RUN pip3 install git+https://github.com/risto-trajanov/nevergrad.git

RUN pip3 install --user -r requirements.txt

RUN Rscript -e "install.packages('irace')"

COPY . /

RUN sed -i -e 's/\r$//' /target-runner-linux
