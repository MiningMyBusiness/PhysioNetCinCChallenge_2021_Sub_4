FROM tensorflow/tensorflow:2.4.1

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER bhattacharyyakiran12@gmail.com

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
