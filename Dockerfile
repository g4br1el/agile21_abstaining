FROM python:3.7.10-buster
ADD requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt
