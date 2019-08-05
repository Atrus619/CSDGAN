FROM python:3.7.3-slim

#RUN adduser --disabled-login csdgan

WORKDIR /home/CSDGAN

COPY requirements.txt requirements.txt
RUN python3 -m venv venv
RUN venv/bin/pip install --upgrade pip setuptools wheel
RUN venv/bin/pip install -r requirements.txt
RUN venv/bin/pip install gunicorn

COPY CSDGAN CSDGAN
COPY utils utils
COPY config.py boot.sh ./
RUN chmod +x boot.sh

ENV FLASK_APP CSDGAN

#RUN chown -R csdgan:csdgan ./
#USER csdgan
#USER root

EXPOSE 5000
ENTRYPOINT ["./boot.sh"]