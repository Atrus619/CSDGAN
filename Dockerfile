FROM nvidia/cuda:10.0-base

WORKDIR /home/CSDGAN

COPY requirements.txt requirements.txt

# Install python
RUN apt-get update -y
RUN apt-get install python3 -y
RUN apt-get install python3-venv -y
RUN apt-get install redis -y
RUN apt-get install python-opencv -y

# Add necessary env vars
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES all

# Set up environment and install requirements
RUN python3 -m venv venv
RUN venv/bin/pip install --upgrade pip setuptools wheel
RUN venv/bin/pip install -r requirements.txt

# Copy over CSDGAN files
COPY CSDGAN CSDGAN
COPY utils utils
COPY config.py boot.sh ./
RUN chmod +x boot.sh

ENV FLASK_APP CSDGAN

# Expose port and run boot script
EXPOSE 5000
ENTRYPOINT ["./boot.sh"]