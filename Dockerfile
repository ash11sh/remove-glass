FROM python:3.9-slim-buster

RUN apt-get -y update \
    && apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*
# create a non-root user
RUN useradd -m -u 1000 user
# switch to non-root user and create a virtual environment
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
RUN python -m venv $HOME/venv
ENV PATH="$HOME/venv/bin:$PATH"
# copy requirements and install packages
COPY --chown=user:users requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --upgrade -r requirements.txt
RUN pip install scikit-image
RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
# set working directory and copy app code
WORKDIR $HOME/app
COPY --chown=user:users . .
# start server
EXPOSE 7860
CMD ["python", "app/server.py", "serve"]
