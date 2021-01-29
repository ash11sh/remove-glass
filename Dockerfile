FROM python:3.6.12-slim-buster

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
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


COPY requirements.txt .


RUN pip install --upgrade -r requirements.txt
RUN pip install scikit-image
RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

COPY app app/

RUN python app/server.py

EXPOSE 5000

CMD ["python", "app/server.py", "serve"]
