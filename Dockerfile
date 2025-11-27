FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV VENV=/opt/venv
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

RUN python -m venv $VENV
ENV PATH="$VENV/bin:$PATH"
RUN python -m pip install --upgrade pip

RUN pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128

RUN pip install \
  prettytable==3.16.0 \
  flask==3.1.0 \
  flask-cors==5.0.1 \
  simplejson==3.20.1 \
  tensorboard \
  tqdm \
  scipy \
  openai \
  datasets \
  requests

WORKDIR /workspace
COPY . .
ENV PYTHONPATH=/workspace:/workspace/scripts

ENTRYPOINT ["python","-m","scripts.train_joint_autoencoder"]
# ENTRYPOINT ["python","-m","scripts.train_imdb_people_decoder"]
