FROM nvidia/cuda:12.3.2-base-ubuntu22.04

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    python-is-python3 \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=.

ENTRYPOINT ["python","-m"]
CMD ["scripts.train_joint_autoencoder"]