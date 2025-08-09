FROM tensorflow/tensorflow:latest-gpu

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/workspace:/workspace/scripts

ENTRYPOINT ["python","-m"]
CMD ["scripts.train_joint_autoencoder"]
