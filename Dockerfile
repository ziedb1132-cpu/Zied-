FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy pandas requests scikit-learn lightgbm \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY zied_quant.py .
VOLUME /app/zied_data
CMD ["python", "-u", "zied_quant.py", "run"]
