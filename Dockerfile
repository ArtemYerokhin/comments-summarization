FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      wget \
      build-essential \
      libxml2-dev \
      libxslt-dev \
      zlib1g-dev \
      git \
 && wget https://download.cdn.yandex.net/mystem/mystem-3.1-linux-64bit.tar.gz \
 && tar -xvf mystem-3.1-linux-64bit.tar.gz \
 && mv mystem /usr/local/bin/ \
 && rm -rf mystem-3.1-linux-64bit.tar.gz \
 && pip install --upgrade pip

COPY requirements.txt .
COPY solution.py .

RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "import nltk; nltk.download('stopwords')"
 && rm -rf /tmp/*
CMD ["python", "solution.py"]