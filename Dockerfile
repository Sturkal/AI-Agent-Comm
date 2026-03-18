FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GUNICORN_CMD_ARGS="--access-logfile - --error-logfile -"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY app ./app
COPY data ./data
COPY tests ./tests

EXPOSE 5000

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app.main:app"]
