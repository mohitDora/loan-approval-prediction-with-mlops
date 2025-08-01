FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
COPY setup.py /app/
COPY pyproject.toml /app/

COPY . /app/

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
