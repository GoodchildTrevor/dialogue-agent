FROM python:3.13-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY tests ./tests
COPY pytest.ini .
COPY alembic ./alembic
COPY alembic.ini .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000
CMD ["./entrypoint.sh"]
