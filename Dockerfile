FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir ".[anthropic]"

ENV MEMRY_DB_PATH=/data/memry.db
VOLUME /data
EXPOSE 8787

CMD ["memry", "serve", "--host", "0.0.0.0", "--port", "8787"]
