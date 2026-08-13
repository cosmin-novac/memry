FROM python:3.12-slim AS package-builder

WORKDIR /app

# This build tool is cached independently and never enters the runtime image.
RUN pip install --no-cache-dir hatchling
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-cache-dir --no-deps --no-build-isolation --wheel-dir /wheels .

FROM python:3.12-slim AS runtime

WORKDIR /app

# Dependencies have their own cache key. Source, docs and version-only metadata
# changes therefore do not reinstall NumPy, MCP, Anthropic, Starlette, etc.
# tests/test_container_build.py keeps this manifest synchronized with pyproject.toml.
COPY requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY --from=package-builder /wheels /wheels
RUN pip install --no-cache-dir --no-deps /wheels/*.whl && rm -rf /wheels

ENV MEMRY_DB_PATH=/data/memry.db
VOLUME /data
EXPOSE 8787

CMD ["memry", "serve", "--host", "0.0.0.0", "--port", "8787"]
