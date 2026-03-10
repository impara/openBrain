# Start with the official Apache AGE image for Postgres 16
FROM apache/age:release_PG16_1.6.0

# Switch to root to install dependencies and compile extensions
USER root

# Install build tools needed to compile pgvector
RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-server-dev-16=16.13-1.pgdg13+1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Compile and install pgvector (v0.6.0+ required for advanced HNSW)
RUN git clone --branch v0.6.0 https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector

# Ship initialization SQL in the image so deployments do not depend on a host bind mount.
COPY init.sql /docker-entrypoint-initdb.d/init.sql

# Revert to the postgres user expected by the base image
USER postgres
