# Start with the official Apache AGE image for Postgres 16
FROM apache/age:release_PG16_1.6.0

# Switch to root to install dependencies and compile extensions
USER root

# Install build tools, partman, and cron
RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-server-dev-16 \
    postgresql-16-partman \
    postgresql-16-cron \
    git \
    && rm -rf /var/lib/apt/lists/*

# Compile and install pgvector (v0.6.0+ required for advanced HNSW)
RUN git clone --branch v0.6.0 https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector

# Extend shared_preload_libraries to include pg_cron alongside AGE (from base image)
RUN sed -i "s/^shared_preload_libraries = '\(.*\)'/shared_preload_libraries = '\1,pg_cron'/" \
    /usr/share/postgresql/16/postgresql.conf.sample || \
    echo "shared_preload_libraries = 'age,pg_cron'" >> /usr/share/postgresql/16/postgresql.conf.sample \
    && echo "cron.database_name = 'open_brain'" >> /usr/share/postgresql/16/postgresql.conf.sample

# Revert to the postgres user expected by the base image
USER postgres