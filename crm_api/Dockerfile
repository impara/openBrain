FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY crm_api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY crm_api /app/crm_api

EXPOSE 8000

CMD ["uvicorn", "crm_api.main:app", "--host", "0.0.0.0", "--port", "8000"]

