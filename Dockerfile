FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml /app/
COPY coverage_pipeline /app/coverage_pipeline
COPY etl_ingest.py etl_validate.py etl_transform.py etl_snapshot.py train_model.py approve_model.py /app/

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "coverage_pipeline.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
