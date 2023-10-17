FROM svizor/zoomcamp-model:3.10.12-slim

WORKDIR /app

# Copy environment files
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies
RUN pip install pipenv && pipenv install --system --deploy

# Copy model files
COPY ["predict_flask.py", "./"]

# Execute gunicorn in the container
EXPOSE 9696
CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict_flask:app"]
