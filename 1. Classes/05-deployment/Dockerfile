FROM continuumio/miniconda3

WORKDIR /app

# Create a Conda environment using environment.yml
COPY environment.yml /app/
RUN conda env create -f environment.yml

# Copy the model files
COPY ["model_C=1.0.bin", "predict.py", "./"]

# The code to run when container is started:
COPY entrypoint.sh ./
RUN chmod +x ./entrypoint.sh
EXPOSE 9696
ENTRYPOINT ["./entrypoint.sh"]