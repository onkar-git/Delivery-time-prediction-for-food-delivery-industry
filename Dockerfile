# set the base image
FROM python:3.12-slim

# install lightgbm dependency
RUN apt-get update && apt-get install -y libgomp1

# set up the working directory
WORKDIR /app

# copy the requirements file
COPY requirements-docker.txt ./

# install the packages
RUN pip install -r requirements-docker.txt

# copy the app contents
COPY app.py ./
COPY ./artifacts/model_trainer/models/model.joblib ./artifacts/model_trainer/models/model.joblib
COPY ./artifacts/data_trans/models/preprocessor.joblib ./artifacts/data_trans/models/preprocessor.joblib 
COPY ./src/Deliveryprediction/utils/pred_data_clean.py ./src/Deliveryprediction/utils/pred_data_clean.py
COPY ./artifacts/model_evaluation/metrics.json ./artifacts/model_evaluation/metrics.json

# expose the port
EXPOSE 8000

# Run the file using command
CMD [ "python","./app.py" ]