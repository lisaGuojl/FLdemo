# Federated learning secure aggregation

A simple Python implementation of a secure aggregation protocole for federated learning.

## Introduction

This project is an implementation of a practical secure aggregation for privacy-preserving Machine Learning as it is described in this [paper](https://dl.acm.org/doi/pdf/10.1145/3133956.3133982?download=true).

It will train a neural network using federated learning. The neural network is created in order to predict human activity based on accelerometer and gyroscope data. The field of recognize human activity is beyond the scope of this work.

It is devided into two parts:

- Server: which will centralize information and orchestrate multiple clients to process new weights and biais of a shared machine learning neural network.
- Client: which will train the shared neural network with a part of the training dataset

Please read the following to understand how to run the project.

## Run the project

### Server
Install the docker engine and docker compose. You can follow the instructions in https://docs.docker.com/engine/install/. 

The server project uses `docker-compose` to run severals services (containers):

- `web`: this service is a web API coded with python and Flask. This service is accessible using `http://localhost:3000`
- `db`: this service is based on the official `postgres` docker image to run a database. This service is accessible using `postgresql://localhost:5432` with username: `postgres` and password: `root`. The default database schema is `federated`. 
- `pgadmin`: this service is a web application to navigate throw the `postgres` database. This service is accessible using `http://localhost:5050`

To run the complete stack, use the following command:
```bash
docker compose up
```

If you want to export the docker image of this API, use the following command:
```bash
docker build . -t <IMAGE_TAG>:<VERSION>
```

### Client

Install the requirements.
```bash
pip install -r requirements.txt
```
Run one round of FL:
```bash
python app/docker-client.py
```

## Python lib used

- Cryptography: https://cryptography.io/en/latest/
- SSlib: https://github.com/jqueiroz/python-sslib

## Références

https://dl.acm.org/doi/pdf/10.1145/3133956.3133982

http://doi.acm.org/10.1145/3302505.3310068
