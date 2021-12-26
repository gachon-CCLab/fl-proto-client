import flwr as fl
import tensorflow as tf
from tensorflow import keras

import os
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import requests
from fastapi import FastAPI, BackgroundTasks
import asyncio
import uvicorn
from pydantic.main import BaseModel
import logging
import json
import boto3

app = FastAPI()


class FLclient_status(BaseModel):
    FLCLstart: bool = False


status = FLclient_status()


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# Load and compile Keras model
# model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Load CIFAR-10 dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Define Flower client
# class CifarClient(fl.client.NumPyClient):
#    def get_parameters(self):
#        return model.get_weights()#

#    def fit(self, parameters, config):
#        model.set_weights(parameters)
#        model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
#        return model.get_weights(), len(x_train), {}

#    def evaluate(self, parameters, config):
#        model.set_weights(parameters)
#        loss, accuracy = model.evaluate(x_test, y_test)
#        return loss, len(x_test), {"accuracy": accuracy}
@app.on_event("startup")
def startup():
    pass
    # loop = asyncio.get_event_loop()
    # loop.set_debug(True)
    # loop.create_task(run_client())


@app.get("/start")
async def flclientstart(background_tasks: BackgroundTasks):
    global status
    print('start')
    status.FLCLstart = True
    background_tasks.add_task(run_client)
    return status


async def run_client():
    global status
    await flower_client_start()


async def flower_client_start():
    print('learning')
    await asyncio.sleep(10)
    await model_save()


async def model_save():
    print('model_save')
    await asyncio.sleep(10)
    await notify_fin()


async def notify_fin():
    global status
    status.FLCLstart = False
    while True:
        print('try')
        r = requests.get('http://localhost:8080/trainFin')
        print('try')
        if r.status_code == 200:
            print('trainFin')
            break
        else:
            print(r.content)
        await asyncio.sleep(5)


# Start Flower client
# s3에 model 없고 환경변수를 탐색하여 ENV가 init이라면 s3에 초기 가중치를 업로드 한다.
from botocore.exceptions import ClientError


def S3_check(s3_client, bucket, key):# 없으면 참
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        return int(e.response['Error']['Code']) != 404
    return True


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    print(tf.version.VERSION)

    if os.environ.get('ENV', 'development') == 'init':
        res = requests.get('http://10.152.183.186:8000' + '/FLSe/info')  # 서버측 manager
        S3_info = res.json()['Server_Status']
        model = build_model()
        model.save(S3_info['S3_key'])
        ##########서버에 secret피일 이미 있음 #################
        ACCESS_KEY_ID =  os.environ.get('ACCESS_KEY_ID')
        ACCESS_SECRET_KEY = os.environ.get('ACCESS_SECRET_KEY')
        BUCKET_NAME = os.environ.get('BUCKET_NAME')
        s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID,
                                 aws_secret_access_key=ACCESS_SECRET_KEY)
        if S3_check(s3_client, BUCKET_NAME, S3_info['S3_key']):

            response = s3_client.upload_file(
                S3_info['S3_key'], BUCKET_NAME, S3_info['S3_key'])
        else:
            print('이미 모델 있음')
    else:
        uvicorn.run("app:app", host='0.0.0.0', port=8002, reload=True)

# fl.client.start_numpy_client(server_address="10.152.183.181:8080", client=CifarClient())
