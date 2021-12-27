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

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = FastAPI()


class FLclient_status(BaseModel):
    FLCLstart: bool = False
    FLCFail: bool = False


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


# Define Flower client
class MnistClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        return model.get_weights()  #

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}


model = build_model()


# Load CIFAR-10 dataset


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
    global model
    try:
        model = keras.models.load_model('/model/model.h5')
        pass
    except Exception as e:
        print('[E] learning', e)
        status.FLCFail = True
        await notify_fail()
        status.FLCFail = False
    await flower_client_start()


async def flower_client_start():
    print('learning')
    global status
    global model
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    try:
        fl.client.start_numpy_client(server_address="10.152.183.181:8080",
                                     client=MnistClient(model, x_train, y_train, x_test, y_test))
        await model_save()
    except Exception as e:
        print('[E] learning', e)
        status.FLCFail = True
        await notify_fail()
        status.FLCFail = False


async def model_save():
    print('model_save')
    global model
    try:
        model.save('/model/model.h5')
        await notify_fin()
    except Exception as e:
        print('[E] learning', e)
        status.FLCFail = True
        await notify_fail()
        status.FLCFail = False


async def notify_fin():
    global status
    status.FLCLstart = False
    loop = asyncio.get_event_loop()
    future2 = loop.run_in_executor(None, requests.get, 'http://localhost:8080/trainFin')
    r = await future2
    print('try notify_fin')
    if r.status_code == 200:
        print('trainFin')
    else:
        print(r.content)


async def notify_fail():
    global status
    status.FLCLstart = False
    loop = asyncio.get_event_loop()
    future1 = loop.run_in_executor(None, requests.get, 'http://localhost:8080/trainFail')
    r = await future1
    print('try notify_fail')
    if r.status_code == 200:
        print('trainFin')
    else:
        print(r.content)


# Start Flower client
# s3에 model 없고 환경변수를 탐색하여 ENV가 init이라면 s3에 초기 가중치를 업로드 한다.
from botocore.exceptions import ClientError


def S3_check(s3_client, bucket, key):  # 없으면 참
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
        ACCESS_KEY_ID = os.environ.get('ACCESS_KEY_ID')
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

#
