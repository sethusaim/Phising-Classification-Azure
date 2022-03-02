import json
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from phising.model.load_production_model import Load_Prod_Model
from phising.model.prediction_from_model import Prediction
from phising.model.training_model import Train_Model
from phising.validation_insertion.prediction_validation_insertion import Pred_Validation
from phising.validation_insertion.train_validation_insertion import Train_Validation
from utils.create_containers import Azure_Container
from utils.read_params import read_params

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = FastAPI()

config = read_params()

templates = Jinja2Templates(directory=config["templates"]["dir"])

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        config["templates"]["index"], {"request": request}
    )

@app.get("/create")
async def create_container():
    try:
        container = Azure_Container()

        container.generate_containers()

        return Response("Created Azure containers !!")

    except Exception as e:
        return Response("Error Occurred! %s" % e)


@app.get("/train")
async def trainRouteClient():
    try:
        raw_data_train_container_name = config["container"]["phising_raw_data"]

        train_val_obj = Train_Validation(container_name=raw_data_train_container_name)

        train_val_obj.training_validation()

        train_model_obj = Train_Model()

        num_clusters = train_model_obj.training_model()

        load_prod_model_obj = Load_Prod_Model(num_clusters=num_clusters)

        load_prod_model_obj.load_production_model()

    except Exception as e:
        return Response("Error Occurred! %s" % e)

    return Response("Training successfull!!")


@app.get("/predict")
async def predictRouteClient():
    try:
        raw_data_pred_container_name = config["container"]["phising_raw_data_container"]

        pred_val = Pred_Validation(raw_data_pred_container_name)

        pred_val.prediction_validation()

        pred = Prediction()

        container, filename, json_predictions = pred.predict_from_model()

        return Response(
            f"prediction file created in {container} container with filename as {filename}, and few of the predictions are {str(json.loads(json_predictions))}"
        )

    except Exception as e:
        return Response("Error Occurred! %s" % e)


if __name__ == "__main__":
    host = config["app"]["host"]

    port = config["app"]["port"]

    uvicorn.run(app, host=host, port=port)
