from flask import Blueprint
from flask import Flask, request
import json
from engine import MotorClasificador
import logging

main = Blueprint('main', __name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@main.route("/user/<string:user_id>", methods=["GET"])
def user_by_id(user_id):
    logger.debug("Obteniendo información de usuario %s", int(user_id))
    user = motor_clasificador.get_user_by_id(int(user_id))
    return json.dumps(user)


def create_app(spark_context, dataset_path):
    global motor_clasificador

    motor_clasificador = MotorClasificador(spark_context, dataset_path)

    app = Flask(__name__)
    app.register_blueprint(main)
    return app
