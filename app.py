from flask import Blueprint
from flask import Flask, request
import json
from engine import MotorClasificador
import logging
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

main = Blueprint('main', __name__)

logging.basicConfig(filename="logs/app.log", level=logging.INFO)
logger = logging.getLogger(__name__)


@main.route("/carga_inicial/", methods=["POST"])
def carga_inicial():
    """Realiza la carga inicial y el entrenamiento del modelo.
    Requiere de la especificacion de los directorios para las 3 categorias.
    EJEMPLO: {"bot":"/carpeta/con/bots","humano":"/carpeta/con/humano/","ciborg":"/carpeta/con/ciborg/"}
    """
    logger.debug("Iniciando carga inicial...")
    contenido = request.json
    logging.info(contenido)
    if not contenido["bot"]:
        logging.info("No se especifico la direccion de la carpeta para los bots")
        return json.dumps(dict(exito=False))
    if not contenido["humano"]:
        logging.info("No se especifico la direccion de la carpeta para los humanos")
        return json.dumps(dict(exito=False))
    if not contenido["ciborg"]:
        logging.info("No se especifico la direccion de la carpeta para los ciborgs")
        return json.dumps(dict(exito=False))

    logger.debug("Ejecutando carga inicial")
    motor_clasificador.modelo = motor_clasificador.carga_inicial(contenido)
    logger.debug("Finalizando carga inicial")
    return json.dumps(dict(exito=True))


@main.route("/tweets", methods=["GET"])
def cantidad_tweets():
    logger.debug("Obteniendo cantidad de tweets en base de conocimiento")
    n = motor_clasificador.cantidad_tweets()
    return json.dumps({"cantidad_tweets": n})


def create_app(spark_context):
    global motor_clasificador

    motor_clasificador = MotorClasificador(spark_context)

    app = Flask(__name__)
    app.register_blueprint(main)
    return app
