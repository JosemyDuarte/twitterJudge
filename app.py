from flask import Blueprint
from flask import Flask, request
import json
from engine import MotorClasificador
import logging
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

main = Blueprint('main', __name__)

logging.basicConfig(filename="logs/engine.log",format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

@main.route("/carga_inicial/", methods=["POST"])
def carga_inicial():
    directorio = request.json.get("directorio")
    logger.debug("Iniciando carga inicial sobre la carpeta: %s", directorio)
    resultado = motor_clasificador.carga_inicial(directorio)
    return json.dumps(dict(resultado=resultado))


@main.route("/cargar_modelo/", methods=["POST"])
# TODO: Cargar modelo para el arbol (requiere de los 6000)
def cargar_modelo():
    """Realiza la carga del set de entrenamiento y genera el modelo.
    Requiere de la especificacion de los directorios para las 3 categorias.
    EJEMPLO: {"bot":"/carpeta/con/bots","humano":"/carpeta/con/humano/","ciborg":"/carpeta/con/ciborg/"}
    """
    logger.debug("Iniciando carga inicial...")
    directorio = request.json
    logging.info(directorio)
    if not directorio["bot"]:
        logging.info("No se especifico la direccion de la carpeta para los bots")
        return json.dumps(dict(exito=False))
    if not directorio["humano"]:
        logging.info("No se especifico la direccion de la carpeta para los humanos")
        return json.dumps(dict(exito=False))
    if not directorio["ciborg"]:
        logging.info("No se especifico la direccion de la carpeta para los ciborgs")
        return json.dumps(dict(exito=False))

    logger.debug("Ejecutando carga y entrenamiento")
    resultado = motor_clasificador.cargar_modelo(directorio)
    logger.debug("Finalizando carga y entrenamiento")
    return json.dumps(dict(exito=resultado))


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
