import json
import logging
import os

from flask import Blueprint
from flask import Flask, request

import engine

os.chdir(os.path.dirname(os.path.abspath(__file__)))

main = Blueprint('main', __name__)

logging.basicConfig(filename="logs/engine.log", format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


@main.route("/entrenar_juez/", methods=["POST"])
def entrenar_juez():
    """
    Realiza la carga del set de entrenamiento y genera el juez.
    Requiere de la especificacion de los directorios para las 3 categorias.
    Returns
    -------
    resultado : Double
        Double que representa el resultado de evaluar la exactitud del juez y matriz de confusion
    Examples
    --------
    > curl -H "Content-Type: application/json" -X POST -d
        '{"bots":"/carpeta/con/bots","humanos":"/carpeta/con/humanos","ciborgs":"/carpeta/con/ciborg", "dir_juez": "./jueces/test1"}'
         http://[host]:[port]/entrenar_juez/
    """
    logger.debug("Iniciando carga inicial...")
    data = request.json
    logging.info(data)
    if "bots" not in data:
        logging.error("No se especifico la direccion de la carpeta para los bots")
        return json.dumps(dict(resultado=False))
    if "humanos" not in data:
        logging.error("No se especifico la direccion de la carpeta para los humanos")
        return json.dumps(dict(resultado=False))
    if "ciborgs" not in data:
        logging.error("No se especifico la direccion de la carpeta para los ciborgs")
        return json.dumps(dict(resultado=False))
    if "dir_juez" not in data:
        logging.error("No se especifico la direccion de la carpeta para guardar el juez entrenado")
        return json.dumps(dict(resultado=False))
    if "num_trees" not in data:
        logging.warn("No se especifico numero de arboles, se utilizaran 3 por defecto")
    if "max_depth" not in data:
        logging.warn("No se especifico profundidad del bosque, se utilizara 2 por defecto")
    logger.debug("Ejecutando carga y entrenamiento")
    accuracy, matrix = motor_clasificador.entrenar_juez(data.get("humanos"), data.get("ciborgs"), data.get("bots"),
                                                        data.get("dir_juez"), data.get("num_trees", 30),
                                                        data.get("max_depth", 8))
    logger.debug("Finalizando carga y entrenamiento")
    return json.dumps(dict(accuracy=accuracy, matrix=matrix))


@main.route("/entrenar_spam/", methods=["POST"])
def entrenar_spam():
    """Realiza la carga del set de entrenamiento y genera el juez.
    Requiere de la especificacion de los directorios para las 2 categorias SPAM y NoSPAM.
    Returns
    -------
    resultado : Double
        Doueble que sera la exactitud del juez en caso de ejecutarse exitosamente
    Examples
    --------
    > curl -H "Content-Type: application/json" -X POST -d
    '{"spam":"/archivo/spam","no_spam":"/archivo/no_spam"}'
    http://[host]:[port]/entrenar_spam/
    > curl -H "Content-Type: application/json" -X POST -d
    '{"spam":"/archivo/spam","no_spam":"/archivo/no_spam"}, "num_trees":3, "max_depth":2'
    http://[host]:[port]/entrenar_spam/
    """
    logger.debug("Iniciando carga...")
    data = request.json
    logging.info(data)
    if "spam" not in data:
        logging.error("No se especifico la direccion del archivo de SPAM")
        return json.dumps(dict(resultado=False))
    if "no_spam" not in data:
        logging.error("No se especifico la direccion del archivo de NOSPAM")
        return json.dumps(dict(resultado=False))
    if "num_trees" not in data:
        logging.warn("No se especifico numero de arboles, se utilizaran 3 por defecto")
    if "max_depth" not in data:
        logging.warn("No se especifico profundidad del bosque, se utilizara 2 por defecto")
    logger.debug("Ejecutando carga y entrenamiento")
    resultado = motor_clasificador.entrenar_spam(data["spam"], data["no_spam"], data.get("num_trees", 30),
                                                 data.get("max_depth", 8))
    logger.debug("Finalizando carga y entrenamiento")
    return json.dumps(dict(resultado=resultado))


@main.route("/evaluar/", methods=["POST"])
def evaluar():
    """
    Realiza la evaluacion de los timelines.
    Requiere de la especificacion del directorio que contiene los timelines
    Returns
    -------
    resultado : diccionario
        Sera False, en caso de error. Contendra el id de los usuarios evaluados.
    Examples
    --------
    > curl -H "Content-Type: application/json" -X POST -d
    '{"directorio":"/carpeta/con/timelines/*"}'
    http://[host]:[port]/evaluar/
    """
    if not request.json.get("directorio"):
        logging.error("No se especifico el parametro 'directorio' para evaluar")
        return json.dumps(dict(resultado=False))
    directorio = request.json.get("directorio")
    logger.info("Iniciando evaluacion sobre: %s", directorio)
    resultado = motor_clasificador.evaluar(directorio)
    return json.dumps(dict(resultado=resultado))


@main.route("/evaluar_online/", methods=["POST"])
def evaluar_online():
    """
    Realiza la evaluacion de un timeline.
    Requiere de la especificacion del timeline de un usuario especifico
    Returns
    -------
    resultado : diccionario
        Sera False, en caso de error. Contendra el id del usuario evaluado.
    Examples
    --------
    > curl -H "Content-Type: application/json" -X POST -d
    '{"timeline":""}'
    http://[host]:[port]/evaluar/
    """
    if not request.json.get("timeline"):
        logging.error("No se especifico el parametro 'timeline' para evaluar")
        return json.dumps(dict(resultado=False))
    timeline = request.json.get("timeline")
    logger.info("Iniciando evaluacion sobre: %s", timeline)
    resultado = motor_clasificador.evaluar_online(timeline)
    return json.dumps(dict(resultado=resultado))


@main.route("/features_importance/", methods=["GET"])
def features_importances_juez():
    return json.dumps(dict(resultado=motor_clasificador.features_importances_juez()))


@main.route("/guardar_juez/", methods=["POST"])
def guardar_juez():
    """
    Alamcena el juez entrenado
    Returns
    -------
    resultado : boolean
        Sera False, en caso de error. True en ejecucion exitosa
    Examples
    --------
    > curl -H "Content-Type: application/json" -X POST -d
    '{"tipo_juez":0, "path":"/carpeta/juez_spam"}'
    http://[host]:[port]/guardar_juez/
    """
    logger.debug("Guardando juez...")
    data = request.json
    logging.info(data)
    if "tipo_juez" not in data:
        logging.error("No se especifico el tipo de juez a almacenar")
        return json.dumps(dict(resultado=False))
    if "path" not in data:
        logging.error("No se especifico el directorio a utilizar")
        return json.dumps(dict(resultado=False))
    tipo_juez = data.get("tipo_juez")
    path = data.get("path")
    return json.dumps(dict(resultado=motor_clasificador.guardar_juez(tipo_juez, path)))


@main.route("/cargar_juez/", methods=["POST"])
def cargar_juez():
    """
    Carga un juez previamente almacenado
    Returns
    -------
    resultado : boolean
        Sera False, en caso de error. True para ejecucion exitosa
    Examples
    --------
    > curl -H "Content-Type: application/json" -X POST -d
    '{"tipo_juez":0, "path":"/carpeta/juez_spam"}'
    http://[host]:[port]/guardar_juez/
    """
    logger.debug("Cargando juez...")
    data = request.json
    logging.info(data)
    if "tipo_juez" not in data:
        logging.error("No se especifico el tipo de juez a almacenar")
        return json.dumps(dict(resultado=False))
    if "path" not in data:
        logging.error("No se especifico el directorio a utilizar")
        return json.dumps(dict(resultado=False))
    tipo_juez = request.json.get("tipo_juez")
    path = request.json.get("path")
    return json.dumps(dict(resultado=motor_clasificador.cargar_juez(tipo_juez, path)))


@main.route("/alive/", methods=["GET"])
def alive():
    """Funcion para verificar disponibilidad del servidor"""
    return json.dumps(dict(resultado="I'm Alive!"))


def create_app():
    global motor_clasificador
    motor_clasificador = engine.MotorClasificador()
    app = Flask(__name__)
    app.register_blueprint(main)
    return app
