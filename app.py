from flask import Blueprint
from flask import Flask, request
import json
from engine import MotorClasificador
import logging
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

main = Blueprint('main', __name__)

logging.basicConfig(filename="logs/engine.log", format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


@main.route("/cargar_juez/", methods=["POST"])
def cargar_juez():
    directorio = request.json.get("directorio")
    logger.debug("Cargando juez almacenado en: %s", directorio)
    resultado = motor_clasificador.cargar_juez(directorio)
    return json.dumps(dict(resultado=resultado))

@main.route("/entrenar_juez/", methods=["POST"])
def entrenar_juez():
    """Realiza la carga del set de entrenamiento y genera el modelo.
    Requiere de la especificacion de los directorios para las 3 categorias.
    EJEMPLO: {"bot":"/carpeta/con/bots","humano":"/carpeta/con/humano/","ciborg":"/carpeta/con/ciborg/"}
    """
    logger.debug("Iniciando carga inicial...")
    directorio = request.json
    logging.info(directorio)
    if not directorio["bots"]:
        logging.info("No se especifico la direccion de la carpeta para los bots")
        return json.dumps(dict(resultado=False))
    if not directorio["humanos"]:
        logging.info("No se especifico la direccion de la carpeta para los humanos")
        return json.dumps(dict(resultado=False))
    if not directorio["ciborgs"]:
        logging.info("No se especifico la direccion de la carpeta para los ciborgs")
        return json.dumps(dict(resultado=False))

    logger.debug("Ejecutando carga y entrenamiento")
    resultado = motor_clasificador.entrenar_juez(directorio)
    logger.debug("Finalizando carga y entrenamiento")
    return json.dumps(dict(resultado=resultado))

@main.route("/entrenar_spam/", methods=["POST"])
def entrenar_spam():
    """Realiza la carga del set de entrenamiento y genera el modelo.
    Requiere de la especificacion de los directorios para las 3 categorias.
    EJEMPLO: {"spam":"/archivo/spam","no_spam":"/archivo/no_spam/"}
    """
    logger.debug("Iniciando carga...")
    directorio = request.json
    logging.info(directorio)
    if not directorio["spam"]:
        logging.error("No se especifico la direccion del archivo de SPAM")
        return json.dumps(dict(resultado=False))
    if not directorio["no_spam"]:
        logging.error("No se especifico la direccion del archivo de NOSPAM")
        return json.dumps(dict(resultado=False))
    logger.debug("Ejecutando carga y entrenamiento")
    resultado = motor_clasificador.entrenar_spam(directorio["spam"], directorio["no_spam"])
    logger.debug("Finalizando carga y entrenamiento")
    return json.dumps(dict(resultado=resultado))

@main.route("/cargar_spam/", methods=["POST"])
def cargar_spam():
    directorio = request.json.get("directorio")
    logger.debug("Cargando juez de spam almacenado en: %s", directorio)
    resultado = motor_clasificador.cargar_spam(directorio)
    return json.dumps(dict(resultado=resultado))


@main.route("/mongo_uri/", methods=["POST"])
def mongo_uri():
    mongodb_host = request.json.get("mongodb_host")
    mongodb_port = request.json.get("mongodb_port")
    mongodb_db = request.json.get("mongodb_db")
    logger.debug("mongo_uri recibio host: %s", mongodb_host)
    logger.debug("mongo_uri recibio puerto: %s", mongodb_port)
    logger.debug("mongo_uri recibio bd: %s", mongodb_db)
    resultado = motor_clasificador.inicializar_mongo(mongodb_host, mongodb_port, mongodb_db)
    return json.dumps(dict(resultado=resultado))

@main.route("/evaluar/", methods=["POST"])
def evaluar():
    directorio = request.json.get("directorio")
    logger.info("Iniciando evaluacion sobre: %s", directorio)
    resultado = motor_clasificador.evaluar(directorio)
    return json.dumps(dict(resultado=resultado))

@main.route("/alive/", methods=["GET"])
def alive():
    return json.dumps(dict(resultado="I'm Alive!"))

def create_app(spark_context):
    global motor_clasificador

    motor_clasificador = MotorClasificador(spark_context)

    app = Flask(__name__)
    app.register_blueprint(main)
    return app
