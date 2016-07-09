import logging
import os
import tools

import pymongo
import ConfigParser

configParser = ConfigParser.RawConfigParser()
configParser.read("config.ini")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# logging.basicConfig(filename="logs/engine.log", format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class MotorClasificador:
    """Motor del clasificador de cuentas
    """

    def __init__(self):
        """Inicializa el SparkContext
        """

        self.sc = None
        self.juez_timelines = None
        self.modelo_spam = None
        self.mongodb_host = None
        self.mongodb_port = None
        self.mongodb_db = None
        self.hive_context = None
        logger.info("Calentando motores...")

    def inicializar_contexto(self, app_name, py_files):
        self.sc = tools.iniciar_spark_context(app_name, py_files)
        self.hive_context = tools.hive_context(self.sc)
        self.mongodb_host = "mongodb://" + configParser.get("database", "host")
        self.mongodb_port = configParser.get("database", "port")
        self.mongodb_db = configParser.get("database", "db")
        client = pymongo.MongoClient(self.mongodb_host + ":" + self.mongodb_port)
        db = client[self.mongodb_db]
        coleccion = db["caracteristicas"]
        coleccion.ensure_index("createdAt", expireAfterSeconds=configParser.get("database", "ttl"))
        client.close()
        return True

    def entrenar_spam(self, dir_spam, dir_no_spam):
        sc = self.sc
        hive_context = self.hive_context
        modelo = tools.entrenar_spam(sc, hive_context, dir_spam, dir_no_spam)
        self.modelo_spam = modelo

        return True

    def entrenar_juez(self, directorio):
        sc = self.sc
        juez_spam = self.modelo_spam
        hive_context = self.hive_context

        logger.info("Entrenando juez")

        juez_timelines = tools.entrenar_juez(sc, hive_context, juez_spam, directorio)

        self.juez_timelines = juez_timelines

        logger.info("Finalizando...")

        return True

    """def cargar_spam(self, directorio):
        self.modelo_spam = RandomForestModel.load(self.sc, directorio)

        return True

    def guardar_juez(self, directorio):
        sc = self.sc
        modelo = self.juez_timelines
        if self.juez_timelines:
            modelo.save(sc, directorio)
            return True
        else:
            logger.error("NO SE HA ENTRENADO NINGUN JUEZ")
            return False

    def cargar_juez(self, directorio):
        self.juez_timelines = RandomForestModel.load(self.sc, directorio)
        return True"""

    def evaluar(self, dir_timeline):
        sc = self.sc
        juez_timeline = self.juez_timelines
        juez_spam = self.modelo_spam
        mongo_uri = self.mongodb_host + ":" + self.mongodb_port + "/" + self.mongodb_db
        hive_context = self.hive_context
        resultado = tools.evaluar(sc, hive_context, juez_spam, juez_timeline, dir_timeline, mongo_uri)
        return resultado

    # Deprecado, funcion inmersa en inicializar_contexto
    def inicializar_mongo(self, mongodb_host, mongodb_port, mongodb_db, mongodb_ttl):
        self.mongodb_host = mongodb_host
        self.mongodb_port = mongodb_port
        self.mongodb_db = mongodb_db
        client = pymongo.MongoClient(mongodb_host + ":" + mongodb_port)
        db = client[mongodb_db]
        coleccion = db["caracteristicas"]
        coleccion.ensure_index("createdAt", expireAfterSeconds=mongodb_ttl)
        client.close()
        return True
