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
        logger.info("Calentando motores...")
        self.sc = tools.iniciar_spark_context(app_name=configParser.get("spark", "name"))
        self.juez_timelines = None
        self.modelo_spam = None
        self.mongodb_host = "mongodb://" + configParser.get("database", "host")
        self.mongodb_port = configParser.get("database", "port")
        self.mongodb_db = configParser.get("database", "db")
        self.mongodb_collection = configParser.get("database", "collection")
        self.hive_context = tools.hive_context(self.sc)
        client = pymongo.MongoClient(self.mongodb_host + ":" + self.mongodb_port)
        db = client[self.mongodb_db]
        coleccion = db[self.mongodb_collection]
        coleccion.ensure_index("createdAt", expireAfterSeconds=int(configParser.get("database", "ttl")))
        client.close()

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

    def evaluar(self, dir_timeline):
        sc = self.sc
        juez_timeline = self.juez_timelines
        juez_spam = self.modelo_spam
        mongo_uri = self.mongodb_host + ":" + self.mongodb_port + "/" + self.mongodb_db + "." + self.mongodb_collection
        hive_context = self.hive_context
        resultado = tools.evaluar(sc, hive_context, juez_spam, juez_timeline, dir_timeline, mongo_uri)
        return resultado

