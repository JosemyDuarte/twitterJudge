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
        """Inicializa el SparkContext, SqlContext y MongoDB
        """
        logger.info("Calentando motores...")
        self.sc = tools.iniciar_spark_context(app_name=configParser.get("spark", "name"))
        self.juez_timelines = None
        self.modelo_spam = None
        self.mongodb_host = "mongodb://" + configParser.get("database", "host")
        self.mongodb_port = configParser.get("database", "port")
        self.mongodb_db = configParser.get("database", "db")
        self.mongodb_collection = configParser.get("database", "collection")
        self.mongodb_collection_trainingset = configParser.get("database", "collection_training")
        self.hive_context = tools.hive_context(self.sc)
        client = pymongo.MongoClient(self.mongodb_host + ":" + self.mongodb_port)
        db = client[self.mongodb_db]
        coleccion = db[self.mongodb_collection]
        coleccion.ensure_index("createdAt", expireAfterSeconds=int(configParser.get("database", "ttl")))
        client.close()

    def entrenar_spam(self, dir_spam, dir_no_spam, num_trees, max_depth):
        """
            Entrena el juez que clasifica los tweets spam
            Parameters
            ----------
            dir_spam : str
                Direccion en la que se encuentra el archivo de entrenamiento para SPAM
            dir_no_spam : str
                Direccion en la que se encuentra el archivo de entrenamiento para NoSPAM
            num_trees: int
                Numero de arboles a utilizar para entrenar el Random Forest
            max_depth: int
                Maxima profundidad utilizada para el bosque del Random Forest
            Returns
            -------
            True : True
                En caso de ejecucion sin problemas, True sera retornado.
            Examples
            --------
            > entrenar_spam("/archivo/spam","/archivo/nospam",3,4)
            > entrenar_spam("hdfs://[host]:[port]/archivo/spam","hdfs://[host]:[port]/archivo/nospam")
            """
        sc = self.sc
        hive_context = self.hive_context
        modelo = tools.entrenar_spam(sc, hive_context, dir_spam, dir_no_spam, num_trees, max_depth)
        self.modelo_spam = modelo

        return True

    def entrenar_juez(self, humanos, ciborgs, bots, num_trees, max_depth):
        """
            Entrena el juez que clasifica los tweets spam
            Parameters
            ----------
            humanos : str
                Direccion del directorio con timelines "humanos"
            ciborgs : str
                Direccion del directorio con timelines "ciborgs"
            bots : str
                Direccion del directorio con timelines "bots"
            num_trees : int
                Numero de arboles a utilizar para entrenar el Random Forest
            max_depth : int
                Maxima profundidad utilizada para el bosque del Random Forest
            Returns
            -------
            True : True
                En caso de ejecucion sin problemas, True sera retornado.
            Examples
            --------
            > entrenar_juez("/carpeta/humanos", "/carpeta/ciborgs", "/carpeta/bots", 2, 4)
            """
        sc = self.sc
        juez_spam = self.modelo_spam
        hive_context = self.hive_context

        logger.info("Entrenando juez...")

        mongo_uri = (self.mongodb_host + ":" + self.mongodb_port + "/" + self.mongodb_db + "." +
                     self.mongodb_collection_trainingset)

        juez_timelines = tools.entrenar_juez(sc, hive_context, juez_spam, mongo_uri, humanos, ciborgs, bots, num_trees,
                                             max_depth)

        self.juez_timelines = juez_timelines

        logger.info("Finalizando...")

        return True

    def evaluar(self, dir_timeline):
        """
            Evalua y clasifica los timelines
            Parameters
            ----------
            dir_timeline : str
                Direccion en la que se encuentran los timelines a clasificar
            Returns
            -------
            Resultado : [int, ] list
                Retorna los IDs de los usuarios evaluados
            Examples
            --------
            > evaluar('{"directorio":"/carpeta/con/timelines/*"}')
            """
        sc = self.sc
        juez_timeline = self.juez_timelines
        juez_spam = self.modelo_spam
        mongo_uri = self.mongodb_host + ":" + self.mongodb_port + "/" + self.mongodb_db + "." + self.mongodb_collection
        hive_context = self.hive_context
        resultado = tools.evaluar(sc, hive_context, juez_spam, juez_timeline, dir_timeline, mongo_uri)
        return resultado
