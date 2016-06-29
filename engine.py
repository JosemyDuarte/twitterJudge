import os
import logging, tools
from pyspark.mllib.tree import RandomForest, RandomForestModel

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# logging.basicConfig(filename="logs/engine.log", format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class MotorClasificador:
    """Motor del clasificador de cuentas
    """

    def __init__(self, sc):
        """Inicializa el SparkContext
        """

        self.sc = sc
        self.juez_timelines = None
        self.modelo_spam = None
        self.datos = None
        logger.info("Calentando motores...")

    def entrenar_spam(self, dir_spam, dir_no_spam):

        sc = self.sc
        modelo = tools.entrenar_spam(sc, dir_spam, dir_no_spam)
        self.modelo_spam = modelo

        return True

    def cargar_spam(self, directorio):

        self.modelo_spam = RandomForestModel.load(self.sc, directorio)

        return True

    def entrenar_juez(self, directorio):
        sc = self.sc
        juez_spam = self.modelo_spam

        logger.info("Entrenando juez")

        juez_timelines = tools.entrenar_juez(sc, juez_spam, directorio)

        self.juez_timelines = juez_timelines

        logger.info("Finalizando...")

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
        return True

    #TODO guardar en BD resultados
    def evaluar(self, dir_timeline):
        sc = self.sc
        juez_timeline = self.juez_timelines
        juez_spam = self.modelo_spam
        resultado, features = tools.evaluar(sc, juez_spam, juez_timeline, dir_timeline)
        return resultado, features
