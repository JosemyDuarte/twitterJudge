import os
from pyspark.sql import SQLContext, Row
import logging, tools
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from dateutil import parser
from pyspark.mllib.util import MLUtils

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#logging.basicConfig(filename="logs/engine.log", format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO cambiar direcciones absolutas para que trabajen con HDFS
class MotorClasificador:
    """Motor del clasificador de cuentas
    """

    def __init__(self, sc):
        """Inicializa el SparkContext
        """

        self.sc = sc
        self.modelo_juez = None
        self.modelo_spam = None
        logger.info("Calentando motores...")

    def carga_inicial(self, directorio):
        """Realiza la carga de los usuarios y guarda sus features
        """
        sc = self.sc
        set_datos = tools.timeline_features(sc, directorio)
        #TODO definir carpeta donde almacenar el resultado
        logger.info("Guardando resultados...")
        set_datos.saveAsPickleFile("/home/jduarte/Workspace/datos/TIMELINES_PROCESADOS", 10)

        return True


    # TODO se pueden pasar parametro del modelo a esta funcion
    def entrenar_juez(self, directorio):
        sc = self.sc
        sqlcontext = SQLContext(sc)
        timeline_humanos = sc.textFile(directorio["humano"])
        timeline_bots = sc.textFile(directorio["bot"])
        timeline_ciborgs = sc.textFile(directorio["ciborg"])

        logger.info("Cargando archivos...")
        df_humanos = sqlcontext.jsonRDD(timeline_humanos)
        df_humanos.repartition(df_humanos.user.id)

        df_bots = sqlcontext.jsonRDD(timeline_bots)
        df_bots.repartition(df_bots.user.id)

        df_ciborgs = sqlcontext.jsonRDD(timeline_ciborgs)
        df_ciborgs.repartition(df_ciborgs.user.id)

        tweets_RDD_humanos = tools.tweets_rdd(df_humanos)
        tweets_RDD_bots = tools.tweets_rdd(df_bots)
        tweets_RDD_ciborgs = tools.tweets_rdd(df_ciborgs)

        usuarios_RDD_humanos = tools.usuario_rdd(df_humanos)
        usuarios_RDD_bots = tools.usuario_rdd(df_bots)
        usuarios_RDD_ciborgs = tools.usuario_rdd(df_ciborgs)

        logger.info("Calculo de features en tweetsRDD_humanos...")
        tweets_features_humanos = tools.tweets_features(tweets_RDD_humanos, sqlcontext)

        logger.info("Calculo de features en tweetsRDD_bots...")
        tweets_features_bots = tools.tweets_features(tweets_RDD_bots, sqlcontext)

        logger.info("Calculo de features en tweetsRDD_ciborgs...")
        tweets_features_ciborgs = tools.tweets_features(tweets_RDD_ciborgs, sqlcontext)

        logger.info("Calculo de features en usuariosRDD_humanos...")
        usuarios_features_humanos = tools.usuarios_features(usuarios_RDD_humanos, 0)

        logger.info("Calculo de features en usuariosRDD_bots...")
        usuarios_features_bots = tools.usuarios_features(usuarios_RDD_bots, 1)

        logger.info("Calculo de features en usuariosRDD_ciborgs...")
        usuarios_features_ciborgs = tools.usuarios_features(usuarios_RDD_ciborgs, 2)

        logger.info("Realizando Union...")

        usuarios = usuarios_features_ciborgs.unionAll(usuarios_features_bots)
        usuarios = usuarios.unionAll(usuarios_features_humanos)
        # usuarios.cache()

        tweets = tweets_features_ciborgs.unionAll(tweets_features_bots)
        tweets = tweets.unionAll(tweets_features_humanos)

        logger.info("Realizando Join...")

        labeledPoint = usuarios.join(tweets, tweets.user_id == usuarios.user_id).map(
            lambda t: LabeledPoint(t.categoria,
                                   [
                                       t.ano_registro,
                                       t.con_descripcion,
                                       t.con_geo_activo,
                                       t.con_imagen_default,
                                       t.con_imagen_fondo,
                                       t.con_perfil_verificado,
                                       t.followers_ratio,
                                       t.n_favoritos,
                                       t.n_listas,
                                       t.n_tweets,
                                       t.reputacion,
                                       t.url_ratio,
                                       t.avg_diversidad,
                                       t.avg_palabras,
                                       t.mention_ratio,
                                       t.avg_hashtags,
                                       t.reply_ratio,
                                       t.avg_long_tweets,
                                       t.avg_diversidad_lex,
                                       t.Mon,
                                       t.Tue,
                                       t.Wed,
                                       t.Thu,
                                       t.Fri,
                                       t.Sat,
                                       t.Sun,
                                       t.h0,
                                       t.h1,
                                       t.h2,
                                       t.h3,
                                       t.h4,
                                       t.h5,
                                       t.h6,
                                       t.h7,
                                       t.h8,
                                       t.h9,
                                       t.h10,
                                       t.h11,
                                       t.h12,
                                       t.h13,
                                       t.h14,
                                       t.h15,
                                       t.h16,
                                       t.h17,
                                       t.h18,
                                       t.h19,
                                       t.h20,
                                       t.h21,
                                       t.h22,
                                       t.h23,
                                       t.web,
                                       t.mobil,
                                       t.terceros
                                   ]))

        (trainingData, testData) = labeledPoint.randomSplit([0.7, 0.3])

        logger.info("Entrenando juez...")
        modelo = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                    numTrees=100, featureSubsetStrategy="auto",
                                    impurity='variance', maxDepth=30, maxBins=32)


        predictions = modelo.predict(testData.map(lambda x: x.features))
        labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
        testMSE = labelsAndPredictions.map(lambda v_p: (v_p[0] - v_p[1]) * (v_p[0] - v_p[1])).sum() /\
    float(testData.count())

        logger.info("Guardando modelo...")

        modelo.save(sc, "/home/jduarte/Workspace/datos/modelo_juez")

        self.modelo_juez = modelo

        logger.info("Finalizando...")

        return testMSE

    def cargar_juez(self, directorio):
        self.modelo_juez = RandomForestModel.load(self.sc, directorio)
        return True

    def evaluar(self, dir_timeline):
        sc = self.sc
        modelo = self.modelo_juez
        features = tools.timeline_features(sc, dir_timeline)
        resultado = modelo.predict(features.map(lambda t: t[1])).collect()
        logger.info(resultado)
        return resultado, features




