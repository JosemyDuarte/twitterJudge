import os
from pyspark.sql import SQLContext, Row
import logging, tools
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from dateutil import parser
from pyspark.mllib.util import MLUtils

os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(filename="logs/engine.log", format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO cambiar direcciones absolutas para que trabajen con HDFS
class MotorClasificador:
    """Motor del clasificador de cuentas
    """

    def __init__(self, sc):
        """Inicializa el SparkContext
        """

        self.sc = sc
        self.df = None
        self.tweets_RDD = None
        self.usuarios_RDD = None
        self.modelo = None
        self.sqlcontext = SQLContext(self.sc)
        logger.info("Calentando motores...")

    def carga_inicial(self, directorio):
        """Realiza la carga de los usuarios y guarda sus features
        """

        timeline = self.sc.textFile(directorio)

        logger.info("Cargando timelines...")
        self.df = self.sqlcontext.jsonRDD(timeline).cache()
        self.df.repartition(self.df.user.id)

        self.tweets_RDD = tools.tweets_rdd(self.df)

        self.usuarios_RDD = tools.usuario_rdd(self.df)

        logger.info("Calculo de features en tweetsRDD...")

        tweets_features = tools.tweets_features(self.tweets_RDD, self.sqlcontext)

        logger.info("Calculo de features en usuariosRDD...")

        usuarios_features = tools.usuarios_features(self.usuarios_RDD)

        logger.info("Realizando Join...")

        set_datos = usuarios_features.join(tweets_features, tweets_features.user_id == usuarios_features.user_id).map(
            lambda t: (t.user_id, (
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
                t.terceros)))

        logger.info("Finalizando...")

        set_datos.saveAsPickleFile("/home/jduarte/Workspace/datos/TIMELINES_PROCESADOS", 10)
        return True

    # TODO se pueden pasar parametro del modelo a esta funcion
    def cargar_modelo(self, directorio):

        timeline_humanos = self.sc.textFile(directorio["humano"])
        timeline_bots = self.sc.textFile(directorio["bot"])
        timeline_ciborgs = self.sc.textFile(directorio["ciborg"])

        logger.info("Cargando timelines...")
        df_humanos = self.sqlcontext.jsonRDD(timeline_humanos)
        df_humanos.repartition(df_humanos.user.id)

        df_bots = self.sqlcontext.jsonRDD(timeline_bots)
        df_bots.repartition(df_bots.user.id)

        df_ciborgs = self.sqlcontext.jsonRDD(timeline_ciborgs)
        df_ciborgs.repartition(df_ciborgs.user.id)

        tweets_RDD_humanos = tools.tweets_rdd(df_humanos)
        tweets_RDD_bots = tools.tweets_rdd(df_bots)
        tweets_RDD_ciborgs = tools.tweets_rdd(df_ciborgs)

        usuarios_RDD_humanos = tools.usuario_rdd(df_humanos)
        usuarios_RDD_bots = tools.usuario_rdd(df_bots)
        usuarios_RDD_ciborgs = tools.usuario_rdd(df_ciborgs)

        logger.info("Calculo de features en tweetsRDD_humanos...")
        tweets_features_humanos = tools.tweets_features(tweets_RDD_humanos, self.sqlcontext)

        logger.info("Calculo de features en tweetsRDD_bots...")
        tweets_features_bots = tools.tweets_features(tweets_RDD_bots, self.sqlcontext)

        logger.info("Calculo de features en tweetsRDD_ciborgs...")
        tweets_features_ciborgs = tools.tweets_features(tweets_RDD_ciborgs, self.sqlcontext)

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

        self.modelo = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                                  numTrees=3, featureSubsetStrategy="auto",
                                                  impurity='variance', maxDepth=4, maxBins=32)

        # Evaluate model on test instances and compute test error
        predictions = self.modelo.predict(testData.map(lambda x: x.features))
        labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
        testMSE = labelsAndPredictions.map(lambda v_p: (v_p[0] - v_p[1]) * (v_p[0] - v_p[1])).sum() / \
                  float(testData.count())

        logger.info("Finalizando...")

        return testMSE

