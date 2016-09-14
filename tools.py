# -*- coding: utf-8 -*-

from __future__ import division

import json
import logging
import math
import os
import sys
import urlparse
from datetime import datetime

import numpy as np
import pymongo_spark
import requests
from dateutil import parser
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import *


os.chdir(os.path.dirname(os.path.abspath(__file__)))
pymongo_spark.activate()

# logging.basicConfig(filename="logs/engine.log", format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def iniciar_spark_context(app_name=None, py_files=None):
    if not app_name:
        app_name = "ExtraerCaracteristicas"
    if not py_files:
        py_files = ['engine.py', 'app.py', 'tools.py']
    conf = SparkConf()
    conf.setAppName(app_name)
    sc = SparkContext(conf=conf, pyFiles=py_files)
    return sc


def hive_context(sc):
    return SparkSession().builder.getOrCreate()


def quantize(signal, partitions, codebook):
    indices = []
    quanta = []
    for datum in signal:
        index = 0
        while index < len(partitions) and datum > partitions[index]:
            index += 1
        indices.append(index)
        quanta.append(codebook[index])
    return indices, quanta


def pattern_mat(x, m):
    """
    Construct a matrix of `m`-length segments of `x`.
    Parameters
    ----------
    x : (N, ) array_like
        Array of input data.
    m : int
        Length of segment. Must be at least 1. In the case that `m` is 1, the
        input array is returned.
    Returns
    -------
    patterns : (m, N-m+1)
        Matrix whose first column is the first `m` elements of `x`, the second
        column is `x[1:m+1]`, etc.
    Examples
    --------
    > p = pattern_mat([1, 2, 3, 4, 5, 6, 7], 3])
    array([[ 1.,  2.,  3.,	4.,	 5.],
           [ 2.,  3.,  4.,	5.,	 6.],
           [ 3.,  4.,  5.,	6.,	 7.]])
    """
    x = np.asarray(x).ravel()
    if m == 1:
        return x
    else:
        n = len(x)
        patterns = np.zeros((m, n - m + 1))
        for i in range(m):
            patterns[i, :] = x[i:n - m + i + 1]
        return patterns


def en_shannon(series, l, num_int):
    if not series:
        raise ValueError("No hay serie definida")
    if not l:
        raise ValueError("No hay dimension (L) definida")
    if not num_int:
        raise ValueError("num_int sin definir")
    # Normalizacion
    series = (series - np.mean(series)) / np.std(series)
    # We the values of the parameters required for the quantification:
    epsilon = (max(series) - min(series)) / num_int
    partition = np.arange(min(series), math.ceil(max(series)), epsilon)
    codebook = np.arange(-1, num_int + 1)
    # Uniform quantification of the time series:
    _, quants = quantize(series, partition, codebook)
    # The minimum value of the signal quantified assert passes -1 to 0:
    quants = [0 if x == -1 else x for x in quants]
    n = len(quants)
    # We compose the patterns of length 'L':
    X = pattern_mat(quants, l)
    # We get the number of repetitions of each pattern:
    num = np.ones(n - l + 1)
    # This loop goes over the columns of 'X':
    if l == 1:
        X = np.atleast_2d(X)
    for j in range(0, n - l + 1):
        for i2 in range(j + 1, n - l + 1):
            tmp = [0 if x == -1 else 1 for x in X[:, j]]
            if (tmp[0] == 1) and (X[:, j] == X[:, i2]).all():
                num[j] += 1
                X[:, i2] = -1
            tmp = -1

    # We get those patterns which are not NaN:
    aux = [0 if x == -1 else 1 for x in X[0, :]]
    # Now, we can compute the number of different patterns:
    new_num = []
    for j, a in enumerate(aux):
        if a != 0:
            new_num.append(num[j])
    new_num = np.asarray(new_num)

    # We get the number of patterns which have appeared only once:
    unique = sum(new_num[new_num == 1])
    # We compute the probability of each pattern:
    p_i = new_num / (n - l + 1)
    # Finally, the Shannon Entropy is computed as:
    SE = np.dot((- 1) * p_i, np.log(p_i))

    return SE, unique


def cond_en(series, l, num_int):
    if not series:
        raise ValueError("No hay serie definida")
    if not l:
        raise ValueError("No hay dimension (L) definida")
    if not num_int:
        raise ValueError("num_int sin definir")
    # Processing:
    # First, we call the Shannon Entropy function:
    # 'L' as embedding dimension:
    se, unique = en_shannon(series, l, num_int)
    # 'L-1' as embedding dimension:
    se_1, _ = en_shannon(series, l - 1, num_int)
    # The Conditional Entropy is defined as a differential entropy:
    ce = se - se_1
    return ce, unique


def correc_cond_en(series, lmax, num_int):
    if not series:
        raise ValueError("No hay serie definida")
    if not lmax:
        raise ValueError("No hay dimension (L) definida")
    if not num_int:
        raise ValueError("num_int sin definir")
    N = len(series)
    # We will use this for the correction term: (L=1)
    e_est_1, _ = en_shannon(series, 1, num_int)
    # Incializacin de la primera posicin del vector que almacena la CCE a un
    # numero elevado para evitar que se salga del bucle en L=2 (primera
    # iteracin):
    # CCE is a vector that will contian the several CCE values computed:
    CCE = sys.maxsize * np.ones(lmax + 1)
    CCE[0] = 100
    CE = np.ones(lmax + 1)
    uniques = np.ones(lmax + 1)
    correc_term = np.ones(lmax + 1)
    for L in range(2, lmax + 1):
        # First, we compute the CE for the current embedding dimension: ('L')
        CE[L], uniques[L] = cond_en(series, L, num_int)
        # Second, we compute the percentage of patterns which are not repeated:
        perc_l = uniques[L] / (N - L + 1)
        correc_term[L] = perc_l * e_est_1
        # Third, the CCE is the CE plus the correction term:
        CCE[L] = CE[L] + correc_term[L]

    # Finally, the best estimation of the CCE is the minimum value of all the
    # CCE that have been computed:
    cce_min = min(CCE)
    return cce_min


def fuente(source):
    mobil = ["http://twitter.com/download/android", "Twitter for Android", "http://blackberry.com/twitter",
             "Twitter for BlackBerry", "https://mobile.twitter.com", "Mobile Web", "http://twitter.com/download/iphone",
             "iOS", "http://twitter.com/#!/download/ipad", "Huawei Social Phone", "Windows Phone",
             "Twitter for Nokia S40"]

    if "Twitter Web Client" in source:
        return 'uso_web'
    elif any(string in source for string in mobil):
        return 'uso_mobil'
    else:
        return 'uso_terceros'


month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7,
    'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}


def parse_time(s):
    return "{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}".format(
        int(s[-4:]),
        month_map[s[4:7]],
        int(s[8:10]),
        int(s[11:13]),
        int(s[14:16]),
        int(s[17:19])
    )


u_parse_time = F.udf(parse_time)


def avg_diversidad(tweets):
    _avg_diversidad = tweets.mapValues(
        lambda t: len(set(t[3].split(" "))) / len(t[3].split(" "))).combineByKey(lambda value: (value, 1),
                                                                                 lambda x, value: (
                                                                                     x[0] + value, x[1] + 1),
                                                                                 lambda x, y: (
                                                                                     x[0] + y[0], x[1] + y[1])).map(
        lambda label_value: Row(user_id=label_value[0], avg_diversidad=float(
            float(label_value[1][0]) / float(label_value[1][1])))).toDF().repartition("user_id")

    return _avg_diversidad


def avg_spam(juez, tweets):
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    wordsData = tokenizer.transform(tweets)

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=200)
    featurizedData = hashingTF.transform(wordsData)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    predictionsAndLabelsDF = juez.transform(rescaledData).groupBy("user_id").agg(
        F.avg('predicted_label').alias("avgSpam"))

    return predictionsAndLabelsDF


def preparar_df(df):
    df.repartition(df.user.id)

    df = df.where(F.length(df.text) > 0)
    df = df.select("*", u_parse_time(df['created_at']).cast('timestamp').alias('created_at_ts'))

    df_intertweet = df.select(df.user.id.alias("user_id"), (
        df.created_at_ts.cast('bigint') - F.lag(df.created_at_ts.cast('bigint'), ).over(
            Window.partitionBy("user.id").orderBy("created_at_ts"))).cast("bigint").alias("time_intertweet"))

    df_list_intertweet = df_intertweet.groupby(df_intertweet.user_id).agg(
        F.collect_list("time_intertweet").alias("lista_intertweet"))

    df = df.join(df_list_intertweet, df["user.id"] == df_list_intertweet["user_id"])

    return df


lengthOfArray = F.udf(lambda arr: len(arr), IntegerType())

nullToInt = F.udf(lambda e: 1 if e else 0, IntegerType())  # BooleanToInt, StringISEmpty

stringToDate = F.udf(lambda date: parser.parse(date), TimestampType())

reputacion = F.udf(lambda followers, friends:
                 float(followers) / (followers + friends) if (followers + friends > 0)  else 0, DoubleType())

followersRatio = F.udf(lambda followers, friends:
                     float(followers) / friends if (friends > 0)  else 0, DoubleType())

diversidadLexicograficaUDF = F.udf(lambda str: float(len(set(str))) / len(str) if str else 0, DoubleType())

cantPalabras = F.udf(lambda text: len(text.split(" ")), IntegerType())

fuentesUDF = F.udf(lambda source: fuente(source), StringType())

entropia = F.udf(lambda lista_intertweet:
                 float(correc_cond_en(lista_intertweet[1:110], len(lista_intertweet[1:110]),
                                      int(np.ceil(
                                          np.log2(max(lista_intertweet[1:110])))))), DoubleType())


def tweetsEnSemana(df):
    return df.groupBy("user_id", "nroTweets") \
        .pivot("dia", ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]) \
        .agg(F.count("text") / df["nroTweets"])


def tweetsAlDia(df):
    return df.groupBy("user_id", "nroTweets").pivot("hora", range(0, 24)).agg(F.count("text") / df["nroTweets"])


def fuenteTweets(df):
    return (df.groupBy("user_id", "nroTweets")
            .pivot("fuente", ["uso_web", "uso_mobil", "uso_terceros"])
            .agg(F.count("text") / df["nroTweets"]))


def dfParaTweets(df):
    return df.select(df.user.id.alias("user_id"), df.text, df.in_reply_to_status_id, df.entities, df.created_at,
                     df.source)


# TODO Avg de Diversidad de Palabras
def tweets_features(df, juez):
    nroTweetsDF = df.groupBy("user_id").agg(F.count("text").alias("nroTweets"))

    df = (df.join(nroTweetsDF, nroTweetsDF.user_id == df.user_id)
          .withColumn("fecha_tweet", u_parse_time("created_at").cast('timestamp'))
          .withColumn("mes", F.month("fecha_tweet"))
          .withColumn("dia", F.date_format("fecha_tweet", "EEEE"))
          .withColumn("hora", F.hour("fecha_tweet"))
          .withColumn("fuente", fuentesUDF("source"))
          .drop(nroTweetsDF.user_id))

    tweetsEnSemanaDF = tweetsEnSemana(df)

    tweetsAlDiaDF = tweetsAlDia(df)

    tweetsFuentesDF = fuenteTweets(df)

    featuresDF = df.groupBy("user_id", "nroTweets").agg(
        (sum(F.size("entities.urls")) / F.col("nroTweets")).alias("urlRatio"),
        (sum(diversidadLexicograficaUDF("text")) / F.col("nroTweets")).alias("diversidadLexicografica"),
        (sum(F.length("text")) / F.col("nroTweets")).alias("avgLongitudTweets"),
        (sum(nullToInt("in_reply_to_status_id")) / F.col("nroTweets")).alias("replyRatio"),
        (sum(lengthOfArray("entities.hashtags")) / F.col("nroTweets")).alias("avgHashtags"),
        (sum(lengthOfArray("entities.user_mentions")) / F.col("nroTweets")).alias("mentionRatio"),
        (sum(cantPalabras("text")) / F.col("nroTweets")).alias("avgPalabras"),
        (sum(lengthOfArray("entities.urls")) / F.col("nroTweets")).alias("urlRatio"))

    spamDF = avg_spam(juez, df)

    featSpamDF = (featuresDF
                  .join(spamDF, featuresDF.user_id == spamDF.user_id)
                  .drop(spamDF.user_id))

    featSpamSemDF = (featSpamDF
                     .join(tweetsEnSemanaDF, tweetsEnSemanaDF.user_id == featSpamDF.user_id)
                     .drop(tweetsEnSemanaDF.user_id))

    featSpamSemHrDF = (featSpamSemDF
                       .join(tweetsAlDiaDF, tweetsAlDiaDF.user_id == featSpamSemDF.user_id)
                       .drop(tweetsAlDiaDF.user_id))

    resultado = (featSpamSemHrDF
                 .join(tweetsFuentesDF, tweetsFuentesDF.user_id == featSpamSemHrDF.user_id)
                 .drop(tweetsFuentesDF.user_id))

    return resultado


def usuarios_features(df, categoria=-1.0):
    logger.info("Calculando features para usuarios...")

    resultado = (df.select("user.id",
                           nullToInt("user.profile_use_background_image").alias("conImagenFondo"),
                           u_parse_time("user.created_at").cast('timestamp').alias("cuentaCreada"),
                           df["user.favourites_count"].alias("nroFavoritos"),
                           nullToInt("user.description").alias("conDescripcion"),
                           F.length("user.description").alias("longitudDescripcion"),
                           nullToInt("user.verified").alias("conPerfilVerificado"),
                           nullToInt("user.default_profile_image").alias("conImagenDefault"),
                           df["user.listed_count"].alias("nroListas"),
                           nullToInt("user.geo_enabled").alias("conGeoAtivo"),
                           reputacion("user.followers_count", "user.friends_count").alias("reputacion"),
                           df["user.statuses_count"].alias("nroTweets"),
                           followersRatio("user.followers_count", "user.friends_count").alias("followersRatio"),
                           df["user.screen_name"].alias("nombreUsuario"),
                           entropia("lista_intertweet").alias("entropia")
                           )
                 .withColumn("anoCreada", F.year("cuentaCreada"))
                 .withColumn("categoria", F.lit(categoria)))

    return resultado


def entrenar_spam(sc, sql_context, dir_spam, dir_no_spam, num_trees=3, max_depth=2):
    input_spam = sc.textFile(dir_spam)
    input_no_spam = sc.textFile(dir_no_spam)

    spam = sql_context.read.json(input_spam).select("text").withColumn("label", F.lit(1.0))
    no_spam = sql_context.read.json(input_no_spam).select("text").withColumn("label", F.lit(0.0))

    training_data = spam.unionAll(no_spam)

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    wordsData = tokenizer.transform(training_data)

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=200)
    featurizedData = hashingTF.transform(wordsData)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    rf = RandomForestClassifier().setLabelCol("label") \
        .setPredictionCol("predicted_label") \
        .setFeaturesCol("features") \
        .setSeed(100088121L) \
        .setMaxDepth(max_depth) \
        .setNumTrees(num_trees)

    modelo = rf.fit(rescaledData)

    return modelo


def cargar_datos(sc, sql_context, directorio):
    timeline = sc.textFile(directorio)
    logger.info("Cargando arhcivos...")
    df = sql_context.read.json(timeline)
    df = preparar_df(df)
    return df


# TODO agregar features faltantes (safety, diversidad url)
def entrenar_juez(sc, sql_context, juez_spam, humanos, ciborgs, bots, mongo_uri=None, num_trees=3, max_depth=2):
    df_humanos = cargar_datos(sc, sql_context, humanos)
    df_bots = cargar_datos(sc, sql_context, bots)
    df_ciborgs = cargar_datos(sc, sql_context, ciborgs)

    tweets_df_humanos = dfParaTweets(df_humanos)
    tweets_df_bots = dfParaTweets(df_bots)
    tweets_df_ciborgs = dfParaTweets(df_ciborgs)

    tweetsDF = sc.union([tweets_df_bots, tweets_df_ciborgs, tweets_df_humanos])

    df_humanos = df_humanos.dropDuplicates(["user.id"])
    df_bots = df_bots.dropDuplicates(["user.id"])
    df_ciborgs = df_ciborgs.dropDuplicates(["user.id"])

    tweets = tweets_features(tweetsDF, juez_spam)

    usuarios_features_humanos = usuarios_features(df_humanos, 0.0)
    usuarios_features_ciborgs = usuarios_features(df_bots, 1.0)
    usuarios_features_bots = usuarios_features(df_ciborgs, 2.0)

    usuarios = usuarios_features_ciborgs.union(usuarios_features_bots).union(usuarios_features_humanos).cache()

    set_datos = usuarios.join(tweets, tweets.user_id == usuarios.user_id).cache()

    labeled_point = set_datos.map(
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
                                   t.uso_lunes,
                                   t.uso_martes,
                                   t.uso_miercoles,
                                   t.uso_jueves,
                                   t.uso_viernes,
                                   t.uso_sabado,
                                   t.uso_domingo,
                                   t.hora_0,
                                   t.hora_1,
                                   t.hora_2,
                                   t.hora_3,
                                   t.hora_4,
                                   t.hora_5,
                                   t.hora_6,
                                   t.hora_7,
                                   t.hora_8,
                                   t.hora_9,
                                   t.hora_10,
                                   t.hora_11,
                                   t.hora_12,
                                   t.hora_13,
                                   t.hora_14,
                                   t.hora_15,
                                   t.hora_16,
                                   t.hora_17,
                                   t.hora_18,
                                   t.hora_19,
                                   t.hora_20,
                                   t.hora_21,
                                   t.hora_22,
                                   t.hora_23,
                                   t.uso_web,
                                   t.uso_mobil,
                                   t.uso_terceros,
                                   t.entropia,
                                   0,
                                   t.avg_spam,
                                   0
                               ])).cache()

    modelo = RandomForest.trainClassifier(labeled_point, numClasses=3, categoricalFeaturesInfo={}, numTrees=num_trees,
                                          featureSubsetStrategy="auto", impurity='gini', maxDepth=max_depth, maxBins=32)

    if mongo_uri:
        set_datos.map(lambda t: t.asDict()).saveToMongoDB(mongo_uri)

    return modelo


def join_tw_usr(tw_features, usr_features):
    set_datos = usr_features.join(tw_features, tw_features.user_id == usr_features.user_id).map(
        lambda t: (Row(user_id=t.user_id,
                       ano_registro=t.ano_registro,
                       con_descripcion=t.con_descripcion,
                       con_geo_activo=t.con_geo_activo,
                       con_imagen_default=t.con_imagen_default,
                       con_imagen_fondo=t.con_imagen_fondo,
                       con_perfil_verificado=t.con_perfil_verificado,
                       followers_ratio=t.followers_ratio,
                       n_favoritos=t.n_favoritos,
                       n_listas=t.n_listas,
                       n_tweets=t.n_tweets,
                       reputacion=t.reputacion,
                       url_ratio=t.url_ratio,
                       avg_diversidad=t.avg_diversidad,
                       avg_palabras=t.avg_palabras,
                       mention_ratio=t.mention_ratio,
                       avg_hashtags=t.avg_hashtags,
                       reply_ratio=t.reply_ratio,
                       avg_long_tweets=t.avg_long_tweets,
                       avg_diversidad_lex=t.avg_diversidad_lex,
                       uso_lunes=t.uso_lunes,
                       uso_martes=t.uso_martes,
                       uso_miercoles=t.uso_miercoles,
                       uso_jueves=t.uso_jueves,
                       uso_viernes=t.uso_viernes,
                       uso_sabado=t.uso_sabado,
                       uso_domingo=t.uso_domingo,
                       hora_0=t.hora_0,
                       hora_1=t.hora_1,
                       hora_2=t.hora_2,
                       hora_3=t.hora_3,
                       hora_4=t.hora_4,
                       hora_5=t.hora_5,
                       hora_6=t.hora_6,
                       hora_7=t.hora_7,
                       hora_8=t.hora_8,
                       hora_9=t.hora_9,
                       hora_10=t.hora_10,
                       hora_11=t.hora_11,
                       hora_12=t.hora_12,
                       hora_13=t.hora_13,
                       hora_14=t.hora_14,
                       hora_15=t.hora_15,
                       hora_16=t.hora_16,
                       hora_17=t.hora_17,
                       hora_18=t.hora_18,
                       hora_19=t.hora_19,
                       hora_20=t.hora_20,
                       hora_21=t.hora_21,
                       hora_22=t.hora_22,
                       hora_23=t.hora_23,
                       uso_web=t.uso_web,
                       uso_mobil=t.uso_mobil,
                       uso_terceros=t.uso_terceros,
                       entropia=t.entropia,  # Entropia
                       diversidad_url=0,  # Diversidad
                       avg_spam=t.avg_spam,  # SPAM or not SPAM
                       safety_url=0,  # Safety url
                       createdAt=datetime.utcnow(),
                       nombre_usuario=t.nombre_usuario)))
    return set_datos


def timeline_features(juez_spam, df):
    _tweets_rdd = tweets_rdd(df)
    _tweets_features = tweets_features(_tweets_rdd, juez_spam)
    df = df.dropDuplicates(["user.id"])
    _usuarios_features = usuarios_features(df)
    logger.info("Realizando join de usuarios con tweets...")
    set_datos = join_tw_usr(_tweets_features, _usuarios_features)
    logger.info("Finalizado el join...")

    return set_datos


def predecir(juez_usuario, features):
    predicciones = juez_usuario.predict(features.map(lambda t: (t.ano_registro,
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
                                                                t.uso_lunes,
                                                                t.uso_martes,
                                                                t.uso_miercoles,
                                                                t.uso_jueves,
                                                                t.uso_viernes,
                                                                t.uso_sabado,
                                                                t.uso_domingo,
                                                                t.hora_0,
                                                                t.hora_1,
                                                                t.hora_2,
                                                                t.hora_3,
                                                                t.hora_4,
                                                                t.hora_5,
                                                                t.hora_6,
                                                                t.hora_7,
                                                                t.hora_8,
                                                                t.hora_9,
                                                                t.hora_10,
                                                                t.hora_11,
                                                                t.hora_12,
                                                                t.hora_13,
                                                                t.hora_14,
                                                                t.hora_15,
                                                                t.hora_16,
                                                                t.hora_17,
                                                                t.hora_18,
                                                                t.hora_19,
                                                                t.hora_20,
                                                                t.hora_21,
                                                                t.hora_22,
                                                                t.hora_23,
                                                                t.uso_web,
                                                                t.uso_mobil,
                                                                t.uso_terceros,
                                                                t.entropia,
                                                                t.diversidad_url,
                                                                t.avg_spam,
                                                                t.safety_url)))
    return predicciones


def evaluar(sc, sql_context, juez_spam, juez_usuario, dir_timeline, mongo_uri=None):
    df = cargar_datos(sc, sql_context, dir_timeline)
    features = timeline_features(juez_spam, df).cache()
    predicciones = predecir(juez_usuario, features)
    features = features.zip(predicciones).map(lambda t: dict(t[0].asDict().items() + [("prediccion", t[1])])).cache()
    if mongo_uri:
        features.saveToMongoDB(mongo_uri)

    return features
