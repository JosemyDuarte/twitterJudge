# -*- coding: utf-8 -*-

from __future__ import division

import logging
import math
import os
import sys

import numpy as np
import pymongo_spark
from dateutil import parser
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.mllib.feature import HashingTF
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql.types import *
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

os.chdir(os.path.dirname(os.path.abspath(__file__)))
pymongo_spark.activate()

# logging.basicConfig(filename="logs/engine.log", format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def iniciar_spark_context(app_name=None, py_files=None, level="ERROR"):
    if not app_name:
        app_name = "ExtraerCaracteristicas"
    if not py_files:
        py_files = ['workspace/engine.py', 'workspace/app.py', 'workspace/tools.py']
    conf = SparkConf()
    conf.setAppName(app_name)
    sc = SparkContext.getOrCreate(conf=conf)
    #sc.setLogLevel(level)
    for file in py_files:
        sc.addPyFile(file)
    return sc


def spark_session():
    return SparkSession.builder.getOrCreate()


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


def avg_spam(juez, tweets):
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    wordsData = tokenizer.transform(tweets)

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=140)
    featurizedData = hashingTF.transform(wordsData)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    predictionsAndLabelsDF = juez.transform(rescaledData).groupBy("user_id").agg(
        F.avg('predicted_label').alias("avg_spam"))

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

    df_list_intertweet = df_list_intertweet.filter(F.size(df_list_intertweet.lista_intertweet) > 3)

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
                                      len(lista_intertweet[1:110]))), DoubleType())

diversidadPalabras = F.udf(lambda text: len(set(text.split(" "))) / len(text.split(" ")), DoubleType())


def tweets_en_semana(df):
    return (df.groupBy("user_id", "nroTweets")
            .pivot("dia", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            .agg(F.count("text") / df["nroTweets"]))


def tweets_al_dia(df):
    return df.groupBy("user_id", "nroTweets").pivot("hora", range(0, 24)).agg(F.count("text") / df["nroTweets"])


def fuente_tweets(df):
    return (df.groupBy("user_id", "nroTweets")
            .pivot("fuente", ["uso_web", "uso_mobil", "uso_terceros"])
            .agg(F.count("text") / df["nroTweets"]))


def df_para_tweets(df):
    return df.select(df.user.id.alias("user_id"),
                     df.text,
                     df.in_reply_to_status_id,
                     df.entities.urls.alias("entities_url"),
                     df.entities.hashtags.alias("entities_hashtags"),
                     df.entities.user_mentions.alias("entities_user_mentions"),
                     df.created_at,
                     df.source)


def tweets_features(df, juez):
    nro_tweets_df = df.groupBy("user_id").agg(F.count("text").alias("nroTweets"))

    logger.info("Calculando features para tweets...")

    df = (df.join(nro_tweets_df, nro_tweets_df.user_id == df.user_id)
          .withColumn("fecha_tweet", u_parse_time("created_at").cast('timestamp'))
          .withColumn("mes", F.month("fecha_tweet"))
          .withColumn("dia", F.date_format("fecha_tweet", "EEEE"))
          .withColumn("hora", F.hour("fecha_tweet"))
          .withColumn("fuente", fuentesUDF("source"))
          .drop(nro_tweets_df.user_id))

    tweets_en_semana_df = tweets_en_semana(df)

    tweets_al_dia_df = tweets_al_dia(df)

    tweets_fuentes_df = fuente_tweets(df)

    featuresDF = df.groupBy("user_id", "nroTweets").agg(
        (F.sum(F.size("entities_url")) / F.col("nroTweets")).alias("url_ratio"),
        (F.sum(diversidadLexicograficaUDF("text")) / F.col("nroTweets")).alias("avg_diversidad_lex"),
        (F.sum(F.length("text")) / F.col("nroTweets")).alias("avg_long_tweets"),
        (F.sum(nullToInt("in_reply_to_status_id")) / F.col("nroTweets")).alias("reply_ratio"),
        (F.sum(lengthOfArray("entities_hashtags")) / F.col("nroTweets")).alias("avg_hashtags"),
        (F.sum(lengthOfArray("entities_user_mentions")) / F.col("nroTweets")).alias("mention_ratio"),
        (F.sum(cantPalabras("text")) / F.col("nroTweets")).alias("avg_palabras"),
        (F.sum(diversidadPalabras("text")) / F.col("nroTweets")).alias("avg_diversidad_palabras"))

    spam_df = avg_spam(juez, df)

    feat_spam_df = (featuresDF
                    .join(spam_df, featuresDF.user_id == spam_df.user_id)
                    .drop(spam_df.user_id))

    feat_spam_sem_df = (feat_spam_df
                        .join(tweets_en_semana_df, tweets_en_semana_df.user_id == feat_spam_df.user_id)
                        .drop(tweets_en_semana_df.user_id)
                        .drop(tweets_en_semana_df.nroTweets))

    feat_spam_sem_hr_df = (feat_spam_sem_df
                           .join(tweets_al_dia_df, tweets_al_dia_df.user_id == feat_spam_sem_df.user_id)
                           .drop(tweets_al_dia_df.user_id)
                           .drop(tweets_al_dia_df.nroTweets))

    resultado = (feat_spam_sem_hr_df
                 .join(tweets_fuentes_df, tweets_fuentes_df.user_id == feat_spam_sem_hr_df.user_id)
                 .drop(tweets_fuentes_df.user_id)
                 .drop(tweets_fuentes_df.nroTweets))

    logger.info("Terminando calculo de features para tweets...")

    return resultado


def usuarios_features(df, categoria=-1.0):
    logger.info("Calculando features para usuarios...")

    resultado = (df.select(df["user.id"].alias("user_id"),
                           nullToInt("user.profile_use_background_image").alias("con_imagen_fondo"),
                           u_parse_time("user.created_at").cast('timestamp').alias("cuenta_creada"),
                           df["user.favourites_count"].alias("n_favoritos"),
                           nullToInt("user.description").alias("con_descripcion"),
                           F.length("user.description").alias("longitud_descripcion"),
                           nullToInt("user.verified").alias("con_perfil_verificado"),
                           nullToInt("user.default_profile_image").alias("con_imagen_default"),
                           df["user.listed_count"].alias("n_listas"),
                           nullToInt("user.geo_enabled").alias("con_geo_activo"),
                           reputacion("user.followers_count", "user.friends_count").alias("reputacion"),
                           df["user.statuses_count"].alias("n_tweets"),
                           followersRatio("user.followers_count", "user.friends_count").alias("followers_ratio"),
                           df["user.screen_name"].alias("nombre_usuario"),
                           entropia("lista_intertweet").alias("entropia")
                           )
                 .withColumn("ano_registro", F.year("cuenta_creada"))
                 .withColumn("categoria", F.lit(categoria))
                 .withColumn("createdAt", F.current_timestamp()))

    return resultado


def entrenar_spam(sc, sql_context, dir_spam, dir_no_spam, num_trees=20, max_depth=8):
    input_spam = sc.textFile(dir_spam)
    input_no_spam = sc.textFile(dir_no_spam)

    spam = sql_context.read.json(input_spam).select("text").withColumn("label", F.lit(1.0))
    no_spam = sql_context.read.json(input_no_spam).select("text").withColumn("label", F.lit(0.0))

    training_data = spam.unionAll(no_spam)

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    wordsData = tokenizer.transform(training_data)

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=140)
    featurizedData = hashingTF.transform(wordsData)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    seed = 1800009193L
    (split_20_df, split_80_df) = rescaledData.randomSplit([20.0, 80.0], seed)

    test_set_df = split_20_df.cache()
    training_set_df = split_80_df.cache()

    rf = RandomForestClassifier().setLabelCol("label") \
        .setPredictionCol("predicted_label") \
        .setFeaturesCol("features") \
        .setSeed(100088121L) \
        .setMaxDepth(max_depth) \
        .setNumTrees(num_trees)

    rf_pipeline = Pipeline()
    rf_pipeline.setStages([rf])

    reg_eval = MulticlassClassificationEvaluator(predictionCol="predicted_label", labelCol="label",
                                                 metricName="accuracy")

    crossval = CrossValidator(estimator=rf_pipeline, evaluator=reg_eval, numFolds=5)
    param_grid = ParamGridBuilder().addGrid(rf.maxBins, [50, 100]).build()
    crossval.setEstimatorParamMaps(param_grid)
    modelo = crossval.fit(training_set_df).bestModel

    predictions_and_labels_df = modelo.transform(test_set_df)

    accuracy = reg_eval.evaluate(predictions_and_labels_df)

    return modelo, accuracy


def cargar_datos(sc, sql_context, directorio):
    timeline = sc.textFile(directorio)
    logger.info("Cargando arhcivos...")
    df = sql_context.read.json(timeline)
    df = preparar_df(df)
    return df


# TODO agregar features faltantes (safety, diversidad url)
def entrenar_juez(sc, sql_context, juez_spam, humanos, ciborgs, bots, mongo_uri=None, num_trees=20, max_depth=8):

    logger.info("Entrenando juez...")
    df_humanos = cargar_datos(sc, sql_context, humanos)
    df_bots = cargar_datos(sc, sql_context, bots)
    df_ciborgs = cargar_datos(sc, sql_context, ciborgs)

    tweets_humanos = df_para_tweets(df_humanos)
    tweets_bots = df_para_tweets(df_bots)
    tweets_ciborgs = df_para_tweets(df_ciborgs)

    tweets_df = tweets_humanos.union(tweets_bots).union(tweets_ciborgs)

    df_humanos = df_humanos.dropDuplicates(["user_id"])
    df_bots = df_bots.dropDuplicates(["user_id"])
    df_ciborgs = df_ciborgs.dropDuplicates(["user_id"])

    tweets = tweets_features(tweets_df, juez_spam)
    tweets.cache()

    usuarios_features_humanos = usuarios_features(df_humanos, 0.0)
    usuarios_features_ciborgs = usuarios_features(df_bots, 1.0)
    usuarios_features_bots = usuarios_features(df_ciborgs, 2.0)

    usuarios = usuarios_features_ciborgs.union(usuarios_features_bots).union(usuarios_features_humanos).cache()

    set_datos = usuarios.join(tweets, tweets.user_id == usuarios.user_id).drop(tweets.user_id).fillna(0).cache()

    seed = 1800009193L
    (split_20_df, split_80_df) = set_datos.randomSplit([20.0, 80.0], seed)

    test_set_df = split_20_df.cache()
    training_set_df = split_80_df.cache()

    vectorizer = VectorAssembler()
    vectorizer.setInputCols([
        "ano_registro", "con_descripcion", "con_geo_activo", "con_imagen_default", "con_imagen_fondo",
        "con_perfil_verificado", "entropia", "followers_ratio", "n_favoritos", "n_listas", "n_tweets", "reputacion",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "0", "1", "2", "3", "4", "5", "6",
        "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "uso_mobil",
        "uso_terceros", "uso_web", "avg_diversidad_lex", "avg_long_tweets", "reply_ratio", "avg_hashtags",
        "mention_ratio", "avg_palabras", "avg_diversidad_palabras", "url_ratio", "avg_spam"
    ])

    vectorizer.setOutputCol("features")

    rf = RandomForestClassifier()

    rf.setLabelCol("categoria") \
        .setPredictionCol("Predicted_categoria") \
        .setFeaturesCol("features") \
        .setSeed(seed) \
        .setMaxDepth(max_depth) \
        .setNumTrees(num_trees)

    rf_pipeline = Pipeline()
    rf_pipeline.setStages([vectorizer, rf])

    reg_eval = MulticlassClassificationEvaluator(predictionCol="Predicted_categoria", labelCol="categoria",
                                                 metricName="accuracy")

    crossval = CrossValidator(estimator=rf_pipeline, evaluator=reg_eval, numFolds=5)
    param_grid = ParamGridBuilder().addGrid(rf.maxBins, [50, 100]).build()
    crossval.setEstimatorParamMaps(param_grid)

    logger.info("Buscando el mejor modelo de RandomForest")

    rf_model = crossval.fit(training_set_df).bestModel

    logger.info("Guardando en Mongo el set de entrenamiento")

    if mongo_uri:
        training_set_df.rdd.map(lambda t: t.asDict()).saveToMongoDB(mongo_uri)

    logger.info("Evaluando set de prueba")

    predictions_and_labels_df = rf_model.transform(test_set_df)
    predictions_and_labels_df.cache()

    accuracy = reg_eval.evaluate(predictions_and_labels_df)

    logger.info("Calculando matriz de confusion")

    hh = predictions_and_labels_df[(predictions_and_labels_df.categoria == 0) & (predictions_and_labels_df.Predicted_categoria == 0)].count()
    hb = predictions_and_labels_df[(predictions_and_labels_df.categoria == 0) & (predictions_and_labels_df.Predicted_categoria == 1)].count()
    hc = predictions_and_labels_df[(predictions_and_labels_df.categoria == 0) & (predictions_and_labels_df.Predicted_categoria == 2)].count()

    bh = predictions_and_labels_df[(predictions_and_labels_df.categoria == 1) & (predictions_and_labels_df.Predicted_categoria == 0)].count()
    bb = predictions_and_labels_df[(predictions_and_labels_df.categoria == 1) & (predictions_and_labels_df.Predicted_categoria == 1)].count()
    bc = predictions_and_labels_df[(predictions_and_labels_df.categoria == 1) & (predictions_and_labels_df.Predicted_categoria == 2)].count()

    ch = predictions_and_labels_df[(predictions_and_labels_df.categoria == 2) & (predictions_and_labels_df.Predicted_categoria == 0)].count()
    cb = predictions_and_labels_df[(predictions_and_labels_df.categoria == 2) & (predictions_and_labels_df.Predicted_categoria == 1)].count()
    cc = predictions_and_labels_df[(predictions_and_labels_df.categoria == 2) & (predictions_and_labels_df.Predicted_categoria == 2)].count()

    return rf_model, accuracy, [[hh,hb,hc],[bh,bb,bc],[ch,cb,cc]]


def timeline_features(juez_spam, df):
    tweets_df = df_para_tweets(df)
    tweets_features_df = tweets_features(tweets_df, juez_spam)
    df = df.dropDuplicates(["user_id"])
    usuarios_features_df = usuarios_features(df)
    logger.info("Realizando join de usuarios con tweets...")
    set_datos = (usuarios_features_df
                 .join(tweets_features_df, tweets_features_df.user_id == usuarios_features_df.user_id)
                 .drop(tweets_features_df.user_id)
                 .fillna(0))
    logger.info("Finalizado el join...")

    return set_datos


def predecir(juez_usuario, features):
    predicciones = (juez_usuario
                    .transform(features)
                    .select("user_id", "ano_registro", "con_descripcion", "con_geo_activo", "nroTweets",
                            "con_imagen_default", "con_imagen_fondo", "con_perfil_verificado", "entropia",
                            "followers_ratio", "n_favoritos", "n_listas", "n_tweets", "reputacion", "Monday", "Tuesday",
                            "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "0", "1", "2", "3", "4", "5", "6",
                            "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22",
                            "23", "uso_mobil", "uso_terceros", "uso_web", "avg_diversidad_lex", "avg_long_tweets",
                            "reply_ratio", "avg_hashtags", "mention_ratio", "avg_palabras", "avg_diversidad_palabras",
                            "url_ratio", "avg_spam", "Predicted_categoria", "nombre_usuario"))
    return predicciones


def evaluar(sc, sql_context, juez_spam, juez_usuario, dir_timeline, mongo_uri=None):
    df = cargar_datos(sc, sql_context, dir_timeline)
    features = timeline_features(juez_spam, df).cache()
    predicciones = predecir(juez_usuario, features)
    if mongo_uri:
        predicciones.rdd.map(lambda t: t.asDict()).saveToMongoDB(mongo_uri)

    return predicciones


def features_importances_juez(juez):
    return juez.stages[1].featureImportances
