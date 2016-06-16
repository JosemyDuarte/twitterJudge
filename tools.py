from __future__ import division
from pyspark.sql import SQLContext, Row
from dateutil import parser
import os
import logging
import json
import numpy as np
import CCE
import requests
import urlparse
from pyspark.mllib.feature import HashingTF

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#logging.basicConfig(filename="logs/engine.log", format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def lexical_diversity(text):
    if len(text) == 0:
        diversity = 0
    else:
        diversity = float(len(set(text))) / len(text)
    return diversity


def fuente(source):
    if "Twitter Web Client" in source:
        return 'web'
    elif "http://twitter.com/download/android" in source or "Twitter for Android" in source:
        return 'mobil'
    elif "http://blackberry.com/twitter" in source or "Twitter for BlackBerry" in source:
        return 'mobil'
    elif "https://mobile.twitter.com" in source or "Mobile Web" in source:
        return 'mobil'
    elif "http://twitter.com/download/iphone" in source or (
                    "http://www.apple.com" in source and "iOS" in source) or "http://twitter.com/#!/download/ipad" in source:
        return 'mobil'
    elif "Huawei Social Phone" in source:
        return 'mobil'
    elif "Windows Phone" in source or "Twitter for Nokia S40" in source:
        return 'mobil'
    else:
        return 'terceros'


def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z


def porcentaje_dias(x):
    aux = check_dias(x[1])
    suma = 0
    for key, value in aux.items():
        suma += value
    for key, value in aux.items():
        aux[key] = value / suma

    return Row(user_id=x[0], **dict(aux))


def porcentaje_horas(x):
    aux = check_horas(x[1])
    suma = 0
    for key, value in aux.items():
        suma += value
    for key, value in aux.items():
        aux[key] = value / suma

    return Row(user_id=x[0], **dict(aux))


def porcentaje_fuentes(x):
    aux = check_fuentes(x[1])
    suma = 0
    for key, value in aux.items():
        suma += value
    for key, value in aux.items():
        aux[key] = value / suma

    return Row(user_id=x[0], **dict(aux))


def check_dias(x):
    if "Mon" not in x:
        x["Mon"] = 0
    if "Tue" not in x:
        x["Tue"] = 0
    if "Wed" not in x:
        x["Wed"] = 0
    if "Thu" not in x:
        x["Thu"] = 0
    if "Fri" not in x:
        x["Fri"] = 0
    if "Sat" not in x:
        x["Sat"] = 0
    if "Sun" not in x:
        x["Sun"] = 0
    return x


def check_horas(x):
    if "00" not in x:
        x["00"] = 0
    if "01" not in x:
        x["01"] = 0
    if "02" not in x:
        x["02"] = 0
    if "03" not in x:
        x["03"] = 0
    if "04" not in x:
        x["04"] = 0
    if "05" not in x:
        x["05"] = 0
    if "06" not in x:
        x["06"] = 0
    if "07" not in x:
        x["07"] = 0
    if "08" not in x:
        x["08"] = 0
    if "09" not in x:
        x["09"] = 0
    if "10" not in x:
        x["10"] = 0
    if "11" not in x:
        x["11"] = 0
    if "12" not in x:
        x["12"] = 0
    if "13" not in x:
        x["13"] = 0
    if "14" not in x:
        x["14"] = 0
    if "15" not in x:
        x["15"] = 0
    if "16" not in x:
        x["16"] = 0
    if "17" not in x:
        x["17"] = 0
    if "18" not in x:
        x["18"] = 0
    if "19" not in x:
        x["19"] = 0
    if "20" not in x:
        x["20"] = 0
    if "21" not in x:
        x["21"] = 0
    if "22" not in x:
        x["22"] = 0
    if "23" not in x:
        x["23"] = 0
    return x


def check_fuentes(x):
    if "web" not in x:
        x["web"] = 0
    if "mobil" not in x:
        x["mobil"] = 0
    if "terceros" not in x:
        x["terceros"] = 0
    return x


def tweets_rdd(df):
    tweets_RDD = df.map(lambda t: (t.user.id, (
        t.user.id,
        t.user.screen_name,
        t.id,
        t.text,
        t.entities,
        t.is_quote_status,
        t.in_reply_to_status_id,
        t.favorite_count,
        t.source,
        t.retweet_count,
        t.geo,
        t.lang,
        t.created_at,
        t.place)))

    return tweets_RDD


def usuario_rdd(df):
    usuarios_RDD = df.map(lambda t: (t.user.id, (
        t.user.id,  # 0
        t.user.default_profile_image,  # 1
        t.user.followers_count,  # 2
        t.user.friends_count,  # 3
        t.user.verified,  # 4
        t.user.listed_count,  # 5
        t.user.statuses_count,  # 6
        t.user.geo_enabled,  # 7
        t.user.screen_name,  # 8
        t.user.lang,  # 9
        t.user.favourites_count,  # 10
        t.user.created_at,  # 11
        t.user.default_profile,  # 12
        t.user.is_translator,  # 13
        t.user.contributors_enabled,  # 14
        t.user.is_translation_enabled,  # 15
        t.user.description,  # 16
        t.user.profile_use_background_image,  # 17
        t.user.profile_background_tile,  # 18
        t.user.profile_link_color,  # 19
        t.user.profile_sidebar_border_color,  # 20
        t.user.profile_background_color,  # 21
        t.user.has_extended_profile,  # 22
        t.user.profile_text_color,  # 23
        t.user.location,  # 24
        t.user.url))).distinct()  # 25

    return usuarios_RDD


def tweets_x_dia(tweets):
    _tweets_x_dia = tweets.map(
        lambda t: ((t[0], parser.parse(t[1][12]).strftime('%a')), 1)).reduceByKey(
        lambda a, b: a + b).map(lambda t: (t[0][0], dict({t[0][1]: t[1]}))).reduceByKey(merge_two_dicts).map(
        porcentaje_dias).toDF().repartition("user_id")

    return _tweets_x_dia


def tweets_x_hora(tweets):
    _tweets_x_hora = tweets.map(
        lambda t: ((t[0], parser.parse(t[1][12]).strftime('%H')), 1)).reduceByKey(
        lambda a, b: a + b).map(lambda t: (t[0][0], dict({t[0][1]: t[1]}))).reduceByKey(merge_two_dicts).map(
        porcentaje_horas).toDF().repartition("user_id")

    return _tweets_x_hora


def fuentes_usuario(tweets):
    _fuentes_usuario = tweets.map(lambda t: ((t[0], fuente(t[1][8])), 1)).reduceByKey(
        lambda a, b: a + b).map(lambda t: (t[0][0], dict({t[0][1]: t[1]}))).reduceByKey(merge_two_dicts).map(
        porcentaje_fuentes).toDF().repartition("user_id")

    return _fuentes_usuario


def avg_diversidad_lexicografica(tweets):
    _avg_diversidad_lexicografica = tweets.mapValues(lambda t: lexical_diversity(t[3])).combineByKey(
        lambda value: (value, 1), lambda x, value: (x[0] + value, x[1] + 1),
        lambda x, y: (x[0] + y[0], x[1] + y[1])).map(lambda label_value: Row(user_id=label_value[0],
                                                                             avg_diversidad_lex=float(
                                                                                 float(label_value[1][0]) / float(
                                                                                     label_value[1][
                                                                                         1])))).toDF().repartition(
        "user_id")

    return _avg_diversidad_lexicografica


def avg_long_tweets_x_usuario(tweets):
    _avg_long_tweets_x_usuario = tweets.mapValues(lambda t: len(t[3])).combineByKey(
        lambda value: (value, 1),
        lambda x, value: (x[0] + value, x[1] + 1),
        lambda x, y: (x[0] + y[0], x[1] + y[1])).map(lambda label_value: Row(user_id=label_value[0],
                                                                             avg_long_tweets=float(
                                                                                 float(label_value[1][0]) / float(
                                                                                     label_value[1][
                                                                                         1])))).toDF().repartition(
        "user_id")

    return _avg_long_tweets_x_usuario


def reply_ratio(tweets):
    _reply_ratio = tweets.mapValues(lambda t: 1 if t[6] is not None else 0).combineByKey(
        lambda value: (value, 1),
        lambda x, value: (x[0] + value, x[1] + 1),
        lambda x, y: (x[0] + y[0], x[1] + y[1])).map(lambda label_value: Row(user_id=label_value[0],
                                                                             reply_ratio=float(
                                                                                 float(label_value[1][0]) / float(
                                                                                     label_value[1][
                                                                                         1])))).toDF().repartition(
        "user_id")

    return _reply_ratio


def avg_hashtags(tweets):
    _avg_hashtags = tweets.mapValues(lambda t: len(t[4].hashtags)).combineByKey(lambda value: (value, 1),
                                                                                lambda x, value: (
                                                                                    x[0] + value, x[1] + 1),
                                                                                lambda x, y: (x[0] + y[0],
                                                                                              x[1] + y[
                                                                                                  1])).map(
        lambda label_value: Row(user_id=label_value[0], avg_hashtags=float(
            float(label_value[1][0]) / float(label_value[1][1])))).toDF().repartition("user_id")

    return _avg_hashtags


def mention_ratio(tweets):
    _mention_ratio = tweets.mapValues(lambda t: len(t[4].user_mentions)).combineByKey(
        lambda value: (value, 1), lambda x, value: (x[0] + value, x[1] + 1),
        lambda x, y: (x[0] + y[0], x[1] + y[1])).map(lambda label_value: Row(user_id=label_value[0],
                                                                             mention_ratio=float(
                                                                                 float(label_value[1][0]) / float(
                                                                                     label_value[1][
                                                                                         1])))).toDF().repartition(
        "user_id")

    return _mention_ratio


def avg_palabras(tweets):
    _avg_palabras = tweets.mapValues(lambda t: len(t[3].split(" "))).combineByKey(lambda value: (value, 1),
                                                                                  lambda x, value: (
                                                                                      x[0] + value,
                                                                                      x[1] + 1),
                                                                                  lambda x, y: (x[0] + y[0],
                                                                                                x[1] + y[
                                                                                                    1])).map(
        lambda label_value: Row(user_id=label_value[0], avg_palabras=float(
            float(label_value[1][0]) / float(label_value[1][1])))).toDF().repartition("user_id")

    return _avg_palabras


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


def url_ratio(tweets):
    _url_ratio = tweets.mapValues(lambda t: len(t[4].urls)).combineByKey(lambda value: (value, 1),
                                                                         lambda x, value: (
                                                                             x[0] + value, x[1] + 1),
                                                                         lambda x, y: (
                                                                             x[0] + y[0], x[1] + y[1])).map(
        lambda label_value: Row(user_id=label_value[0], url_ratio=float(
            float(label_value[1][0]) / float(label_value[1][1])))).toDF().repartition("user_id")

    return _url_ratio


def spam_or_not(juez, tweets):
    tf = HashingTF(numFeatures=100)
    prediccion = juez.predict(tweets.map(lambda t: tf.transform(t[1][3].split(" "))))
    idYPrediccion = tweets.map(lambda t: t[0]).zip(prediccion)
    return idYPrediccion


def get_final_url(url):
    try:
        resultado = requests.get(url, timeout=10)
        if resultado.status_code >= 400:
            return str(resultado.status_code)
        else:
            return urlparse.urldefrag(resultado.url)[0]
    except Exception:
        return "600"


def diversidad_urls(urls):
    resultado_urls = []
    hosts = []
    diversidad_url = 0
    n_urls = 0
    logger.debug("Iniciando proceso para %s urls", str(len(urls)))
    for _url in urls:
        for url in _url:
            logger.debug("Obteniendo redireccion final de url %s", str(url["expanded_url"]))
            finalurl = get_final_url(url["expanded_url"])
            logger.debug("ANTES: %s RESULTADO: %s", str(url["expanded_url"]), str(finalurl))
            if not finalurl.isdigit():
                resultado_urls.append(finalurl)
                hosts.append(urlparse.urlparse(finalurl).netloc)
                n_urls += 1
    if n_urls != 0:
        diversidad_url = len(set(hosts)) / n_urls
    return diversidad_url


def intertweet_urls(directorio):
    lista_intertweet = []
    lista_urls = []
    i = 0
    with open(directorio) as timeline:
        lines = timeline.readlines()
        if json.loads(lines[0]) and len(lines) > 100:
            while i + 1 != len(lines) and i < 110:
                tweet = (json.loads(lines[i]), json.loads(lines[i + 1]))
                i += 1
                if i <= 110:
                    date = (parser.parse(tweet[0]['created_at']), parser.parse(tweet[1]['created_at']))
                    lista_intertweet.append(abs((date[1] - date[0]).total_seconds()))
                if tweet[0]['entities']['urls'] and tweet[0]['entities']['urls'][0]:
                    lista_urls.append(tweet[0]['entities']['urls'])
    return json.dumps(dict(intertweet_delay=lista_intertweet, user_id=json.loads(lines[0])["user"]["id"], urls=lista_urls))


def entropia_urls(directorio, urls=False):
    data = intertweet_urls(directorio)
    entropia = CCE.correc_cond_en(data["intertweet_delay"], len(data["intertweet_delay"]),
                                   int(np.ceil(np.log2(max(data["intertweet_delay"])))))
    if urls:
        diversidad = diversidad_urls(data["urls"])
        return entropia, diversidad
    return entropia


 #TODO falta SPAM y entropia, diversidad url
def tweets_features(tweets_RDD, sqlcontext):

    logger.info("Calculando features para tweets...")

    logger.info("Iniciando calculo de tweets por dia...")

    _tweets_x_dia = tweets_x_dia(tweets_RDD)

    logger.info("Iniciando calculo de tweets por hora...")

    _tweets_x_hora = tweets_x_hora(tweets_RDD)

    logger.info("Iniciando exploracion de las fuentes de los tweets...")

    _fuentes_usuario = fuentes_usuario(tweets_RDD)

    logger.info("Iniciando calculo de diversidad lexicografica...")

    _avg_diversidad_lexicografica = avg_diversidad_lexicografica(tweets_RDD)

    logger.info("Iniciando calculo del promedio de la longuitud de los tweets...")

    _avg_long_tweets_x_usuario = avg_long_tweets_x_usuario(tweets_RDD)

    logger.info("Iniciando calculo del ratio de respuestas...")

    _reply_ratio = reply_ratio(tweets_RDD)

    logger.info("Iniciando calculo del promedio de los hashtags...")

    _avg_hashtags = avg_hashtags(tweets_RDD)

    logger.info("Iniciando calculo del promedio de menciones...")

    _mention_ratio = mention_ratio(tweets_RDD)

    logger.info("Iniciando calculo del promedio de palabras por tweet...")

    _avg_palabras = avg_palabras(tweets_RDD)

    logger.info("Iniciando calculo del promedio de diversidad de palabras...")

    _avg_diversidad = avg_diversidad(tweets_RDD)

    logger.info("Iniciando calculo del ratio de urls...")

    _url_ratio = url_ratio(tweets_RDD)

    logger.info("Registrando tablas...")

    _url_ratio.registerTempTable("url_ratio")
    _avg_diversidad.registerTempTable("avg_diversidad")
    _avg_palabras.registerTempTable("avg_palabras")
    _mention_ratio.registerTempTable("mention_ratio")
    _avg_hashtags.registerTempTable("avg_hashtags")
    _reply_ratio.registerTempTable("reply_ratio")
    _avg_long_tweets_x_usuario.registerTempTable("avg_long_tweets")
    _avg_diversidad_lexicografica.registerTempTable("avg_diversidad_lex")
    _tweets_x_dia.registerTempTable("tweets_x_dia")
    _tweets_x_hora.registerTempTable("tweets_x_hora")
    _fuentes_usuario.registerTempTable("fuentes_usuario")

    logger.info("Join entre tweets...")

    _tweets_features = sqlcontext.sql(
    "select url_ratio.user_id, url_ratio, avg_diversidad, avg_palabras, mention_ratio, avg_hashtags, reply_ratio, avg_long_tweets, avg_diversidad_lex, Mon,Fri,Sat,Sun,Thu,Tue,Wed, `00` as h0,`01` as h1,`02` as h2,`03` as h3,`04` as h4,`05` as h5,`06` as h6,`07` as h7,`08` as h8,`09` as h9,`10` as h10,`11` as h11,`12` as h12,`13` as h13,`14` as h14,`15` as h15,`16` as h16,`17` as h17,`18` as h18,`19` as h19,`20` as h20,`21` as h21, `22` as h22, `23` as h23, mobil,terceros,web from url_ratio, avg_diversidad, avg_palabras, mention_ratio, avg_hashtags, reply_ratio, avg_long_tweets, avg_diversidad_lex, tweets_x_dia, tweets_x_hora, fuentes_usuario where url_ratio.user_id=avg_diversidad.user_id and avg_diversidad.user_id=avg_palabras.user_id and avg_palabras.user_id=mention_ratio.user_id and mention_ratio.user_id=avg_hashtags.user_id and avg_hashtags.user_id=reply_ratio.user_id and reply_ratio.user_id=avg_long_tweets.user_id and avg_long_tweets.user_id=avg_diversidad_lex.user_id and avg_diversidad_lex.user_id=tweets_x_dia.user_id and tweets_x_dia.user_id=tweets_x_hora.user_id and tweets_x_hora.user_id=fuentes_usuario.user_id")

    return _tweets_features


def usuarios_features(usuarios, categoria=-1):
    logger.info("Calculando features para usuarios...")
    _usuarios_features = usuarios.map(lambda t: Row(user_id=t[0],
                                                    con_imagen_fondo=(1 if t[1][17] == True else 0),
                                                    ano_registro=int(parser.parse(t[1][11]).strftime('%Y')),
                                                    n_favoritos=t[1][10],
                                                    con_descripcion=(1 if len(t[1][16]) > 0 else 0),
                                                    con_perfil_verificado=(1 if t[1][4] == True else 0),
                                                    con_imagen_default=(1 if t[1][1] == True else 0),
                                                    n_listas=t[1][5],
                                                    con_geo_activo=(1 if t[1][7] == True else 0),
                                                    reputacion=(t[1][2]/(t[1][2] + t[1][3]) if t[1][2] or t[1][3] or (t[1][2] + t[1][3] > 0) else 0),
                                                    n_tweets=t[1][6],
                                                    followers_ratio=(t[1][2] / t[1][3] if t[1][3] > 0 else 0),
                                                    categoria=categoria)).toDF()

    return _usuarios_features


def timeline_features(sc, directorio):

        timeline = sc.textFile(directorio)
        sqlcontext = SQLContext(sc)

        logger.info("Cargando arhcivos...")
        df = sqlcontext.jsonRDD(timeline)
        df.repartition(df.user.id)

        tweets_RDD = tweets_rdd(df)

        usuarios_RDD = usuario_rdd(df)

        _tweets_features = tweets_features(tweets_RDD, sqlcontext)

        _usuarios_features = usuarios_features(usuarios_RDD)

        logger.info("Realizando join de usuarios con tweets...")

        set_datos = _usuarios_features.join(_tweets_features, _tweets_features.user_id == _usuarios_features.user_id).map(
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
                t.terceros,
                0,#Diversidad url
                0,#Entropia
                0,#SPAM or not SPAM
                0))) #Safety url

        logger.info("Finalizado el join...")

        return set_datos

