from pyspark.sql import SQLContext
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def not_none(a):
    if a is not None:
        return 1
    else:
        return 0


def lexical_diversity(text):
    if len(text) == 0:
        diversity = 0
    else:
        diversity = float(len(set(text))) / len(text)
    return diversity


class MotorClasificador:
    """Motor del clasificador de cuentas
    """

    def __init__(self, sc, dataset_path):
        """Arranca el motor para las clasificaciones dado un Spark Context y una dirección para el dataset
        """

        self.sc = sc
        logger.info("Starting up the Recommendation Engine: ")

        timeline = sc.textFile(dataset_path)
        sqlcontext = SQLContext(self.sc)
        logger.info("Cargando timelines...")
        df1 = sqlcontext.jsonRDD(timeline)
        logger.info("Filtrando archivos vacios...")
        self.df = df1.filter("length(text)>0").cache()
        self.df.repartition(self.df.user.id)

        self.tweets_RDD = self.df.map(lambda t: (t.id, (
            t.user.id,
            t.user.screen_name,
            t.id,
            t.truncated,
            t.text,
            t.entities,
            t.is_quote_status,
            t.in_reply_to_status_id,
            t.favorite_count,
            t.source,
            t.in_reply_to_screen_name,
            t.in_reply_to_user_id,
            t.retweet_count,
            t.geo,
            t.lang,
            t.created_at,
            t.place)))

        # tweets.repartition(tweets.user.id)

        self.usuarios_RDD = self.df.map(lambda t: (t.user.id, (
            t.user.id,
            t.user.default_profile_image,
            t.user.followers_count,
            t.user.friends_count,
            t.user.verified,
            t.user.listed_count,
            t.user.statuses_count,
            t.user.geo_enabled,
            t.user.screen_name,
            t.user.lang,
            t.user.favourites_count,
            t.user.created_at,
            t.user.default_profile,
            t.user.is_translator,
            t.user.contributors_enabled,
            t.user.is_translation_enabled,
            t.user.description,
            t.user.profile_use_background_image,
            t.user.profile_background_tile,
            t.user.profile_link_color,
            t.user.profile_sidebar_border_color,
            t.user.profile_background_color,
            t.user.has_extended_profile,
            t.user.profile_text_color,
            t.user.location,
            t.user.url))).distinct()

    def cantidad_tweets(self):
        """Contar la cantidad de tweets en
                self.tweets_RDD
        """
        return self.tweets_RDD.count()

    def tweets_usuario(self):
        """Contar la cantidad de tweets
                en self.tweets_RDD y agrupar el resultado
                por usuario
        """
        return self.tweets_RDD.map(lambda t: (t[1][0], 1)).reduceByKey(lambda a, b: a + b)

    def caracteres_usuario(self):
        """Contar la cantidad de caracteres
                totales utilizados en todos los
                tweets de cada usuario
        """
        return self.tweets_RDD.map(lambda t: (t[1][0], len(t[1][4]))).reduceByKey(lambda a, b: a + b)

    def __sum_count_tweets_usuario(self):
        """Cuenta la cantidad de tweets por usuario
                y la cantidad de caracteres totales utilizados
                EJEMPLO: (id,caracteres totales, ntweets) EJEMPLO: [(191514816, (104415, 800))]
        """
        return self.tweets_RDD.map(lambda t: (t[1][0], len(t[1][4]))).combineByKey(
            lambda value: (value, 1),
            lambda x, value: (x[0] + value, x[1] + 1),
            lambda x, y: (x[0] + y[0], x[1] + y[1]))

    def avg_long_tweets_usuario(self):
        """Promedio de longitud de los tweets
                por usuario.
                EJEMPLO: (id_user,avg) -> [(191514816, 130.51875)]
        """
        return self.__sum_count_tweets_usuario().map(
            lambda label_value: (label_value[0], float(float(label_value[1][0]) / float(label_value[1][1]))))

    def __sum_count_diversidad_lex(self):
        """Cuenta el promedio de caracteres diferentes
                utilizados por los usuarios en sus
                tweets.
                :return: RDD (id, lexical_diversity, ntweets) [(187698476, (306.0745307679238, 999))
        """
        return self.tweets_RDD.map(lambda t: (t[1][0], lexical_diversity(t[1][4]))).combineByKey(
            lambda value: (value, 1),
            lambda x, value: (x[0] + value, x[1] + 1),
            lambda x, y: (x[0] + y[0], x[1] + y[1]))

    def avg_diversidad_lexicografica(self):
        """Promedio de diversidad lexicografica
                del usuario en sus tweets
                EJEMPLO: (id_user,avg) -> [(191514816, 130.51875)]
        """
        return self.__sum_count_diversidad_lex().map(
            lambda label_value: (label_value[0], float(float(label_value[1][0]) / float(label_value[1][1]))))

    def replys_usuario(self):
        """Cuenta la cantidad de replys que hace un usuario
                en su timeline
        :return: RDD cantidad de replys por ratio (id_user,n_replys)
        """
        return self.tweets_RDD.map(lambda t: (t[1][0], not_none(t[1][7]))).reduceByKey(lambda a, b: a + b)

    def __sum_count_reply(self):
        """ Cuenta la cantidad de replys y tweets totales por
                usuario
        :return: RDD con la cantidad de replys y el total de tweets
        EJEMPLO: (id, n_replys, n_tweets) [(187698476, (30, 999))
        """
        return self.tweets_RDD.map(lambda t: (t[1][0], not_none(t[1][7]))).combineByKey(
            lambda value: (value, 1),
            lambda x, value: (x[0] + value, x[1] + 1),
            lambda x, y: (x[0] + y[0], x[1] + y[1]))

    def reply_ratio(self):
        """ Promedio de replys en el timeline
                de los usuarios
        :return: RDD reply ratio por usuario
        """
        return self.__sum_count_reply().map(
            lambda label_value: (label_value[0], float(float(label_value[1][0]) / float(label_value[1][1]))))

    def __tweets_con_hashtags(self):
        """ Filtra los tweets que utilicen hashtags
        :return: RDD con solo tweets que tengan hashtags
        """
        return self.tweets_RDD.filter(lambda t: len(t[1][5].hashtags) > 0)

    def hashtags_usuario(self):
        """ Cuenta la cantidad de hashtags que utilizan los
        usuarios en sus timelines
        :return: RDD con la cantidad de hashtags que utiliza cada usuario
        """
        return self.__tweets_con_hashtags().map(lambda t: (t[1][0], len(t[1][5].hashtags))).reduceByKey(
            lambda a, b: a + b)

    def count_usuarios(self):
        """ Contar la cantidad de usuarios en el
                dataset
        :return: N de usuarios
        """
        return self.usuarios_RDD.count()

    def followers_friends_ratio(self):
        """ Calcula el ratio followers/friends de cada usuario
        :return: RDD con el ratio de followers/friends de cada usuario
        """
        return self.usuarios_RDD.mapValues(lambda t: (t[2], t[3], float(float(t[2]) / float(t[3]))))

    def reputacion(self):
        """ Calcula la reputación de cada cuenta
                reputacion = followers/(followers + friends)
        :return: RDD (iduser,reputacion)
        """
        return self.usuarios_RDD.mapValues(lambda t: (t[2], t[3], float(float(t[2]) / float(t[2]) + float(t[3]))))

    def geo_enable(self):
        """ Filtra la cuentas que posean el geo_enable = True
        :return: RDD de cuentas con geo_enable = True
        """
        return self.usuarios_RDD.filter(lambda t: t[1][7] == True)

    def perfil_verificado(self):
        """ Filtra las cuentas que posean el perfil verificado
        :return: RDD de cuentas con perfil verficado
        """
        return self.usuarios_RDD.filter(lambda t: t[1][4] == True)

    def imagen_perfil_default(self):
        """ Filtra las cuentas que no posean la imagen de perfil por defecto
        :return: RDD de cuentas con imagen de perfil por defecto
        """
        return self.usuarios_RDD.filter(lambda t: t[1][1] == True)

    def get_user_by_id(self, user_id):
        """ Obtiene la informacion referente al id del usuario solicitado
        :param user_id: id del usuario a recuperar
        :return: usuario
        """
        return self.usuarios_RDD.filter(lambda t: t[0] == user_id).first()

    def get_user_by_screen_name(self, user_screen_name):
        """ Obtiene la informacion referente al screen_name solicitado
        :param user_screen_name: screen name del usuario requerido
        :return: usuario
        """
        return self.usuarios_RDD.filter(lambda t: t[1][8] == user_screen_name).first()

    def fuentes_distintas_general(self):
        """Retorna las distintas fuentes generadas por todos los tweets
        y la cantidad de veces utilizada
        :return: (fuente:Nrepticiones) EJEMPLO: {u'<a href="http://www.hootsuite.com" rel="nofollow">Hootsuite</a>': 861}
        """
        return self.tweets_RDD.map(lambda t: t[1][9]).countByValue()
