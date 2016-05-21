import os
from pyspark.sql import SQLContext, Row
import logging, tools
from dateutil import parser
from pyspark.mllib.util import MLUtils

os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(filename="logs/engine.log", format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


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
        """Realiza la carga inicial del clasificador
        y entrena el modelo
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

        return set_datos.collect()

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
        return self.tweets_RDD.map(lambda t: (t[1][0], tools.lexical_diversity(t[1][4]))).combineByKey(
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
        # return self.tweets_RDD.map(lambda t: (t[1][0], not_none(t[1][7]))).reduceByKey(lambda a, b: a + b)

    def __sum_count_reply(self):
        """ Cuenta la cantidad de replys y tweets totales por
                usuario
        :return: RDD con la cantidad de replys y el total de tweets
        EJEMPLO: (id, n_replys, n_tweets) [(187698476, (30, 999))
        """
        """return self.tweets_RDD.map(lambda t: (t[1][0], not_none(t[1][7]))).combineByKey(
            lambda value: (value, 1),
            lambda x, value: (x[0] + value, x[1] + 1),
            lambda x, y: (x[0] + y[0], x[1] + y[1]))"""

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
        """ Calcula la reputacion de cada cuenta
                reputacion = followers/(followers + friends)
        :return: RDD (iduser,(followers, friends, reputacion)
        """
        return self.usuarios_RDD.mapValues(lambda t: (t[2], t[3], float(float(t[2]) / (float(t[2]) + float(t[3])))))

    def get_reputacion(self, user_id):
        """ Calcula la reputacion del usuario especificado
                reputacion = followers/(followers + friends)
        :param user_id: id del usuario a calcular la reputacion
        :return: Array: (iduser,(followers, friends, reputacion)
         EJEMPLO: [(194598018, (184, 79, 80.0))]
        """
        return self.usuarios_RDD.filter(lambda t: t[0] == user_id). \
            mapValues(lambda t: (t[2], t[3], float(float(t[2]) / float(t[2]) + float(t[3])))).collect()

    def geo_enable(self):
        """ Filtra la cuentas que posean el geo_enable = True
        :return: RDD de cuentas con geo_enable = True
        """
        return self.usuarios_RDD.filter(lambda t: t[1][7] == True)

    def check_user_geo_enable(self, user_id):
        """ Revisa si el usuario posee la opcion geo_enable activada
        :param user_id: id del usuario a verificar
        :return: True or false
        """
        if len(self.usuarios_RDD.filter(lambda t: t[0] == user_id and t[1][7] == True).collect()) > 0:
            return True
        else:
            return False

    def perfil_verificado(self):
        """ Filtra las cuentas que posean el perfil verificado
        :return: RDD de cuentas con perfil verficado
        """
        return self.usuarios_RDD.filter(lambda t: t[1][4] == True)

    def check_perfil_verificado(self, user_id):
        """ Revisa si el usuario posee el perfil verificado
        :param user_id: id del usuario a verificar
        :return: True or false
        """
        if len(self.usuarios_RDD.filter(lambda t: t[0] == user_id and t[1][4] == True).collect()) > 0:
            return True
        else:
            return False

    def imagen_perfil_default(self):
        """ Filtra las cuentas que no posean la imagen de perfil por defecto
        :return: RDD de cuentas con imagen de perfil por defecto
        """
        return self.usuarios_RDD.filter(lambda t: t[1][1] == True)

    def check_imagen_perfil_default(self, user_id):
        """ Revisa si el usuario posee la imagen de perfil por defecto
        :param user_id: id del usuario a verificar
        :return: True or false
        """
        if len(self.usuarios_RDD.filter(lambda t: t[0] == user_id and t[1][1] == True).collect()) > 0:
            return True
        else:
            return False

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
        """
        Retorna las distintas fuentes generadas por todos los tweets
        y la cantidad de veces utilizada
        :return: RDD (fuente,Nrepticiones)
        EJEMPLO: {u'<a href="http://www.hootsuite.com" rel="nofollow">Hootsuite</a>', 861}
        """
        return self.tweets_RDD.map(lambda t: (t[1][9], 1)).reduceByKey(lambda a, b: a + b)

    def fuentes_mas_utilizadas_general(self, n, desc=True):
        """ Retorna las n fuentes mas utilizadas en la base general de tweets
        :param n: cantidad de fuentes a retornar
        :param desc: orden en el que se desea obtener el resultado
        :return: Array con las N fuentes MAS o MENOS utilizadas, dependiendo de si se ordena
        ascendente o descendete.
        EJEMPLO: RDD.fuentes_mas_utilizadas_general(2) =>
        [(u'<a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>', 459),
        (u'<a href="http://blackberry.com/twitter" rel="nofollow">Twitter for BlackBerry\xae</a>', 183)]
        """
        if desc:
            return self.fuentes_distintas_general().takeOrdered(n, key=lambda x: -x[1])
        else:
            return self.fuentes_distintas_general().takeOrdered(n, key=lambda x: x[1])

    def fuentes_por_usuario(self):
        """ Calcula la cantidad de fuentes distintas utilizadas
        en el timeline de cada usuario
        :return RDD (id_usuario,fuente,nveces)
        """
        return self.tweets_RDD.map(lambda t: ((t[1][0], t[1][9]), 1)).reduceByKey(lambda a, b: a + b)

    def fuentes_de_usuario(self, user_id):
        """ Retorna las fuentes utilizadas por el usuario dado
        :param user_id: id del usuario a retornar
        :return Array ((user_id,fuente),nveces)
        EJEMPLO: [((192286676, u'<a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>'), 106),
         ((192286676, u'<a href="https://about.twitter.com/products/tweetdeck" rel="nofollow">TweetDeck</a>'), 59)]
        """
        return self.tweets_RDD().filter(lambda t: t[1][0] == user_id).map(lambda t: ((t[1][0], t[1][9]), 1)) \
            .reduceByKey(lambda a, b: a + b).collect()

    def fuentes_mas_utilizadas_de_usuario(self, user_id, n, desc=True):
        """ Retorna las n fuentes mas utilizadas del usuario user_id
        :param user_id: ID del usuario a buscar
        :param n: numero de fuentes a retornar
        :param desc: Ordenar de forma descendente o ascendente?
        :return: Array con las N fuentes MAS o MENOS utilizadas, dependiendo del ordenado
        ascendente o descendete.
        EJEMPLO:  RDD.fuentes_mas_utilizadas_de_usuario(2,183641931) =>
        [((183641931, u'<a href="http://twitter.com/download/android" rel="nofollow">Twitter for Android</a>'), 52),
         ((183641931, u'<a href="http://www.steelthorn.com" rel="nofollow">QuickPull</a>'), 2)]
        """
        if desc:
            return self.fuentes_de_usuario(user_id).takeOrdered(n, key=lambda x: -x[1])
        else:
            return self.fuentes_de_usuario(user_id).takeOrdered(n, key=lambda x: x[1])

    def avg_palabras(self):
        """ Calcula el promedio de palabras que utilizan los usuario
        en sus tweets
        :return: RDD (id_usuario,avg_palabras)
        """
