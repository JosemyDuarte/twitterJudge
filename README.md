# Twitter Judge

## Tabla de contenido

* [Introducción](#introducción)
* [Instalación](#instalación)
* [Quick Start](#quickstart)
* [Referencias](#referencias)

### Introducción
Twitter Judge es un clasificador de usuarios de Twitter en Humanos, Bots o Ciborgs. Este es un proyecto desarrollado con Apache Spark en su versión 2.0 y su respectiva librería para Machine Learning. Todo el código se encuentra en Python. El proyecto se encuentra dockerizado para facilitar su despliegue y utilización, sin embargo, se puede ejecutar localmente instalando las dependencias necesarias. Al ejecutarlo se levanta un servidor que proveerá de servicios que permitirán entrenar, evaluar y clasificar timelines de usuarios Twitter según ciertos criterios definidos mediante un Random Forest. 

Los usuarios se clasifican en las siguientes 3 categorías:

**Humanos** : Cuentas de twitter correspondiente a las personas e individuos comunes, con comportamiento irregular y generadores de contenido original.

**Bots**: Cuentas donde la generación de contenido se encuentra completamente automatizada.

**Ciborgs**: Cuentas mixtas donde la publicación de contenido es compartida entre humanos y bots.

### Instalación

Es necesario contar con una Base de Datos MongoDB, la cual actuará como capa de persistencia. Para ello haremos uso del docker oficial. 

```bash
docker run -d -p 27017:27017 --name <mongo_alias> mongo
```

Si ya se tiene esta imagen en el repositorio local, se ejecutará inmediatamente una instancia del docker de mongo, en caso contrario, tomará unos minutos en realizar la descarga y posteriormente tendremos una instancia de mongo ejecutándose.

Posteriormente, descargaremos la imagen que provee del proyecto.

```bash
docker pull josemyd/apispark
```

### Quickstart

Por defecto, el servidor escuchará las peticiones por el puerto ```5433```, por lo que debemos hacer port-fowarding para poder realizar las peticiones respectivas a la instancia en ejecución. Además, debemos realizar el enlace necesario para que el proyecto sea capaz de comunicarse con la instancia de **mongo** que se está ejecutando en background.

```bash
docker run -ti --rm --link mongo:mongo -p 5433:5433 josemyd/apispark bin/spark-submit workspace/server.py
```

Una vez hecho esto, el servidor estará listo para escuchar nuestras peticiones. Dentro de la carpeta ```workspace/entrenamiento``` se proveen de 4 timelines distintos para cada categoría, además de un pequeño set de tweets spam y no spam. Igualmente, en la carpeta **evaluar** se encuentran 3 timelines distintos. Todo esto con el fin de facilitar probar las funcionalidades del **Twitter Judge**.

**Entrenar detector de SPAM**: Primero debemos entrenar el clasificador de SPAM.

```bash
curl -H "Content-Type: application/json" -X POST -d '{"spam":"/usr/spark-2.0.0/workspace/entrenamiento/spam","no_spam":"/usr/spark-2.0.0/workspace/entrenamiento/no_spam", "num_trees": 3, "max_depth": 2}' http://localhost:5433/entrenar_spam/

> {"resultado": 0.7222222222222222}
```

El campo ```resultado``` obtenido nos indicará la exactitud del modelo entrenado.

**Entrenar Juez**: Luego entrenamos el juez con los datos de ejemplo disponibles en ```workspace/entrenamiento```.

```bash
curl -H "Content-Type: application/json" -X POST -d '{"bots":"/usr/spark-2.0.0/workspace/entrenamiento/Bots","humanos":"/usr/spark-2.0.0/workspace/entrenamiento/Humanos","ciborgs":"/usr/spark-2.0.0/workspace/entrenamiento/Ciborgs", "dir_juez": "jueces/test1", "num_trees": 3, "max_depth": 2}' http://localhost:5433/entrenar_juez/

> {"matrix": [[1, 0, 0], [0, 0, 0], [0, 0, 1]], "accuracy": 1.0}
```

El campo ```matriz``` representa la matriz de confunsión generada sobre el 20% de los datos de entrenamiento. El campo ```accuracy``` indica la exactitud general del modelo entrenado.

**Evaluar**: Finalmente podemos evaluar timelines. Para este ejemplo se hace uso de los timelines en ```workspace/evaluar```.

```bash
curl -H "Content-Type: application/json" -X POST -d '{"directorio":"/usr/spark-2.0.0/workspace/evaluar/*"}' http://localhost:5433/evaluar/

> {"resultado": [[3455637141, [1.0, 0.0, 0.0]]}
```

El campo ```resultado``` contendrá una lista de los ids de usuarios evaluados junto a un arreglo que indicará su pertenencia a cada una de las categorias ([humano, bot, ciborg]).

### Referencias

[Venezolanos en Twitter: ¿Humanos, Bots o Ciborgs? ](http://concisa.net.ve/memorias/CoNCISa2016/CoNCISa2016-p057-064.pdf)
[Apache Spark](https://spark.apache.org/releases/spark-release-2-0-0.html)
[Docker](https://www.docker.com/)