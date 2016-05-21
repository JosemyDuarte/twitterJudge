import time, sys, cherrypy, os
from paste.translogger import TransLogger
from app import create_app
from pyspark import SparkContext
from pyspark.conf import SparkConf

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def init_spark_context():
    conf = SparkConf()
    conf.setAppName('ExtraerCaracteristicas')

    _sc = SparkContext(conf=conf, pyFiles=['/home/jduarte/Workspace/TesisSpark/engine.py',
                                          '/home/jduarte/Workspace/TesisSpark/app.py',
                                          '/home/jduarte/Workspace/TesisSpark/tools.py'])
    return _sc


def run_server(app):
    # Enable WSGI access logging via Paste
    app_logged = TransLogger(app)

    # Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app_logged, '/')

    # Set the configuration of the web server
    cherrypy.config.update({
        'engine.autoreload.on': True,
        'log.screen': True,
        'server.socket_port': 5432,
        'server.socket_host': '0.0.0.0'
    })

    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()


if __name__ == "__main__":
    # Init spark context and load libraries
    sc = init_spark_context()
    app = create_app(sc)
    # start web server
    run_server(app)
