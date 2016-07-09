import ConfigParser

import cherrypy
import os
from paste.translogger import TransLogger

from app import create_app

os.chdir(os.path.dirname(os.path.abspath(__file__)))
configParser = ConfigParser.RawConfigParser()
configParser.read("config.ini")


def run_server(app):
    # Enable WSGI access logging via Paste
    app_logged = TransLogger(app)

    # Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app_logged, '/')

    # Set the configuration of the web server
    cherrypy.config.update({
        'engine.autoreload.on': True,
        'log.screen': True,
        'server.socket_port': int(configParser.get("server", "port")),
        'server.socket_host': configParser.get("server", "host")
    })

    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()


if __name__ == "__main__":
    # Init spark context and load libraries
    app = create_app()
    # start web server
    run_server(app)
