# Motor Clasificador de Usuarios de Twitter (humanos, bots, ciborgs)

## Tabla de contenido

* [Introducción](#introducción)
* [Instalación](#instalacion)
* [Quick Start](#quickstart)
* [Referencias](#referencias)

### Introducción
Este es un proyecto desarrollado con Apache Spark en su versión 2.0 y su respectiva libreria para Machine Learning. Todo el codigo se encuentra en Python. El proyecto se encuentra dockerizado para facilitar su despliegue y utilización. Al ejecutarlo se levanta un servidor que proveerá de servicios que permitiran entrenar, evaluar y clasificar timelines de usuarios Twitter según ciertos criterios definidos mediante un Random Forest en humanos, bots o ciborgs. 

### Instalación

Es necesario contar con una Base de Datos MongoDB, la cual actuará como capa de persistencia. Para ello haremos uso del docker oficial. 

```bash
docker run -p 27017:27017 --name <mongo_alias> mongo
```

### Quick Start

### Referencias