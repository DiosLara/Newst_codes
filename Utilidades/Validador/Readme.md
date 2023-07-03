*** REQUISITOS ***
python 3+

# Validador
Esta clase crea objetos que contienen metodos para validar una base de datos del área de inteligencia fiscal.
El objetivo es tener la mayor calidad en los datos que tenemos, este proyecto surge por la necesidad de revisar las bases que nos llegan al área. Dentro de los principales problemas tenemos:

- Deteccion de anomalias: Tenemos la necesidad de ver un panorama general de los datos. de manera que podamos visualizar la longitud de caracteres de cada columna, principalmente en columnas que se usan como ***llaves de cruce***, ademas de visualizar la suma (promedio y otras) de los datos numericos para tenerlos de cotejo para la posteriedad, en caso de ser necesarios. 

- Validador_rfcs: Tenemos las necesidad de visualizar cada registro que no sea valido como 