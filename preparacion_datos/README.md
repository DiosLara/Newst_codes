# Cruces necesarios para preparar las bases
El orden en que deben usar los notebooks son:

1) cruce_curtxcasas; Esto es para obtener los CURTS que ya tenemos y rescatarlos, ademas de incluir otro cruce con la base de industrias para tenerlas identificarlas.

2) Cruces_limpieza_casas; Hace una serie de cruces para identificar zonas de interes, como ejes, areas verdes, etc.

3) Cruces_limpieza_terrenos; Una similitud con Cruces_limpieza_casas solo que se aplica a la deteccion de yolo que es de terrenos 

4) Hacer predial, el codigo aun se esta mejorando; Se cruza CURT x mzns obteniendo los deudores y pagadores de un año determinado.

Al final de aplicar estos notebooks obtendremos 2 shapes. Uno de casas y otro de terrenos. Para posteriormente aplicar el valor Z, y estara listo para obtener el valor catastral
