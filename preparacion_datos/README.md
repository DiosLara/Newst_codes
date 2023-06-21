# Cruces necesarios para preparar las bases
El orden en que deben usar los notebooks son:

1) Hacer crucesxpredial, se cruza CURT x predial obteniendo los montos correspondientes al año de interés.

2) Cruces_limpieza_casas; Hace una serie de cruces para identificar y obtener los CURT, industrias y zonas de interes, como ejes, areas verdes, etc.

3) Cruces_limpieza_terrenos; Una similitud con Cruces_limpieza_casas solo que se aplica a la detección de yolo que es de terrenos.

4) Limpieza_bases; Sirve para limpiar las columnas que se dejarán en la base de entrega final.



Al final de aplicar estos notebooks obtendremos 2 shapes. Uno de casas y otro de terrenos. Para posteriormente aplicar el valor Z, y estará listo para obtener el valor catastral
