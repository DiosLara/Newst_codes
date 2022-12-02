import pandas as pd
import numpy as np
import logging
import glob
from tqdm import tqdm
import os
import concurrent.futures
from dotenv import load_dotenv
from geoloc import Geocoder
from city import City

logging.basicConfig(
    filename="geolocalizacion.log",
    filemode="w",
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def task_chunks(unique_latlongs):
    # Ejecutamos la query de direcciones
    unique_latlongs["DIRECCIONES"] = geocoder.query_google(
        unique_latlongs["latlong"],
        "Municipios_procesados_1/" + 'BR' + "_partial.txt",
    )
    
    unique_latlongs["DIRECCIONES"] = unique_latlongs["DIRECCIONES"].where(
        pd.notnull(unique_latlongs["DIRECCIONES"]), None
    )

    unique_latlongs["DIRECCIONES"] = unique_latlongs["DIRECCIONES"].apply(
        lambda addresses: addresses if addresses is not None else []
    )
    
    
    unique_latlongs["DIRECCION"] = unique_latlongs["DIRECCIONES"].apply(
        lambda directions_list: directions_list[0]
        if len(directions_list) > 0
        else np.nan
    )
    
    
    unique_latlongs["JSONS_EXTRAS"] = unique_latlongs["DIRECCIONES"].apply(
        lambda directions_list: [
            direction.raw if direction is not np.nan else direction
            for direction in directions_list
        ]
    )

    return unique_latlongs

if __name__ == "__main__":
    # Encontrar todos los archivos para convertir
    # load_dotenv(dotenv_path='E:/dsanchez/padagua_2021/env')
    # BASE_FOLDER = os.getenv("BASE_FOLDER")
    # BASE_CORR = BASE_FOLDER + "/BASE_CORRECCIONES/"
    base_folder = r'C:\Users\dlara\padagua_2021\Geolocalizacion\reverse_geocode\src'
    # # print(base_folder)
    cities_files = glob.glob(r"C:\Users\dlara\padagua_2021\Geolocalizacion\reverse_geocode\src\Municipios_proceso_1")

    # print(cities_files)

    # Si no existe el directorio para guardar los resultados, creamos uno
    # try:
    #     os.mkdir("/Municipios_proceso_1/")
    # except: 
    #     pass
    # try :
    #     os.mkdir("C:/Users/dlara/Documents/Geometria_Prueba/Municipios_procesado_1/")
    # except: 
    #     pass

    # Instanciamos el geocoder que usaremos para todas las requests
    
    lista_apis = "C:/Users/dlara/lista.txt" ## Ruta de la lista de Apis
    geocoder = Geocoder(lista_apis)

    for city_file in cities_files:
        # print(city_file)
        # Try catch en cada uno aplicando nuestra funcion de lat, long -> direccion
        print(city_file)
        city = City(city_file, base_folder)
        city.load_data_pols()
        unique_latlongs = city.get_unique_latlongs(test=False)

        # Ejecutamos la query de direcciones

        # "Municipios_procesados_1/" + 'Query_puntos_ecatepec_2' + "_partial.txt" para partial
        
        ## Aqui mandas a llamar el pool
        unique_latlongs_chunks =  np.array_split(unique_latlongs, os.cpu_count() - 1) ##Aqui se especifica si se requiere un loc y los chunks

        df_concat = pd.DataFrame()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for df_chunk in tqdm(zip(executor.map(task_chunks, unique_latlongs_chunks)), total = len(unique_latlongs_chunks)):
                df_concat = pd.concat([df_chunk[0],df_concat], axis=0)

        # Hacemos el join con el dataframe existente
        GL_S = city.clean_df.merge(
            df_concat[["latlong", "DIRECCION"]], on="latlong", how="left"
        )
        # print(GL_S.head())
        # Escribir resultado a otro txt
        logging.info("Guardando nuevo archivo txt")
        # print(GL_S.columns)
        GL_S["DIRECCION"] = GL_S["DIRECCION"]

        GL_S["RAW_ADDRESS"] = GL_S["DIRECCION"].apply(
            lambda address: address.raw if address is not np.nan else address
        )

        city.main_output = GL_S

        city.save_main_output(base_folder)
        city.save_extra_addresses(df_concat)


