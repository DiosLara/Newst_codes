import os
import requests
import json
import logging
from tqdm import tqdm
from random import shuffle
from typing import Optional
from functools import partial

import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim, GoogleV3
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderQueryError
from geopy import distance


def havershine(base):
    distancias_resultantes = [0]
    for i in range(0, len(base.index)):
        try:
            distancia = distance.distance(
                base["latlong"][i], base["latlong"][i + 1]
            ).meters
            distancias_resultantes.append(distancia)
        except:
            pass

    return distancias_resultantes


class Geocoder:
    def __init__(self, key_files_path: str):
        self.logger = logging.getLogger()
        # Lista de las posibles api_keys que puede usar
        
        with open(key_files_path, "r") as key_file:
            self.api_keys = key_file.read().splitlines()

        shuffle(self.api_keys)

        self.queries_count = [0 for _ in self.api_keys]
        # Api key en uso
        self.api_keys_index = 0
        self.api_key = self.api_keys[self.api_keys_index]

        # Iniicializamos los diferentes locators
        self.open_maps_locator = Nominatim(user_agent="openmapquest")
        self.google_locator = GoogleV3(api_key=self.api_key)

    def next_api_key(self) -> Optional[str]:
        if len(self.api_keys) > self.api_keys_index + 1:
            self.api_keys_index += 1
            self.api_key = self.api_keys[self.api_keys_index]
            self.google_locator = GoogleV3(api_key=self.api_key)

            return self.api_key

        return None

    def query_nominatim(self, df: pd.DataFrame, file_name: Optional[str] = None):
        return self.query_geocoder(self.open_maps_locator, df, file_name)

    def query_google(self, df: pd.DataFrame, file_name: Optional[str] = None):
        return self.query_geocoder(self.google_locator, df, file_name)

    def save_partial_results(self, df: pd.DataFrame, file_name: str):
        if os.path.isfile(file_name):
            os.remove(file_name)
            

        df.to_csv(
            file_name, sep="\t", index=False, encoding="utf-8-sig",
        )

    def query_google_direct(self, direcciones: pd.Series):
        """
        Esta función toma como argumento una columna de direcciones dirección, en formato string, 
        y hace una query a google para regresar un lat long correspondiente.
        """
        latlongs = []
        for address in direcciones:
            if self.api_key is not None:
                try:
                    result = self.google_locator.geocode(address, exactly_one=False)
                   
                    latlongs.append(result)
                except GeocoderQueryError:
                    self.logger.info(
                        f"Cambiando llaves de api, la llave {self.api_key} dio un error"
                    )
                    self.next_api_key()
                except Exception as error:
                    self.logger.exception(
                        f"Error inesperado al hacer llamadas de api con llave {self.api_key}"
                    )
                    self.next_api_key()

        return latlongs

    def query_geocoder(
        self, locator, latlongs: pd.Series, file_name: Optional[str] = None
    ) -> Optional[pd.Series]:
        rate_limited_locator = RateLimiter(
            locator.reverse,
            min_delay_seconds=0.1,
            max_retries=2,
            swallow_exceptions=False,
            return_value_on_exception=None,
        )

        
        addresses = []
        n = 1000
        for _, partial_latlongs in latlongs.groupby(np.arange(len(latlongs)) // n):
            if self.api_key is not None:
                try:
                    tqdm.pandas()
                    partial_addresses = partial_latlongs.progress_apply(
                        partial(
                            rate_limited_locator, language="spanish", exactly_one=False
                        )
                    )

                    addresses.append(partial_addresses)
                    # if file_name:
                    #     self.save_partial_results(pd.concat(addresses), file_name)

                except GeocoderQueryError:
                    self.logger.info(
                        f"Cambiando llaves de api, la llave {self.api_key} dio un error"
                    )
                    self.next_api_key()
                except Exception as error:
                    self.logger.exception(
                        f"Error inesperado al hacer llamadas de api con llave {self.api_key}"
                    )
                    self.next_api_key()

        return pd.concat(addresses) if len(addresses) > 0 else None

    def query_nearby_api(self, latlong: str, radius):
        url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={latlong}&radius={radius}&key={self.api_key}"

        response = requests.get(url)
        self.queries_count[self.api_keys_index] += 1
        response_text = json.loads(response.text)

        return response_text

    def query_geocode_api(self, latlong: str):
        try:
            url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latlong}&key={self.api_key}"
            response = requests.get(url)
        except ConnectionError:
            _ = self.next_api_key(self)
            url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latlong}&key={self.api_key}"
            response = requests.get(url)

        self.queries_count[self.api_keys_index] += 1
        response_text = json.loads(response.text)
        print(response_text)
        return response_text

    def is_google_api_valid(self, latlong: str = "19.1656, -99.5695") -> bool:
        try:
            url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latlong}&key={self.api_key}"
            _ = requests.get(url)

            return True
        except ConnectionError:

            return False


