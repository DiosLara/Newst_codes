import os
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import cx_Oracle
from typing import Optional

from PipelinesMunicipios.cache import Cache


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class OracleDB(metaclass=SingletonMeta):
    """
    Clase para conexión a base de datos en Oracle. Singleton.

    Requisitos para su uso:
        - Tener un archivo .env con los datos de conexión en el path que está usando python
        - Tener la VPN activa
        - Descargar el cliente de Oracle (https://www.oracle.com/database/technologies/instant-client/winx64-64-downloads.html) y agregar su path al .env

    Para ejecutar una query:
        >>> from OracleIntfis.db_connection import OracleDB
        >>> db = OracleDB()
        >>> df = db.test_query()
    """

    def __init__(self, dotenv_path: Optional[str] = None, cache: bool = True):
        load_dotenv(dotenv_path)

        cx_Oracle.init_oracle_client(lib_dir=os.getenv("ORACLE_CLIENT_PATH"))

        self.cache = Cache("./") if cache else None

        self.connect_to_db()

    def connect_to_db(self):
        # Datos para la conexión
        # Deben estar en un archivo .env en el mismo directorio
        HOST = os.getenv("HOST")
        PORT = os.getenv("PORT")
        SERVICE_NAME = os.getenv("SERVICE_NAME")
        USER = os.getenv("USER")
        PASSWORD = os.getenv("PWD")

        preq = cx_Oracle.makedsn(HOST, PORT, service_name=SERVICE_NAME)
        con_string = f"oracle://{USER}:{PASSWORD}@{preq}"

        self.engine = create_engine(
            con_string, max_identifier_length=128, convert_unicode=False
        )

    def query_to_df(self, query) -> pd.DataFrame:
        if self.cache:
            df = self.cache.load(query)
            if isinstance(df, pd.DataFrame):
                return df

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
            df.columns = df.columns.str.upper()

        if self.cache:
            self.cache.store(query, df)
        self.close()

        return df

    def read_table(self, table_name) -> pd.DataFrame:
        query = "SELECT * FROM " + table_name

        return self.query_to_df(query)

    def test_query(self) -> pd.DataFrame:
        query = "SELECT b.PARAMETER, A.VALUE SESION, B.VALUE BASE, C.VALUE INSTANCIA \
            FROM nls_database_parameters B, nls_session_parameters A, \
            nls_instance_parameters C WHERE b.PARAMETER = a.PARAMETER(+) \
            AND b.PARAMETER = C.PARAMETER(+)"

        return self.query_to_df(query)
