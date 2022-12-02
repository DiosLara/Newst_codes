import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv


def verify_path_is_writeable(path: str):
    if os.path.isfile(path):
        print(
            f"El archivo {path} ya existe, hay que moverlo (o borrarlo) para poder volver a procesar este municipio"
        )
        raise Exception

    with open(path, "w") as file:
        file.write("Test de escritura a archivo")

    # Si no tenemos ningun error, borramos el archivo que creamos
    os.remove(path)

# def convert_curts():
    
class City:
    def __init__(self, city_file: str, base_folder:str):

        self.city_file_pols = './Municipios_procesados_1/BUSQUEDAS_BOSQUE_REAL_QUERY2.csv'
        self.file_name_pols = 'BUSQUEDAS_BOSQUE_REAL_QUERY2.csv'
        print(self.city_file_pols)

        self.base_folder = base_folder
        # print(self.file_name)
        try:
            os.mkdir(base_folder +"./Municipios_procesados_1/")
        except:
            pass
        self.main_output_file_name_pols = (base_folder +"./Municipios_procesados_1/"
            + self.file_name_pols.replace(".csv", "_procesado.txt")
        )
        self.extra_output_file_name_pols = (base_folder +
            "./Municipios_procesados_1/"
            + self.file_name_pols.replace(".csv", "_direcciones_extra.txt")
        )

 
        self.main_output = None

    # def verify_folder_structure(self):
    #     verify_path_is_writeable(self.base_folder+"/Municipios_procesados_1")

        # verify_path_is_writeable(self.extra_output_file_name)

    # def find_lista(base_folder):
    #     try:
    #         lista = pd.read_csv(base_folder + str("lista.txt"), sep=',', header=None)
    #     except: 
    #         print('El path no contiene lista')

    def load_data_pols(self):
        GL_P1 = pd.read_csv(self.city_file_pols, sep=",", encoding="utf-8-sig",low_memory=False)
        # GL_P1.rename(columns={'DIRECCION':'DIRECCION_CATS'}, inplace=True)
        GL_P1["GEO_GEOMETRY_1"] = GL_P1["geometry"].str.replace(
            "POINT", "", regex=False
        )
        GL_P1["GEO_GEOMETRY_1"] = GL_P1["GEO_GEOMETRY_1"].str.replace(
            ")", "", regex=False
        )
        GL_P1["GEO_GEOMETRY_1"] = GL_P1["GEO_GEOMETRY_1"].str.replace(
            "(", "", regex=False
        )
        GL_P1["GEO_GEOMETRY_1"] = GL_P1["GEO_GEOMETRY_1"].str.replace(
            ",", "", regex=False
        )
        n = (
            GL_P1["GEO_GEOMETRY_1"]
            .str.split(" ", expand=True)
            .rename(columns={1: "lon", 2: "lat"})
        )
        GL_S = pd.concat([GL_P1, n], axis=1)

        # Redondeamos a 4 decimales, que es una precision de ~10 m para
        # reducir el numero de busquedas que tenemos que hacer
        GL_S["lon"] = GL_S["lon"].astype(float)
        GL_S["lat"] = GL_S["lat"].astype(float)
        GL_S["latlong"] = GL_S["lat"].astype(str) + ", " + GL_S["lon"].astype(str)

        self.clean_df = GL_S
    def load_data_query(self):
        GL_Q = pd.read_csv(self.city_file_query, sep="\t", encoding="utf-8-sig")
        self.clean_df =GL_Q

    def get_unique_latlongs(self, test=False):
        unique_latlongs = pd.DataFrame({"latlong": self.clean_df["latlong"].unique()})

        if test:
            unique_latlongs = unique_latlongs.head(20)
            # print(unique_latlongs)

        # unique_latlongs["DIRECCION"] = np.nan

        return unique_latlongs
    def get_unique_doms(self, test=False):
        unique_doms = pd.DataFrame({"FULL_DOM": self.clean_df["FULL_DOM"].unique()})

        if test:
            unique_latlongs = unique_doms.head(20)

        # unique_doms["latlongs"] = np.nan

        return unique_doms

    def is_missing_addresses(self, unique_latlongs: pd.DataFrame) -> bool:
        """
        Aplicamos una funcion que calcula el largo de cada campo en direcciones 
        y verifica que no sea 0 cuando la suma de estos valores sea mayor a 0, 
        sabemos que nos faltan direcciones por procesar
        """
        return (
            sum(
                unique_latlongs["DIRECCIONES"][
                    unique_latlongs["DIRECCIONES"] is not None
                ].apply(lambda addresses: len(addresses) == 0)
            )
            == 0
        )

    def save_main_output(self, base_folder):
        all_addresses = []
        self.write_to_txt(self.main_output, base_folder +"./Municipios_procesados_1/"+ self.file_name_pols.replace(".csv", "_procesado_test.txt"))
       
        for index, row in self.main_output.iterrows():
            if row["RAW_ADDRESS"] is not np.nan:

                address_dict = row["RAW_ADDRESS"]
                try:
                    expanded_dict = pd.json_normalize(address_dict, record_prefix=True, sep=".")
                    expanded_dict["original_index"] = index
                    all_addresses.append(expanded_dict)
                except:
                    all_addresses.append(pd.DataFrame())
                    all_addresses["original_index"] = index
            else:
                all_addresses.append(pd.DataFrame())
                
        all_expanded = pd.concat(all_addresses, axis=0)
        self.main_output = self.main_output.merge(all_expanded, how="left", left_index = True, right_on="original_index").reset_index(drop=True)
        # self.main_output.drop(columns=["index", "RAW_ADDRESS", "original_index"])

        self.write_to_txt(self.main_output, self.main_output_file_name_pols)

    def columna_key(base):
        ind= base[['key']]
        base= base.drop(columns=ind.columns)
        base.insert(loc=0, column=ind.columns[0], value= ind[ind.columns[0]])
        return(base)
    def save_extra_addresses(self, df: pd.DataFrame):
        all_addresses = []
        
        try:
            for _, row in df.iterrows():
                if row["JSONS_EXTRAS"] is not np.nan:
                    list_of_addresses = row["JSONS_EXTRAS"]
                    
                    for address in list_of_addresses:
                        
                        
                        expanded_dict = pd.json_normalize(address, record_prefix=True, sep=".")
                        expanded_dict["latlong"] = row["latlong"]
                        all_addresses.append(expanded_dict)
                else:
                    all_addresses.append(pd.DataFrame())
        except:
            combination= pd.DataFrame()
            pass
                
        combination = pd.concat(all_addresses, axis=0)
        df.merge(combination, how="left", left_on="latlong", right_on="latlong")

        self.write_to_txt(df, self.extra_output_file_name_pols)

    def write_to_txt(self, df: pd.DataFrame, path: str):
        print(path)
        df.to_csv(
            path, sep="\t", index=False, encoding="utf-8-sig",
        )
