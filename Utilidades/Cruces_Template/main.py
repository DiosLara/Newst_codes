import pandas as pd
import Preproceso
import abrir_archivos
import warnings
import Postproceso
import filtros
import cruces
import os 
warnings.filterwarnings('ignore')
class Cruce_preparacion:

    '''def __init__(self, base1:pd.DataFrame, base2:pd.DataFrame, nombre1:str='Base_1', nombre2:str='Base_2'):
        self.df1 = base1
        self.df2 = base2
        self.nombre1 = nombre1
        self.nombre2 = nombre2'''

    def __init__(self):
        ruta1, ruta2 = abrir_archivos.read_dfs()
        self.df1 = abrir_archivos.read(ruta1 )
        self.df2 = abrir_archivos.read(ruta2)
        self.nombre1 = 'Base_1'
        self.nombre2 = 'Base_2'
        self.run = 0
        


    def Preproceso_Cruce(self, 
                        otros:list=None,
                        columns_extra_agg1:list=[],
                        columns_extra_agg2:list=[], 
                        tipo_agg:str='Max' ):
        '''
        (Method)
            Este metodo hace un preproceso a cada base, para revisar los siguientes puntos
            * Revisa el tipo de variable asociado a cada columna
            * Quita duplicados en caso de tener columnas con el mismo nombre
            * Quitar acentos de las columnas que tienen razon social, ademas de una limpeza de espacios final e inicial. Hace mayusculas todo el texto
            * Revisa que no haya duplicados en la(s) columnas key
            * 
        (Parameters)
            - col_nombre1: Nombre de la columna que tiene la razon social en la base1
            - col_key1: Nombre de la columna que fungira como llave en la base1
            - col_nombre2: Nombre de la columna que tiene la razon social en la base2, en caso de no especificar, se tomara la misma que base1
            - col_key2: Nombre de la columna que fungira como llave en la base2, en caso de no especificar, se tomara la misma que base1
            - otros:  En caso de querer quitar otros tipos de caracteres aparte de los que se mencionan, aqui los agrega en una lista.
            - columns_extra_agg1: Una lista de columnas extra que se requieren para una agrupación, puesto que llega a darse el caso de que 
                                la columna key, tiene duplicados, tenemos que agrupar, para que el cruce se bueno, en tales caso se agrupara 
                                por la columna nombre y la columna key, pero en caso de querer otra, anexe aqui. Concerniente a la base1
            - columns_extra_agg2 : Una lista de columnas extra que se requieren para una agrupación, puesto que llega a darse el caso de que 
                                la columna key, tiene duplicados, tenemos que agrupar, para que el cruce se bueno, en tales caso se agrupara 
                                por la columna nombre y la columna key, pero en caso de querer otra, anexe aqui. Concerniente a la base2
            - tipo_agg: Determina el tipo de agrupacion que desea para ambas bases, puede indicar: max, min, sum, y mean

        '''
        if self.run == 0:
            # Buscamos las columnas protagonistas asi como generar la column key
            self.df1, self.__col_nombre1, self.__col_key1 = Preproceso.seleccion_cols_principales(self.df1)
            self.df2, self.__col_nombre2, self.__col_key2 = Preproceso.seleccion_cols_principales(self.df2)
            os.system('cls')
            
            # Corregimos el tipo de variable para cada DataFrame
            print('Actualizacion de tipo de variable por campo para ', self.nombre1)
            self.df1 = Preproceso.corregir_tipo_dato(self.df1)
            print('Actualizacion de tipo de variable por campo para ', self.nombre2)
            self.df2 = Preproceso.corregir_tipo_dato(self.df2)
            os.system('cls')

            # Filtramos en caso de ser necesario
            ans = input('Desea filtrar la Base 1?: ')
            if ans.lower() in ['si', 'yes', 'claro', '1', 'sip', 'sipi', 'afirmativo', 'por supesto', 'ok', 'porsupollo']:
                self.df1 = filtros.filtrar_df(self.df1,vup=15)
            ans = input('Desea filtrar la Base 2?: ')
            if ans.lower() in ['si', 'yes', 'claro', '1', 'sip', 'sipi', 'afirmativo', 'por supesto', 'ok','porsupollo']:
                self.df2 = filtros.filtrar_df(self.df2,vup=15)
            os.system('cls')

            # Quitamos columnas duplicadas
            self.df1 = Preproceso.quitar_duplicados_x_columnas(self.df1)
            self.df2 = Preproceso.quitar_duplicados_x_columnas(self.df2)
            print('Paso quitar duplicados')

            # Estandarizamos Columna referente a Razon Social
            self.df1 = Preproceso.Estandarisacion_str_col(self.df1, columns=self.__col_nombre1 , otros= otros)
            self.df2 = Preproceso.Estandarisacion_str_col(self.df2, columns=self.__col_nombre2 , otros= otros)
            print('Paso estandarizacion')

            
            # Creamos campo Full name
            self.df1['Full name'] = self.df1[self.__col_nombre1]
            self.df2['Full name'] = self.df2[self.__col_nombre2]
            print('Paso generar Full name', self.df1.shape)

        # Preprocesing para cruce.
        print(self.__col_key1, self.__col_nombre1)
        self.df1, self.df_defect1 = Preproceso.preprocesing(self.df1, col_key=['key',self.__col_key1], col_nombre=self.__col_nombre1, 
                                                            cols_agrupacion=columns_extra_agg1,
                                                            tipo_agrupacion=tipo_agg )
                                                        
        self.df2, self.df_defect2 = Preproceso.preprocesing(self.df2,col_key=['key',self.__col_key2],col_nombre=self.__col_nombre2, 
                                                            cols_agrupacion=columns_extra_agg2,
                                                            tipo_agrupacion=tipo_agg )
        self.run += 1

    def Cruce(self, tipo_cruce:str):
        '''
        (Method)
            Este metod realiza un cruce de ambas bases, se debe especificar el tipo de cruce.
        (Parameters)
            - tipo_cruce: Indica el tipo de cruce que desea, puede ser: "left", "rigth", "inner", "outer"

        '''
        if self.run >1:
            # Creamos campo Full name
            self.df1['Full name'] = self.df1[self.__col_nombre1]
            self.df2['Full name'] = self.df2[self.__col_nombre2]

        # Borramos el campo "_merge" en caso de que exista
        try:
            self.df1.pop('_merge')
        except:
            pass # No tienen el campo
        try:
            self.df2.pop('_merge')
        except:
            pass # No tienen el campo

        # Preguntamos si queremos mas de una llave para cruce.
        ans = input('Ha ingresado una columna como llave, desea hacer el cruce solo por esa llave o agregamos la Razon Social como llave de cruce?  ')
        if ans.lower() in ['si', 'yes','afirmativo','por supesto', 'claro', '1', 'sipi']:
            key_cruce = ['key', 'Full name']
        else:
            key_cruce = ['key']

        # Realizamos el cruce de los validos
        self.__df_cruce_val = cruces.cruce_instantaneo(self.df1, self.df2, 
                                                nombre1=self.__col_nombre1, nombre2=self.__col_nombre2,
                                                tipo_cruce=tipo_cruce, key_cruce=key_cruce)
        # Ahora realizamos el cruce de los defectuosos
        if self.df_defect1.shape[0]>0 & self.df_defect2.shape[0]>0:
            self.__df_cruce_defect = cruces.cruce_instantaneo(self.df_defect1, self.df_defect2, 
                                                    nombre1=self.__col_nombre1, nombre2=self.__col_nombre2,
                                                    tipo_cruce=tipo_cruce, key_cruce=key_cruce)
            self.__df1 = pd.concat([self.df1, self.df_defect1])
            self.__df2 = pd.concat([self.df2, self.df_defect2])

        elif self.df_defect1.shape[0]>0:
            self.__df_cruce_defect = self.df_defect1.copy()
            self.__df_cruce_defect['_merge'] = 'left_only'
            self.__df1 = pd.concat([self.df1, self.df_defect1])
            self.__df2 = self.df2
        
        elif self.df_defect2.shape[0]>0:
            self.__df_cruce_defect = self.df_defect2.copy()
            self.__df_cruce_defect['_merge'] = 'right_only'
            self.__df2 = pd.concat([self.df2, self.df_defect2])
            self.__df1 = self.df1
        else:
            self.__df_cruce_defect = pd.DataFrame()
            self.__df1 = self.df1
            self.__df2 = self.df2


        # Concatenamos los cruces
        self.df_cruce = pd.concat([self.__df_cruce_val, self.__df_cruce_defect])
        self.df_cruce.reset_index(drop=True, inplace=True)
                                            

    def PostProceso_cruce(self, revision_numeralia:bool=True):
        '''
        (Method)
            Este metodo revisa que las numeralias cuadren tomando el cruce y los elementos del mismo. 
            Ademas verifca que y repara (en caso de poderse) aquellas claves que tengan un posible error de dedo lo que 
            halla hecho que le cruce saliera incompleto.
        (Parameters)

        '''
        self.run += 1
        # Se imprime la numeralia
        if revision_numeralia:
            # Buscar las columnas numericas
            cols_numericas1 = Postproceso.Buscar_cols_numericas(self.df1)
            cols_numericas2 = Postproceso.Buscar_cols_numericas(self.df2)

            # Imprimir resultados de numeralia
            Postproceso.verificaciones_sumas(self.__df1, self.__df2, self.df_cruce, 
                                            col_verificaciones_cfdi=cols_numericas1,
                                            col_verificaciones_issemym=cols_numericas2 )

        # Revisamos duplicados por llave en el cruce.
        key_cruce_duplicada = self.df_cruce[self.df_cruce['key'].duplicated()]['key']
        if len(key_cruce_duplicada)>0:
            #cruce0_key_repetida = self.df_cruce[((self.df_cruce[self.col_key].isin(key_cruce_duplicada)))].sort_values('key')
            indicadora = True
            while indicadora:
                ans = input(f'''Tenemos keys duplicadas debido a que el nombre es diferente, sobre que base quieres que se hagan los cambios:
                \n 0: {self.nombre1}
                \n 1: {self.nombre2}''')
                if ans in ['0', self.__col_nombre1]:
                    self.df1 = Postproceso.solucion_key_duplicada(self.df_cruce, self.df1,
                                                                col_key='key',col_nombreissemym=self.__col_nombre2, 
                                                                col_nombrerazonsocial=self.__col_nombre1)
                    indicadora = False
                elif ans in ['1', self.__col_nombre2]:
                    self.df2 = Postproceso.solucion_key_duplicada(self.df_cruce, self.df2,
                                                                col_key='key',col_nombreissemym=self.__col_nombre2, 
                                                                col_nombrerazonsocial=self.__col_nombre1)
                    indicadora = False
                else:
                    print('Ingreso un formato no valido')
            
        # Revisamos duplicados por nombre cuya llave pueda tener un error de dedo.
        rs_cruce_duplicada = self.df_cruce[self.df_cruce['Full name'].duplicated()]['Full name'] 
        if len(rs_cruce_duplicada) >0:
            indicadora = True
            while indicadora:
                ans = input(f'''Tenemos Razones Sociales duplicadas debido a que la key es diferente, sobre que base quieres que se hagan los cambios:
                \n 0: {self.nombre1}
                \n 1: {self.nombre2}''')
                if ans in ['0', self.__col_nombre1]:
                    self.df1 = Postproceso.solucion_nombre_duplicado(self.df_cruce, self.df1)
                    indicadora = False
                elif ans in ['1', self.__col_nombre2]:
                    self.df2 = Postproceso.solucion_nombre_duplicado(self.df_cruce, self.df2)
                    indicadora = False
                else:
                    print('Ingreso un formato no valido')

        
                


        