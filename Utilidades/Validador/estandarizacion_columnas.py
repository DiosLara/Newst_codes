import datetime
import re
import pandas as pd
numero=re.compile("\d\,\d|\d\.\d|\d")
fecha=re.compile("^(?P<day>\d\d?)[\-|\/|\s](?P<month>\d\d?)[\-|\/|\s](?P<year>\d\d\d\d)|^(?P<year1>\d\d\d\d)[\-|\/|\s](?P<month1>\d\d?)[\-|\/|\s](?P<day1>\d\d?)")
def corregir_tipo_dato(df:pd.DataFrame,corregir=False)->pd.DataFrame:
    """Funcion cambia los tipos de datos al dato identificado automaticamente y retorna el dataframe"""
    datos=pd.DataFrame(df.dtypes,columns=["Anterior"])
    validador_date=0
    for colum in df.columns:
        if str(datos.loc[colum,"Anterior"])=="object":
            valores=df[colum].unique()
            a=0
            df[colum]=df[colum].astype("str")
            for val in valores[:100]:
                if val=="":
                    print(colum+" con valores nulos")
                try:
                    m=fecha.match(val)
                    if m.group("year"):
                        date=datetime.date(int(m.group("year")),int(m.group("month")),int(m.group("day")))
                    else:
                        date=datetime.date(int(m.group("year1")),int(m.group("month1")),int(m.group("day1")))
                    df.loc[df[colum]==val,colum]=date
                    validador_date=1
                except:
                    validador_date=0
                    try:
                        df[colum]=df[colum].astype("int64")
                    except:
                        try:
                            validador_numero=numero.findall(val)
                            if len(validador_numero)>0:
                                a+=1
                            else:
#                                 print(val)
                                if a>0:
                                    if corregir:
                                        print("Columna: "+colum+" Tiene dato invalido: "+val+" corregido a 0")
                                        df.loc[df[colum]==val,colum]="0"
                                    else:
                                        print("Columna: "+colum+" Tiene dato invalido: "+val)
                        except:
                            pass


            if a>0:
                df[colum]=df[colum].str.replace(",","")
                try:
                    df=df.astype({colum:"float64"})
                except:
                    pass
            if validador_date==1:
                df=df.astype({colum:"datetime64[ns]"})
    datos=pd.concat([datos,pd.DataFrame(df.dtypes,columns=["Nueva"])],axis=1)
    print(datos)
    return df