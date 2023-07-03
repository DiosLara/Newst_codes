import sqlite3
import pandas as pd
from tkinter import filedialog
from tkinter import *

def read_sql(sqlpathfile,sqltablename):
    """Funcion para leer una base sql mas simple
    argumentos:
        sqlpathfile: ruta archivo sqlite3
        sqltablename: nombre de la tabla dentro del sql Nota: Puede incluir query despues del nombre del tabla, 
        ejemplo:
        sqltablename="CFDI where RFC like 'XAXX%'"
    output:
    dataframe"""
    con=sqlite3.connect(sqlpathfile)
    base=pd.read_sql_query("Select * From "+sqltablename,con)
    con.close()
    return base

def read(filename,encoding="UTF-8-sig",separador=","):
    '''Funcion para leer dataframe en cualquier formato excel, csv o txt,
    argumentos:
        filename: path del archivo
        encoding="UTF-8-sig" asigna la codificacion de lectura para archivos txt o csv
        separador="," asigna el elemento separador de cadenas para archivos txt o csv
    output:
        Dataframe'''
    file=filename.split(".")
    ext=file[1]
    file=file[0]
    print(filename)
    df=[]
    df1=[]
    if ext=="csv":
        return pd.read_csv(filename,encoding=encoding)

    elif ext=="xlsx" or ext=="xlsb" or ext=="xls" or ext=="xlsm":
        xls=pd.ExcelFile(filename)
        nam=xls.sheet_names
        texto=[]
        df=pd.DataFrame()
        if len(nam)==1:
            df=pd.read_excel(filename)
            df["hoja"]=nam[0]
        else:
            for nu,name in enumerate(nam):
                texto.append(str(nu)+": "+ name)
            seleccion=int(input("Seleciona el indice de la hoja "+", ".join(texto)+", "+str(nu+1)+": Todas "))
            i=1
            if seleccion>=len(nam):
                for i in range(len(nam)):
                    df1=pd.read_excel(filename,sheet_name=nam[i])
                    df1["hoja"]=nam[i]
                    df=pd.concat([df,df1],ignore_index=True)
                    i+=1
            else:
                df=pd.read_excel(filename,sheet_name=nam[seleccion])
                df["hoja"]=nam[seleccion]
        return pd.DataFrame(df,columns=df.columns)

    elif ext=="txt":
        with open(filename,encoding=encoding) as f:
            lines = f.readlines()
        df=[]
        for line in lines:
            line=line.split("\n")
            line=line[0]
            df.append(line.split(separador))
        df=pd.DataFrame(df)
        df.columns=df.iloc[0]
        df=df[1:]
        try:
            df=df.drop(columns=[None])
        except:
            pass
        return pd.DataFrame(df,columns=df.columns)
base1=""
base2=""
def read_dfs():
    """La Funcion permite mandar a llamar 2 bases de datos mediante una interfaz grafica
    output: vector con la ubicacion absoluta del ambas bases"""
    root = Tk() 
    def resquestbase1():
        global base1
        root.filename1 =  filedialog.askopenfilename(initialdir = "Downloads",title = "Select file",filetypes = (("Excel files",["*.xlsx","*.csv","*.xlsb","*.xls","*.txt"]),("all files","*.*")))
        base1=root.filename1
        b1["text"]=base1
        return base1
    def resquestbase2():
        global base2
        root.filename2 =  filedialog.askopenfilename(initialdir = "Downloads",title = "Select file",filetypes = (("Excel files",["*.xlsx","*.csv","*.xlsb","*.xls","*.txt"]),("all files","*.*")))
        base2=root.filename2
        b2["text"]=base2
        return base2
    b1=Button(root, text="base1", command= lambda:base1==resquestbase1())
    b1.pack()
    b2=Button(root, text="base2", command=lambda:base2==resquestbase2())
    b2.pack()
    b3=Button(root, text="Salir/Aceptar", command=root.destroy)
    b3.pack()
    root.mainloop() 
    return base1, base2

    
if __name__ == '__main__':
    base1,base2=read_dfs()
    base1=read(base1)
    base2=read(base2)

