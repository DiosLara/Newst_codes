def depurar_gabinete(tab_f, geom_clas8, df_final_c):
    pdf2= pd.DataFrame()
    
    for cve in tab_f['CVEGEO_MANZANA'].unique():
        pdf=pd.DataFrame()
        # print(cve)
        df=tab_f.loc[tab_f['CVEGEO_MANZANA']==cve]
        # print(df.head())
        df['diff']= df['diff'].fillna(0).astype(float)
        geom_clas8.reset_index(drop=True, inplace=True)
        i=0
        # print(df['MODELO GEO V3 HABITADAS'])
        # print(df['diff'])
        if float(df['diff'].astype(float))< 0:
            # print('menor')
            if (float(df['prop_geo'])>=float(df['prop_df'])):
                if (abs(float(df['prop_geo'])<=1)) &  (abs(float(df['prop_geo'])>0.7)):
                    # print('primero, primero')
                    pdf=geom_clas8.loc[geom_clas8['CVEGEO_MANZANA']==cve].reset_index().iloc[:int(df['MODELO GEO V3 HABITADAS'])]
                    # print(pdf.shape)
                    pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) 
                    pdf2= pd.concat([pdf, pdf2],axis=0)
                    # print(pdf2.shape)
                elif (abs(float(df['prop_geo'])<=0.7)) &  (abs(float(df['prop_geo'])>0.5)):
                    # print('primero, segundo')
                    if any(geom_clas8.loc[geom_clas8['CVEGEO_MANZANA']==cve]['CLAVE_CASA'].notna()):
                        pdf=geom_clas8.loc[geom_clas8['CVEGEO_MANZANA']==cve].dissolve('CLAVE_CASA').reset_index()
                        ppdf=geom_clas8.loc[(geom_clas8['CVEGEO_MANZANA']==cve) & (geom_clas8['CLAVE_CASA'].isna())].drop_duplicates('LATLON')         
                        tab2= pd.merge(df, pdf.groupby('CVEGEO_MANZANA').count()['Latitud'].reset_index(), on ='CVEGEO_MANZANA')
                        tab2['diff']=tab2['MODELO GEO V3 HABITADAS'] -tab2['Latitud']
                        tab2['diff']= tab2['diff'].astype(int)
                        pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) 
                        pdf2= pd.concat([ppdf, pdf2],axis=0)
                        # print(pdf.shape)
                        # print('tab2', tab2.shape)

                        if float(tab2['diff'])< 0:
                            pdf=pdf.reset_index(drop=True).iloc[:int(tab2['MODELO GEO V3 HABITADAS'])]
                            pdf2= pd.concat([pdf, pdf2],axis=0)
                            pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) +'_tab2'
                            # print(pdf2.shape)
                        elif float(tab2['diff']) > 0:
                            cut=geom_clas8.loc[(geom_clas8['CVEGEO_MANZANA']==cve) &(~geom_clas8['ORDEN'].isin(pdf['ORDEN']))].reset_index(drop=True).iloc[:int(tab2['diff'])]
                            #             cut.drop(columns=['CLAVECATASTRAL', 'BDINTERNA_CONT_NOMBRE_COMPLETO', 'BDINTERNA_CONT_RFC',
                # 'BDINTERNA_CONT_EMAIL', 'BDINTERNA_CONT_DIRECCION'],inplace=True)
                            cut.drop(columns=['CLAVECATASTRAL_RECAUDACION'],inplace=True)
                            cut['ID_INTFIS']= cut.reset_index(drop=True).index.astype(str) + cut['CVEGEO_MANZANA'].astype(str)+'_tab2'
                            pdf2= pd.concat([pdf, cut, pdf2], axis=0)
                            # print(pdf2.shape)
                            
                elif (abs(float(df['prop_geo'])<=0.5)):
                    # print('primero, tercero')
                    try:
                        all(geom_clas8.loc[geom_clas8['CVEGEO_MANZANA']==cve]['curt'].notna())
                    except:
                        geom_clas8['curt']= float('NaN')

                    if all(geom_clas8.loc[geom_clas8['CVEGEO_MANZANA']==cve]['curt'].notna()): 
                        pdf=geom_clas8.loc[geom_clas8['CVEGEO_MANZANA']==cve].dissolve('curt').reset_index()
                        ppdf=geom_clas8.loc[(geom_clas8['CVEGEO_MANZANA']==cve) & (geom_clas8['curt'].isna())].drop_duplicates('LATLON')
                  
                        
                        tab2= pd.merge(df, pdf.groupby('CVEGEO_MANZANA').count()['Latitud'].reset_index(), on ='CVEGEO_MANZANA')
                        tab2['diff']=tab2['MODELO GEO V3 HABITADAS'] -tab2['Latitud']
                        tab2['diff']=tab2['diff'].astype(float)
                        pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) 
                        pdf2= pd.concat([ppdf, pdf2],axis=0)
                    elif any(geom_clas8.loc[geom_clas8['CVEGEO_MANZANA']==cve]['CLAVE_CASA'].notna()): 
                        pdf=geom_clas8.loc[geom_clas8['CVEGEO_MANZANA']==cve].dissolve('CLAVE_CASA').reset_index()
                        ppdf=geom_clas8.loc[(geom_clas8['CVEGEO_MANZANA']==cve) & (geom_clas8['CLAVE_CASA'].isna())].drop_duplicates('LATLON')
                        pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) 
                        tab2= pd.merge(df, pdf.groupby('CVEGEO_MANZANA').count()['Latitud'].reset_index(), on ='CVEGEO_MANZANA')
                        tab2['diff']=tab2['MODELO GEO V3 HABITADAS'] -tab2['Latitud']
                        tab2['diff']=tab2['diff'].astype(float)
                        pdf2= pd.concat([ppdf, pdf2],axis=0)
                    else: 
                        pdf=geom_clas8.loc[geom_clas8['CVEGEO_MANZANA']==cve].reset_index(drop=True).iloc[:int(df['MODELO GEO V3 HABITADAS'])]
                        tab2= pd.merge(df, pdf.groupby('CVEGEO_MANZANA').count()['Latitud'].reset_index(), on ='CVEGEO_MANZANA')
                        tab2['diff']=tab2['MODELO GEO V3 HABITADAS'] -tab2['Latitud']
                        tab2['diff']=tab2['diff'].astype(float)
                        pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) 
                        pdf2= pd.concat([pdf, pdf2],axis=0)
                        
                        # print(pdf.shape)
                        print('tab2', tab2)
                    try:    
                        if float(tab2['diff'])< 0:
                            pdf=pdf.reset_index(drop=True).iloc[:float(tab2['MODELO GEO V3 HABITADAS'])]
                            pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) +'_tab2'
                            pdf2= pd.concat([pdf, pdf2],axis=0)
                            

                            # print(pdf2.shape)
                        elif float(tab2['diff']) > 0:
                            cut=geom_clas8.loc[(geom_clas8['CVEGEO_MANZANA']==cve) &(~geom_clas8['ORDEN'].isin(pdf['ORDEN']))].reset_index(drop=True).iloc[:int(tab2['diff'])]
                #             cut.drop(columns=['CLAVECATASTRAL', 'BDINTERNA_CONT_NOMBRE_COMPLETO', 'BDINTERNA_CONT_RFC',
                # 'BDINTERNA_CONT_EMAIL', 'BDINTERNA_CONT_DIRECCION'],inplace=True)
                            cut.drop(columns=['CLAVECATASTRAL_RECAUDACION'],inplace=True)

                            cut['ID_INTFIS']= cut.reset_index(drop=True).index.astype(str) + cut['CVEGEO_MANZANA'].astype(str)+'_tab2'
                            pdf2= pd.concat([pdf, cut, pdf2], axis=0)
                            # print(pdf2.shape)
                    except:
                        pdf2= pd.concat([pdf, pdf2],axis=0)
                        # print(pdf2.shape)
                        
            elif (float(df['prop_geo'])<float(df['prop_df'])):
                # print('segundo')
                if float(df['prop_df']) <1:
                    # print('segundo, primero')
                    if (abs(float(df['prop_df'])<=0.9)) &  (abs(float(df['prop_df'])>0.7)):
                        pdf=df_final_c.loc[df_final_c['CVEGEO_MANZANA']==cve].reset_index(drop=True).iloc[:int(df['MODELO GEO V3 HABITADAS'])]
                        pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) 
                        pdf2= pd.concat([pdf, pdf2],axis=0)
                        # print(pdf2.shape)
                    elif (abs(float(df['prop_df'])<=0.7)) &  (abs(float(df['prop_df'])>0.5)):
                        # print('segundo, segundo')
                        if all(geom_clas8.loc[geom_clas8['CVEGEO_MANZANA']==cve]['CLAVE_CASA'].notna()):
                            pdf=df_final_c.loc[df_final_c['CVEGEO_MANZANA']==cve].dissolve('CLAVE_CASA').reset_index()
                            ppdf=df_final_c.loc[(df_final_c['CVEGEO_MANZANA']==cve) & (df_final_c['CLAVE_CASA'].isna())].drop_duplicates('LATLON')
                            pdf2= pd.concat([ppdf, pdf2],axis=0)
                            # print(pdf.shape)
                            pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) 
                            tab2= pd.merge(df, pdf.groupby('CVEGEO_MANZANA').count()['Latitud'].reset_index(), on ='CVEGEO_MANZANA')
                            tab2['diff']=tab2['MODELO GEO V3 HABITADAS'] -tab2['Latitud']
                        else: 
                            pdf=df_final_c.loc[df_final_c['CVEGEO_MANZANA']==cve].reset_index(drop=True).iloc[:int(df['MODELO GEO V3 HABITADAS'])]
                            # print(pdf.shape)
                            pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) 
                            pdf2= pd.concat([pdf, pdf2],axis=0)
                            tab2= pd.merge(df, pdf.groupby('CVEGEO_MANZANA').count()['Latitud'].reset_index(), on ='CVEGEO_MANZANA')
                            tab2['diff']=tab2['MODELO GEO V3 HABITADAS'] -tab2['Latitud']
                        if float(tab2['diff'])< 0:
                            pdf=pdf.reset_index(drop=True).iloc[:int(tab2['MODELO GEO V3 HABITADAS'])]
                            pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) +'_tab2'
                            pdf2= pd.concat([pdf, pdf2],axis=0)
                            # print(pdf2.shape)
                        elif float(tab2['diff']) > 0:
                            cut=df_final_c.loc[(df_final_c['CVEGEO_MANZANA']==cve) &(~df_final_c['ORDEN'].isin(pdf['ORDEN']))].reset_index(drop=True).iloc[:int(tab2['diff'])]
                            #             cut.drop(columns=['CLAVECATASTRAL', 'BDINTERNA_CONT_NOMBRE_COMPLETO', 'BDINTERNA_CONT_RFC',
                # 'BDINTERNA_CONT_EMAIL', 'BDINTERNA_CONT_DIRECCION'],inplace=True)
                            cut.drop(columns=['CLAVECATASTRAL_RECAUDACION'],inplace=True)
                            cut['ID_INTFIS']= cut.reset_index(drop=True).index.astype(str) + cut['CVEGEO_MANZANA'].astype(str)+'_tab2'
                            pdf2= pd.concat([pdf, cut, pdf2], axis=0)
                            # print(pdf2.shape)
                        else:
                            pdf2= pd.concat([pdf, pdf2],axis=0)
                            # print(pdf2.shape)
                    else:
                        if all(geom_clas8.loc[geom_clas8['CVEGEO_MANZANA']==cve]['CLAVE_CASA'].notna()):
                            pdf=df_final_c.loc[df_final_c['CVEGEO_MANZANA']==cve].dissolve('CLAVE_CASA').reset_index()
                            ppdf=df_final_c.loc[(df_final_c['CVEGEO_MANZANA']==cve) & (df_final_c['CLAVE_CASA'].isna())].drop_duplicates('LATLON')
                            pdf2= pd.concat([ppdf, pdf2],axis=0)
                            # print(pdf.shape)
                            tab2= pd.merge(df, pdf.groupby('CVEGEO_MANZANA').count()['Latitud'].reset_index(), on ='CVEGEO_MANZANA')
                            tab2['diff']=tab2['MODELO GEO V3 HABITADAS'] -tab2['Latitud']
                            pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) 
                        else: 
                            pdf=df_final_c.loc[df_final_c['CVEGEO_MANZANA']==cve].reset_index(drop=True).iloc[:int(df['MODELO GEO V3 HABITADAS'])]
                            # print(pdf.shape)
                            tab2= pd.merge(df, pdf.groupby('CVEGEO_MANZANA').count()['Latitud'].reset_index(), on ='CVEGEO_MANZANA')
                            tab2['diff']=tab2['MODELO GEO V3 HABITADAS'] -tab2['Latitud']
                            pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) 
                            pdf2= pd.concat([pdf, pdf2],axis=0)
                        if float(tab2['diff'])< 0:
                            pdf=pdf.reset_index(drop=True).iloc[:int(tab2['MODELO GEO V3 HABITADAS'])]
                            pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) +'_tab2'
                            pdf2= pd.concat([pdf, pdf2],axis=0)
                            # print(pdf2.shape)
                        elif float(tab2['diff']) > 0:
                            cut=df_final_c.loc[(df_final_c['CVEGEO_MANZANA']==cve) &(~df_final_c['ORDEN'].isin(pdf['ORDEN']))].reset_index(drop=True).iloc[:int(tab2['diff'])]
                            #             cut.drop(columns=['CLAVECATASTRAL', 'BDINTERNA_CONT_NOMBRE_COMPLETO', 'BDINTERNA_CONT_RFC',
                # 'BDINTERNA_CONT_EMAIL', 'BDINTERNA_CONT_DIRECCION'],inplace=True)
                            cut.drop(columns=['CLAVECATASTRAL_RECAUDACION'],inplace=True)
                            cut['ID_INTFIS']= cut.reset_index(drop=True).index.astype(str) + cut['CVEGEO_MANZANA'].astype(str)+'_tab2'
                            pdf2= pd.concat([pdf, cut, pdf2], axis=0)
                            # print(pdf2.shape)
                        else:
                            pdf2= pd.concat([pdf, pdf2],axis=0)
                            # print(pdf2.shape)

                elif (abs(float(df['prop_geo'])<=0.5)):
                    if any(df_final_c.loc[df_final_c['CVEGEO_MANZANA']==cve]['curt'].notna()): 
                        pdf=df_final_c.loc[df_final_c['CVEGEO_MANZANA']==cve].dissolve('curt').reset_index()
                        ppdf=df_final_c.loc[(df_final_c['CVEGEO_MANZANA']==cve) & (df_final_c['curt'].isna())].drop_duplicates('LATLON')
                        pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) 
                        tab2= pd.merge(df, pdf.groupby('CVEGEO_MANZANA').count()['Latitud'].reset_index(), on ='CVEGEO_MANZANA')
                        tab2['diff']=tab2['MODELO GEO V3 HABITADAS'] -tab2['Latitud']
                        pdf2= pd.concat([ppdf, pdf2],axis=0)
                    elif any(geom_clas8.loc[geom_clas8['CVEGEO_MANZANA']==cve]['CLAVE_CASA'].isna()): 
                        pdf=df_final_c.loc[df_final_c['CVEGEO_MANZANA']==cve].dissolve('CLAVE_CASA').reset_index()
                        ppdf=df_final_c.loc[(df_final_c['CVEGEO_MANZANA']==cve) & (df_final_c['CLAVE_CASA'].isna())].drop_duplicates('LATLON')
                        tab2= pd.merge(df, pdf.groupby('CVEGEO_MANZANA').count()['Latitud'].reset_index(), on ='CVEGEO_MANZANA')
                        tab2['diff']=tab2['MODELO GEO V3 HABITADAS'] -tab2['Latitud']
                        ppdf['ID_INTFIS']= ppdf.reset_index(drop=True).index.astype(str) + ppdf['CVEGEO_MANZANA'].astype(str) 
                        pdf2= pd.concat([ppdf, pdf2],axis=0)
                            
                    else: 
                        pdf=df_final_c.loc[df_final_c['CVEGEO_MANZANA']==cve].reset_index(drop=True).iloc[:int(df['MODELO GEO V3 HABITADAS'])]
                        tab2= pd.merge(df, pdf.groupby('CVEGEO_MANZANA').count()['Latitud'].reset_index(), on ='CVEGEO_MANZANA')
                        tab2['diff']=tab2['MODELO GEO V3 HABITADAS'] -tab2['Latitud']
                        tab2['diff']=tab2['diff'].astype(float)
                        pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) 
                        pdf2= pd.concat([pdf, pdf2],axis=0)

                    try:
                        if float(tab2['diff'])< 0:
                            pdf=pdf.reset_index(drop=True).iloc[:int(tab2['MODELO GEO V3 HABITADAS'])]
                            pdf['ID_INTFIS']= pdf.reset_index(drop=True).index.astype(str) + pdf['CVEGEO_MANZANA'].astype(str) +'_tab2'
                            pdf2= pd.concat([pdf, pdf2],axis=0)
                            # print(pdf2.shape)
                        elif float(tab2['diff']) > 0:
                            cut=df_final_c.loc[(df_final_c['CVEGEO_MANZANA']==cve) &(~df_final_c['ORDEN'].isin(pdf['ORDEN']))].reset_index(drop=True).iloc[:int(tab2['diff'])]
                            #             cut.drop(columns=['CLAVECATASTRAL', 'BDINTERNA_CONT_NOMBRE_COMPLETO', 'BDINTERNA_CONT_RFC',
                # 'BDINTERNA_CONT_EMAIL', 'BDINTERNA_CONT_DIRECCION'],inplace=True)
                            cut.drop(columns=['CLAVECATASTRAL_RECAUDACION'],inplace=True)
                            cut['ID_INTFIS']= cut.reset_index(drop=True).index.astype(str) + cut['CVEGEO_MANZANA'].astype(str)+'_tab2'
                            pdf2= pd.concat([pdf, cut, pdf2], axis=0)
                            # print(pdf2.shape)
                        
                    except:
                        pdf2= pd.concat([pdf, pdf2],axis=0)
                        # print(pdf2.shape)

            pdf2= pd.concat([pdf, pdf2],axis=0)
                        
        
        print('Final',pdf2.shape)
    return(pdf2)
    


def puntos_internos(base, poli):
    assert poli.crs== 4326
    base= transform_df_to_gpd(base, lon_col='Longitud', lat_col='Latitud', crs=4326)
    cruce= gpd.sjoin(base.to_crs(4326),poli)
    return(cruce)
