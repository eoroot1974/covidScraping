#!/usr/bin/env python
# coding: utf-8

# # Detección de patrones existentes en diferentes indicadores con respecto a datos de la COVID-19
# 
# 
# <img src="covidLogo.png" alt="covid" style="float:left;" width="300"/>
# <img src="./UOCLogo.jpg" alt="uoc" style="margin-top: 41px;float: right;" width="300"/>

# __Universidad Oberta de Catalunya__
# 
# 
# __Tipología y Ciclo de vida de los datos__
# 
# 
# __Enrique Otero Espinosa__
# 
# 
# __Francisco Javier Melchor González__

# # Contexto

# La materia que trata el conjunto de datos generado, corresponde con los datos correspondientes con la pandemia COVID-19 desde que esta se empezó a manifestar en los diferentes países. Además, el conjunto de datos contiene indicadores importantes de los diferentes países (así como la expectance de vida, la fertilidad, el acceso a la electricidad, el número de tests realizados...) que desde nuestro punto de vista, consideramos que pueden llegar a tener cierta influencia en las muertes que ha ocasionado dicha pandemia en los diferentes países
# 
# Los sitios webs que se han elegido para extraer esta información, han sido:
# 
# 
# * [Wolrldometer](https://www.worldometers.info/), sitio web de referencia que proporciona estimaciones y estadísticas en tiempo real para diversos temas.
# * [WorldBankData](https://databank.worldbank.org/home), sitio web que sirve como herramienta de análisis y visualización que contiene colecciones de datos de diferentes temas.
# 
# Ambas fuentes, proporcionan esta información porque son páginas dedicadas a proporcionar información de diversos temas para que se puedan realizar análisis sobre los mismos.

# Para conseguir el fin descrito anteriormente, se han elegido dos fuentes de datos:
# - a) Por un lado, la **página https://www.worldometers.info/coronavirus/** que corresponde con la fuente de datos que proporciona información sobre la pandemia en los diferentes países, para obtener la información de la misma se ha utilizado la técnica de Web Scrapping 
# - b) Por otro lado, la **librería wbdata**, que corresponde con la fuente de datos que proporciona información sobre los valores de los distintos indicadores de cada país. Dicha librería, realmente permite obtener información de la página del Banco Mundial de datos abiertos: https://databank.worldbank.org/home de una manera sencilla y parametrizada
# 
# La razón por la que se han elegido estas fuentes es porque se consideran fuentes de datos fiables y además de uso público que incluso normalmente se publican con el fin de que se puedan realizar estudios de investigación sobre los mismos, por lo que resulta más sencillo que obtenerlo de páginas que realmente no los publican con este fin.

# # Título del dataset

# Debido a que el dataset que se quiere conseguir fusiona información de la pandemia COVID-19 en los diferentes países y de los valores de los diferentes indicadores que se consideren a simple vista que pueden llegar a tener una relación con las muertes ocasionadas por la pandemia, el nombre que hemos elegido para el dataset que se va a generar es: **covid19AndWbIndicators**

# # Descripción del dataset
# El conjunto de datos generado como parte de este proyecto reúne por un lado las métricas que permiten medir el impacto de la pandemia COVID-19, con respecto a las muertes y a los casos registrados en cada país, y por otro lado un conjunto de indicadores que se han considerado que pueden tener cierta relación con dicho impacto causado.

# # Representación gráfica
# ![RepGrafica](Covid_vs_WorldBankdata.jpeg)

# # Contenido:
# Por cada país que contiene el dataset creado, se almacena la siguiente información:
# *  **TotalDeaths1M:** Número total de muertes causadas por la COVID-19 en el país en concreto en unidades de millón.
# * **TotalTests1M:** Número total de tests para detectar la COVID-19 realizados en el país en concreto en unidades de millón.
# * **GDPperCap:** Producto interior bruto per cápita del país en concreto, lo que indica la producción económica por persona de un país.
# * **LifeExpect: Expectancia de vida del país en concreto en años.
# * **DrsPer1K: Número de médicos por cada mil personas que hay en el país en concreto.
# * **NewInfectHIV: Número de nuevos casos de VIH.
# * **FertilityRate: Número de descendientes por cada mujer que habita en el país en concreto.
# * **UrbanPopulation: Número de habitantes pertenecientes a una población urbana, es decir, que vive en las grandes ciudades, del país en concreto
# * **Country:** País en concreto al que pertencen los datos
# 

# # Contenido:
# Los campos finales del dataset son 25 y provienen de las 2 fuentes de datos usadas. Los datos se obtienen en un dia concreto ya que el objetivo del dataset no es realizar análisis de series temporales o tendencias sino análisis estáticos como búsqueda de patrones, clustering o incluso análisis de regresiones lineales.
# Los campos son los siguientes:  
# 
# * Campos obtenidos de la página de [Worldometer](https://www.worldometers.info/coronavirus/) dedicada a datos sobre la COVID-19
# 
#     *	**ID:** Campo numérico identificador de país. 
#     *	**TotalCases:** Número de contagiados totales de Coronavirus reportados desde el inicio de la pandemia. (float).
#     *	**TotalDeaths:** Número de fallecimientos totales por Coronavirus reportados desde el inicio de la pandemia. (float).
#     *	**TotalCases1M:** Número de contagiados por millón de habitantes. (float).
#     *	**TotalDeaths1M:** Número de fallecimientos por millón de habitantes. (float).
#     *	**TotalTests:** Número total de tests realizados. (float).
#     *	**TotalTests1M:** Número de tests realizados por millón de habitantes. (float).
#     *	**Population:** Población del pais. (float).
#     *	**Continent:** Continente.
#     *	**1CaseEvery:** Ratio entre número total de habitantes y número total de contagiados. (float).
#     *	**1DeathEvery:** Ratio entre número total de habitantes y número total de fallecimientos. (float).
#     *	**1TestEvery:** Ratio entre número total de habitantes y número total tests realizados. (float).
#     
#     
# * Campos obtenidos de la página de WorldBankData, a través de la librería  [wbdata](https://pypi.org/project/wbdata/#description) de Python con la que cuenta la propia página para poder acceder a los datos. Aunque es parametrizable, concretamente el dataset desarrollado en este proyecto, contiene índices recopilados durante el año 2018, por ser este el último año que contiene datos más completos sobre todas las variables. 
#     * **AccessElectricity:** Porcentaje de la población que tiene acceso a electricidad. (percent)
#     * **GDP:** PIB - Producto Interior Bruto del país (del inglés Gross Domestic Product). (float).
#     * **GDPperCap:** Producto Interior Bruto per cápita. (float).
#     * **LifeExpect:** Esperanza o expectativa de vida al nacer. (float).
#     * **DrsPer1k:** Número de médicos por cada 1000 habitantes. (float).
#     * **NewInfecHIV:** Nuevas personas infectadas con HIV desde el año anterior. (float).
#     * **FertilityRate:** Tasa de fertilidad o tasa de fecundidad. Número de nacimientos por cada mil mujeres en edad fértil (15-49 años) en un año. (float).
#     * **UrbanPopulation:** Habitantes viviendo en ciudades o población urbana. (float).
#     * **UrbanPopulationPerc:** Porcentaje de habitantes viviendo en ciudades. (percent)

# # Agradecimientos 
# Los datos han sido recolectados de dos fuente, como se ha indicado anteriormente:
# * En el caso de la página **Worldometer**, se ha hecho uso del la técnica de Web Scraping, con el lenguaje de programación Python. Esta página, está dirigida por un equipo internacional de desarrolladores, investigadores y voluntarios con el objetivo de hacer que las estadísticas mundiales estén disponibles en un formato que invite a la reflexión y sea relevante en el tiempo para una amplia audiencia en todo el mundo. Es publicado por una pequeña empresa de medios digitales independiente con sede en los Estados Unidos. Los datos son de dominio público y no poseen una categorización DOI.
# 
# * En el caso de la página **WorldBankData** también son de dominio público sin registro en DOI. Según los términos de uso es recomendable, aunque no obligatorio, incluir una cita que incluya el año y los indicadores usados de la forma:
# 
#     *World Bank, 2018 indicators, 
#     "EG.ELC.ACCS.ZS":"AccessElectricity","NY.GDP.MKTP.CD":"GDP","NY.GDP.PCAP.CD":"GDPperCap","SH.XPD.CHEX.PC.CD":"HealthExpenseperCap","IT.NET.USER.ZS":"IndividUsingInternet","SP.DYN.LE00.IN":"LifeExpect","SH.MED.PHYS.ZS":"DrsPer1k","GB.XPD.RSDV.GD.ZS":"RDExpen","SH.HIV.INCD":"NewInfecHIV","SP.DYN.TFRT.IN":"FertilityRate","per_si_allsi.cov_pop_tot":"CovSocialInsurance","SP.URB.TOTL":"UrbanPopulation", "SP.URB.TOTL.IN.ZS":"UrbanPopulationPerc"*
#     
#     Por otro lado, la librería wbdata utilizada para obtener los datos del WorldBankData, ha sido desarrollada por Oliver Sherouse. En la descripción de la librería establece recomendable, aunque no obligatorio incluir una cita de la siguiente forma:
#     
#     *Sherouse, Oliver (2014). Wbdata. Arlington, VA.       
# Available from http://github.com/OliverSherouse/wbdata.13*
# 
# Por último, no se han encontrado conjuntos de datos o proyectos que contengan un dataset con ambas fuentes de datos como el desarrollado en este proyecto
# 
# 

# # Inspiración
# Actualmente, debido a la pandemia COVID-19 estamos pasando por una situación que está marcando un hecho histórico a nivel mundial. Han sido muchas las muertes causadas por la misma, las secuelas dejadas en algunas personas y lo mucho que ha afectado esta a nivel económico a los diferentes países del mundo. Es por ello, que consideramos muy interesante tratar datos con respecto a la misma y relacionarlos con los diferentes indicadores de los diferentes países, para tratar de encontrar así alguna relación existente entre los valores de los diferentes indicadores, y el nivel con el que está afectando la pandemia a cada uno de los países analizados.
# 
# La pregunta que se trata de responder realizando esta recolección de datos y este anális son: 
# * ¿Qué indicadores están relacionados con el nivel de afectación de la pandemia en cada país?
# * ¿Por qué dichos indicadores están relacionados con el nivel de afectación de la pandemia a cada país?
# 
# Para tratar así de estudiar la relación y tratar de extraer como conclusión a que se debe que un país tenga más casos que otros (con respecto a los indicadores)
# 
# La manera de medier el nivel de afectación de la pandemia que hemos tomado en este proyecto es el número de muertes causadas por la misma en cada país

# # Licencia
# La licencia elegida para el dataset desarrollado es la siguiente:
# * **Released Under CC0: Public Domain License**
# 
# Debido a que las fuentes de las cuales se han extraído los datos, son ambas de dominio público.

# # Código

# In[1]:


import wbdata
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.ensemble import ExtraTreesClassifier
import statistics
from geopy.geocoders import Nominatim
import math
import folium
from folium.plugins import HeatMap
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.cluster import KMeans
import seaborn as sns


# ## Extracción de datos

# In[2]:


url = 'https://www.worldometers.info/coronavirus/'
requests.get(url)
wdweb = requests.get(url)


# In[3]:


wdweb


# In[4]:


wdwebsoup = BeautifulSoup(wdweb.text, 'lxml')


# In[5]:


wdwebtable_data = wdwebsoup.find('table', id = 'main_table_countries_yesterday')


# In[6]:


headers = []
for i in wdwebtable_data.find_all('th'):
    title = i.text
    headers.append(title)


# In[ ]:





# In[7]:


covid = pd.DataFrame(columns = headers)


# In[8]:


for j in wdwebtable_data.find_all('tr')[1:]:
        row_data = j.find_all('td')
        row = [td.text for td in row_data]
        length = len(covid)
        covid.loc[length] = row


# In[9]:


covid.columns = ['ID','country','TotalCases','NewCases','TotalDeaths','NewDeaths','TotalRecov','NewRecov','ActiveCases','SeriousCritical','TotalCases1M','TotalDeaths1M','TotalTests','TotalTests1M','Population','Continent','1CaseEvery','1DeathEvery','1TestEvery']
covid.set_index('country', inplace=True, drop=True)


# In[10]:


covid.shape


# In[11]:


covid.index


# In[12]:


pd.set_option('display.max_rows', None)
covid.index=covid.index.str.replace("\n","")
covid=covid.rename(index={'USA': 'United States'})


# In[13]:


data_date = datetime.datetime(2018, 12, 31), datetime.datetime(2018, 12, 31)
# countries = [i['id'] for i in wbdata.get_country(incomelevel='HIC')]
indicators = {"EG.ELC.ACCS.ZS":"AccessElectricity", "NY.GDP.MKTP.CD":"GDP", 
             "NY.GDP.PCAP.CD":"GDPperCap", "SH.XPD.CHEX.PC.CD":"HealthExpenseperCap", 
              "IT.NET.USER.ZS":"IndividUsingInternet", "SP.DYN.LE00.IN":"LifeExpect", 
              "SH.MED.PHYS.ZS":"DrsPer1k", "GB.XPD.RSDV.GD.ZS":"RDExpen", 
              "SH.HIV.INCD":"NewInfecHIV", "SP.DYN.TFRT.IN":"FertilityRate",
             "per_si_allsi.cov_pop_tot":"CovSocialInsurance", 
             "SP.URB.TOTL":"UrbanPopulation", "SP.URB.TOTL.IN.ZS":"UrbanPopulationPerc"}
wbdf = wbdata.get_dataframe(indicators, country="all", data_date=data_date)


# In[14]:


covidandwb_merged = pd.merge(covid,wbdf, on=["country"])


# In[15]:


covidandwb_merged.head()


# In[16]:


covidandwb_merged.shape


# In[17]:


covidandwb = covidandwb_merged


# ## Procesamiento de los datos obtenidos

# In[18]:


covidandwb = covidandwb.drop(["NewCases", "NewDeaths", "TotalRecov", "NewRecov", 
                              "ActiveCases", "SeriousCritical"], axis=1)
covidandwb = covidandwb.drop(["World"], axis=0)
covidandwb = covidandwb.drop(["North America"], axis=0)


# In[19]:


covidandwb.head()


# In[20]:


TotalDeathsindex = covidandwb[covidandwb['TotalDeaths'].str.match(' ')].index
covidandwb.loc[TotalDeathsindex, 'TotalDeaths'] = 0
covidandwb['TotalDeaths']=covidandwb['TotalDeaths'].str.replace(",","").astype(float)
covidandwb.loc[TotalDeathsindex, 'TotalDeaths1M'] = 0
covidandwb['TotalDeaths1M']=covidandwb['TotalDeaths1M'].str.replace(",","").astype(float)
covidandwb['TotalCases1M']=covidandwb['TotalCases1M'].str.replace(",","").astype(float)
covidandwb['TotalCases']=covidandwb['TotalCases'].str.replace(",","").astype(float)


# In[21]:


covidandwb.head()


# In[22]:


covidandwb.dtypes


# In[23]:


covidandwb['1DeathEvery']= covidandwb['1DeathEvery'].str.replace(",","")
covidandwb['1DeathEvery']=covidandwb['1DeathEvery'].str.replace(r'^\s*$','NaN')
covidandwb['1DeathEvery']=covidandwb['1DeathEvery'].astype(float)


# In[24]:


covidandwb['TotalTests']= covidandwb['TotalTests'].str.replace(",","")
covidandwb['TotalTests']=covidandwb['TotalTests'].str.replace(r'^\s*$','NaN')
covidandwb['TotalTests']=covidandwb['TotalTests'].astype(float)


# In[25]:


covidandwb['TotalTests1M']= covidandwb['TotalTests1M'].str.replace(",","")
covidandwb['TotalTests1M']=covidandwb['TotalTests1M'].str.replace(r'^\s*$','NaN')
covidandwb['TotalTests1M']=covidandwb['TotalTests1M'].astype(float)


# In[26]:


covidandwb['Population']= covidandwb['Population'].str.replace(",","")
covidandwb['Population']=covidandwb['Population'].str.replace(r'^\s*$','NaN')
covidandwb['Population']=covidandwb['Population'].astype(float)


# In[27]:


covidandwb['1CaseEvery']= covidandwb['1CaseEvery'].str.replace(",","")
covidandwb['1CaseEvery']=covidandwb['1CaseEvery'].str.replace(r'^\s*$','NaN')
covidandwb['1CaseEvery']=covidandwb['1CaseEvery'].astype(float)


# In[28]:


covidandwb['HealthExpenseperCap']= covidandwb['HealthExpenseperCap'].str.replace(",","")
covidandwb['HealthExpenseperCap']=covidandwb['HealthExpenseperCap'].str.replace(r'^\s*$','NaN')
covidandwb['HealthExpenseperCap']=covidandwb['HealthExpenseperCap'].astype(float)


# In[29]:


covidandwb['1TestEvery']= covidandwb['1TestEvery'].str.replace(",","")
covidandwb['1TestEvery']=covidandwb['1TestEvery'].str.replace(r'^\s*$','NaN')
covidandwb['1TestEvery']=covidandwb['1TestEvery'].astype(float)


# In[30]:


covidandwb.dtypes


# In[31]:


covidandwb.shape


# In[32]:


covidandwb.isnull().sum()


# In[33]:


covidandwb.columns


# Eliminamos:
# * **CovSocialInsurance**
# * **RDExpen**
# * **HealthExpenseperCap**
# * **IndividUsingInternet**
# 
# Debido a que todas ellas cuentan con más de un 60% de datos nulos

# In[34]:


columns_acceptables = ['ID', 'TotalCases', 'TotalDeaths', 'TotalCases1M', 'TotalDeaths1M',
       'TotalTests', 'TotalTests1M', 'Population', 'Continent', '1CaseEvery',
       '1DeathEvery', '1TestEvery', 'AccessElectricity', 'GDP', 'GDPperCap', 'LifeExpect','DrsPer1k', 'NewInfecHIV', 'FertilityRate',
       'UrbanPopulation', 'UrbanPopulationPerc']


# In[35]:


covidandwb = covidandwb[columns_acceptables]
covidandwb.isnull().sum()


# In[36]:


def imputationFunct(x, indexColumn):
    if math.isnan(x.iloc[indexColumn]):
        x.iloc[indexColumn] = statistics.median(covidandwb.loc[covidandwb['Continent'] == x.iloc[8]].iloc[:, indexColumn].dropna())
    return x.iloc[indexColumn]


# In[37]:


columnsWithNaN = [2,4,5,6,10,11,12,13,14,15,16,17,18,19,20]


# ### Imputation Data

# In[38]:


for column in columnsWithNaN:
    covidandwb.iloc[:,[column]] = covidandwb.apply(imputationFunct,axis=1,args=(column,))


# In[39]:


covidandwb.isnull().sum()


# In[40]:


covidandwb.head()


# In[41]:


geolocator = Nominatim(user_agent='myapplication')


# In[42]:


def getLatitude(x):
    return geolocator.geocode(x[0]).latitude


# In[43]:


def getLongitude(x):
    return geolocator.geocode(x[0]).longitude


# In[44]:


covidandwb.head()


# In[45]:


covidandwb = covidandwb.reset_index()
covidandwb.head()


# In[46]:


covidandwb['Latitude'] = covidandwb.apply(getLatitude,axis=1)


# In[47]:


covidandwb['Longitude'] = covidandwb.apply(getLongitude,axis=1)


# In[48]:


covidandwb.head()


# ## Representaciones gráficas de los datos

# ### HeatMap

# In[49]:


map_hooray = folium.Map([40.4166359,-3.7059988], zoom_start=3)


# In[50]:


HeatMap(data=covidandwb[['Latitude','Longitude','TotalCases1M']].groupby(['Latitude','Longitude']).sum().reset_index().values.tolist()).add_to(map_hooray)


# In[51]:


map_hooray


# In[52]:


covidandwb.set_index('country', inplace=True, drop=True)


# ### Plots of relations

# In[53]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 20,16
plt.scatter(covidandwb['DrsPer1k'], covidandwb['TotalDeaths1M'])
plt.show()


# In[54]:


# Recreamos el campo UrbanPopulationPerc


# In[55]:


covidandwb['UrbanPopulationPerc'].head()


# In[56]:


def fx(x, y):
    return x*100/y
covidandwb['UrbanPopulationPerc'] = np.vectorize(fx)(covidandwb['UrbanPopulation'], covidandwb['Population'])


# In[57]:


covidandwb['UrbanPopulationPerc'].head()


# In[58]:


# Exportamos el Datframe final a CSV


# In[59]:


csv_path='./covidandwb.csv'
covidandwb.to_csv(csv_path, index=False, header=True)


# ### Clusterización con KMeans

# In[60]:


covidkmeans_cols=['TotalDeaths1M', 'TotalTests1M', 'GDPperCap', 'LifeExpect','DrsPer1k', 'NewInfecHIV', 'FertilityRate',
       'UrbanPopulation']
covidkmeans=covidandwb[covidkmeans_cols]
covidkmeans.head()


# In[61]:


scaler = preprocessing.MinMaxScaler()
features_normal = scaler.fit_transform(covidkmeans)


# In[62]:


features_normal


# In[63]:


inertia = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(features_normal)
    kmeanModel.fit(features_normal)
    inertia.append(kmeanModel.inertia_)
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()


# In[64]:


kmeans = KMeans(n_clusters=5).fit(features_normal)


# In[65]:


covidkmeans.head()


# In[66]:


labels = pd.DataFrame(kmeans.labels_) #This is where the label output of the KMeans we just ran lives. Make it a dataframe so we can concatenate back to the original data
covidkmeans=covidkmeans.assign(Country=covidkmeans.index.get_level_values('country'))
covidkmeans.reset_index(drop=True, inplace=True)
labeledcovidkmeans = pd.concat((covidkmeans,labels), axis=1)
labeledcovidkmeans = labeledcovidkmeans.rename({0:'labels'},axis=1)


# In[67]:


labeledcovidkmeans.head()


# In[68]:


sns.pairplot(labeledcovidkmeans,hue='labels')


# In[69]:


sns.lmplot(x='FertilityRate',y='TotalDeaths1M',data=labeledcovidkmeans,hue='labels',fit_reg=False, height=8)


# In[70]:


sns.lmplot(x='NewInfecHIV',y='TotalDeaths1M',data=labeledcovidkmeans,hue='labels',fit_reg=False,height=8)


# In[71]:


sns.lmplot(x='LifeExpect',y='TotalDeaths1M',data=labeledcovidkmeans,hue='labels',fit_reg=False, height=8)


# In[72]:


sns.lmplot(x='TotalTests1M',y='TotalDeaths1M',data=labeledcovidkmeans,hue='labels',fit_reg=False, height=8)


# Finalmente, atendiendo a los clustering obtenidos por el modelo, se detecta que hay un grupo de países donde los parámetros parecen indicar que son países subdesarrollados, en los cuales se detectan un menor número de muertes debido a la pandemia de la COVID-19

# # Dataset:
# Tras publicar el dataset en Zenodo el DOI obtenido es el siguiente: 10.5281/zenodo.4256839
# 
# [DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4256839.svg)
# 

# <a href="https://doi.org/10.5281/zenodo.4256839"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4256839.svg" alt="DOI"></a>
# 

# | Contribuciones                         | Firma |
# |----------------------------------------|-------|
# | Investigación Inicial                  | <img src="../FirmaFran.jpg" alt="Firma Fran" style="float:left;" width="60"/><img src="../EnriqueOtero_Firma.jpg" alt="Firma Enrique" style="margin-left: 3px;float: right;" width="60"/>|
# | Análisis de herramientas Scrappping    |<img src="../FirmaFran.jpg" alt="Firma Fran" style="float:left;" width="60"/><img src="../EnriqueOtero_Firma.jpg" alt="Firma Enrique" style="margin-left: 3px;float: right;" width="60"/>       |
# | Codigo python Scrapping y app          |<img src="../FirmaFran.jpg" alt="Firma Fran" style="float:left;" width="60"/><img src="../EnriqueOtero_Firma.jpg" alt="Firma Enrique" style="margin-left: 3px;float: right;" width="60"/>       |
# | Investigación de Indicadores WorldBank |<img src="../FirmaFran.jpg" alt="Firma Fran" style="float:left;" width="60"/><img src="../EnriqueOtero_Firma.jpg" alt="Firma Enrique" style="margin-left: 3px;float: right;" width="60"/>       |
# | Obtención de datos Covid y WorldBank   |<img src="../FirmaFran.jpg" alt="Firma Fran" style="float:left;" width="60"/><img src="../EnriqueOtero_Firma.jpg" alt="Firma Enrique" style="margin-left: 3px;float: right;" width="60"/>       |
# | Procesado y limpieza de Datos          |<img src="../FirmaFran.jpg" alt="Firma Fran" style="float:left;" width="60"/><img src="../EnriqueOtero_Firma.jpg" alt="Firma Enrique" style="margin-left: 3px;float: right;" width="60"/>       |
# | Análisis ejemplo Clustering            |<img src="../FirmaFran.jpg" alt="Firma Fran" style="float:left;" width="60"/><img src="../EnriqueOtero_Firma.jpg" alt="Firma Enrique" style="margin-left: 3px;float: right;" width="60"/>       |
# | Export de DataSet                      |<img src="../FirmaFran.jpg" alt="Firma Fran" style="float:left;" width="60"/><img src="../EnriqueOtero_Firma.jpg" alt="Firma Enrique" style="margin-left: 3px;float: right;" width="60"/>       |
# | Documentacion                          |<img src="../FirmaFran.jpg" alt="Firma Fran" style="float:left;" width="60"/><img src="../EnriqueOtero_Firma.jpg" alt="Firma Enrique" style="margin-left: 3px;float: right;" width="60"/>       |
