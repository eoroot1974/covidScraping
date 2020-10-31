# covidScrapping
El objetivo del proyecto es construir un dataset con datos de 2 fuentes distintas: 

    a. datos de covid de worldometers.info via scraping y .
    b. datos del Banco Mundial por país. Para los datos del Banco Mundial usamos una librería de python diseñada precisamente para descargar info de esta institución (libreria wbdata).

Este DataSet es estatico ya que no esta orientado a analisis de series temporales sino a analisis tipo clustering para agrupar estadisticas de covid con indices del Banco Mundial. En el codigo python anexo se incluye un breve analisis inicial con KMeans Clustering. Tambien es suscptible de analisis tipo regresiones lineales con las variables de Numero de Muertes por Covid o Numero de Muertes por Covid por 1M de habitantes como variables target, explicadas por el resto de variables.
