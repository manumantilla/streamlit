import streamlit as st
# Título
st.title("Delitos en Bucaramanga y su Analisis de Datos")
# Introducción
st.write("""
Esta es la representacion, explicacion y conclusiones despues de analizar los delitos cometidos en Bucaramanga Grupo: 
         Andres Felipe Jaimes Rico Manuel Delgado Mantilla.
""")
nombre_lector = st.text_input("Introduce tu nombre", "Nombre")
if nombre_lector:
    st.write(f"Bienvenido, {nombre_lector}!Gracias por tomarte el tiempo de leer este analisis de datos sobre los delitos cometidos en Bucaramanga y asi conocer esta problemática más cerca.")
else:
    st.write("Por favor, introduce tu nombre arriba para una bienvenida personalizada.")
# Imagen
st.image("analisis de datos delitos bga.png")
# Código de ejemplo
st.write("Vamos a comenzar a desglosar este Analisis :)")
st.write("# ANALISIS")
st.write("Importamos las librerias necesarias")
codigo_python = """
import pandas as pd
import numpy as np
import os
#Graficadores
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import urllib
import plotly.express as px
"""
st.code(codigo_python, language="python")
st.write("##Necesitamos datos asi que los traemos, estos datos son suministrados por un ente gubernamental provenientes de este link: https://www.datos.gov.co/Seguridad-y-Defensa/92-Delitos-en-Bucaramanga-enero-2016-a-julio-de-20/x46e-abhz")
codigo_python = """
import pandas as pd
import numpy as np
import os
#Graficadores
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import urllib
import plotly.express as px
"""
st.code(codigo_python, language="python")
st.write("En el tratamiento de datos necesitamos unir algunas columnas, quitar las vacias etc")
codigo_python = """
#La biblioteca merge realiza fusion entre DataFrames
df=pd.merge(df,dfbarrios,on="NOM_COM")
#Vamos a comenzar a eliminar columnas, el argumento es Unname:0 y loc, con el axis decimos que elimine la columna
#inplcae=true, estamso diciendo que mdoifique el df y no cree otro
df.drop(['Unnamed: 0','loc'],axis=1,inplace=True)
#Aqui concatenamos fecha y hora y agregamos un espacio en blanco
df['FECHA_COMPLETA'] = df["FECHA_HECHO"]+ ' ' + df["HORA_HECHO"]
# Agrupamos por año
#Por count agregamos cuantas veces aparece cada valor unico en la columna
#Convertimos a DataFrame
cantidadaño=df.groupby(df["FECHA_HECHO"].dt.year)["DESCRIPCION_CONDUCTA"].count().to_frame()
cantidadaño
"""
st.code(codigo_python, language="python")
#G R A F I C A C I O N
st.write(" Las graficas nos muestran los datos de una manera mas entendible y este es el momento de usarlos:")
codigo_python = """
#Graficamos la informacion de arriba en barras
fig,ax,=plt.subplots()
ax.bar(cantidadaño.index,cantidadaño["DESCRIPCION_CONDUCTA"])
"""
st.code(codigo_python, language="python")
# 
st.write(" Se muestra la relacion entre la cantidad de casos respecto a cada año")
codigo_python = """
#Graficamos la informacion de arriba en barras
fig,ax,=plt.subplots()
ax.bar(cantidadaño.index,cantidadaño["DESCRIPCION_CONDUCTA"])
"""
st.code(codigo_python, language="python")
st.image("2grafico.jpg")
#
st.write("Intentamos encontrar con regresión encontrar relacion entre dos variables:")
codigo_python = """
#Usamos la libreria Seaborn, que pueda mostrar la relacion entre dos variables
# X, representa los años
# Y, representa la cantidad de casos
#line_kws: La apariencia de la regresion lineal
sns.regplot(x=cantidadxañosin2023.index,y=cantidadxañosin2023["DESCRIPCION_CONDUCTA"],scatter_kws={"color":"purple", "alpha":0.8},line_kws={"color":"green","alpha":0.8})
"""
st.code(codigo_python, language="python")
st.image("3grafico.jpg")
#
st.write("Mostramos un mapa de calor, despues de haber modificado nuestro DataFrame para que nos diera las coordenadas:")
codigo_python = """
fig = px.density_mapbox(cantidadComuna, lat = 'lat', lon = 'lon',z='DESCRIPCION_CONDUCTA',
                        radius = 50,
                        hover_name='NOM_COM',
                        color_continuous_scale='rainbow',
                        center = dict(lat = 7.12539, lon = -73.1198),
                        zoom = 12,
                        mapbox_style = 'open-street-map')
fig.show()
"""
st.code(codigo_python, language="python")
st.image("4grafico.jpg")
#
st.write("Mostramos un mapa de calor, despues de haber modificado nuestro DataFrame para que nos diera las coordenadas:")
codigo_python = """
frecuencias_barrios_filtradas = frecuencias_barrios[frecuencias_barrios >= 1000] #filtramos
plt.figure(figsize=(8, 4))  # Tamaño de la figura
frecuencias_barrios_filtradas.plot(kind='barh', color='skyblue')  # Tipo de gráfico y sus colores
plt.title('Registro de frecuencia en Barrios (Filtrados & Organizados)')
plt.xlabel('FRECUENCIA')  # Etiqueta del eje x
plt.ylabel('BARRIOS')  # Etiqueta del eje y
plt.tight_layout()
plt.show()  # Muestra el gráfico
"""
st.code(codigo_python, language="python")
st.image("5grafico.jpg")

#
st.write("Graficos de barras Frecuencias Delito:")
codigo_python = """
# Filtrar los tipos de delito con una cantidad mayor a 400
frecuencias_filtradas = frecuencias_delito[frecuencias_delito > 400]

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))  # Tamaño del gráfico
frecuencias_filtradas.plot(kind='bar', color='red')
plt.title('Cantidad de Delitos por Tipo (Filtrado)')  # Título del gráfico
plt.xlabel('Tipo de Delito')  # Etiqueta del eje x
plt.ylabel('Cantidad')  # Etiqueta del eje y
plt.xticks(rotation=90)  # Rotar las etiquetas del eje x para mayor legibilidad
plt.tight_layout()

# Mostrar el gráfico
plt.show()
"""
st.code(codigo_python, language="python")
st.image("6grafico.png")
#

st.write("Graficos de Pastel: Frecuencia del Genero")
codigo_python = """
plt.figure(figsize=(5, 5))
plt.pie(frecuencias_genero, labels=frecuencias_genero.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Frecuencia GENERO')
plt.show()
"""
st.code(codigo_python, language="python")
st.image("7grafico.png")

st.write("Grafico pastel: Frecuencia de Edades")
codigo_python = """
# Crea el gráfico de pastel
plt.figure(figsize=(13, 12))
plt.pie(cantidadporrango, labels=cantidadporrango.index, autopct='%1.1f%%', pctdistance=0.8, startangle=140, labeldistance=1.05)
plt.axis('equal')
plt.title('Frecuencia de Edades')
plt.show()
"""
st.code(codigo_python, language="python")
st.image("8grafico.png")

st.write("Grafico pastel: Frecuencia de Horario")
codigo_python = """
# Crea el gráfico de pastel
plt.figure(figsize=(13, 12))
plt.pie(cantidadporrango, labels=cantidadporrango.index, autopct='%1.1f%%', pctdistance=0.8, startangle=140, labeldistance=1.05)
plt.axis('equal')
plt.title('Frecuencia de Edades')
plt.show()
"""
st.code(codigo_python, language="python")
st.image("8grafico.png")







# Conclusiones

# Agrega emojis y estilos de fuente personalizados
st.markdown("# 🚀 **Conclusión ** 🎨")

st.markdown("1. 🏙️ **El Centro es el Hotspot:** La mayor cantidad de robos ocurre en el corazón de la ciudad, posiblemente debido a su vibrante actividad comercial y la falta de presencia policial en áreas cercanas a los barrios residenciales.")

st.markdown("2. 🌟 **Estratégico y Vulnerable:** Se podría inferir que la concentración de robos en el centro se debe a su ubicación estratégica y a la proximidad de barrios con menor presencia policial. ¡Un desafío para la seguridad!")

st.markdown("3. 💼 **Delitos No Sexuales Dominan:** En el lado oscuro de la estadística, los delitos no sexuales superan en número a los delitos sexuales en la ciudad. ¿Cómo podemos abordar esta variabilidad en la seguridad?")

st.markdown("4. 🌞🌙 **Hora de la Delincuencia:** Los delitos matutinos y madrugadores tienden a tener horarios fijos, mientras que los delitos en la tarde y noche son más impredecibles. ¡La ciudad nunca duerme!")

st.markdown("5. 👶👴 **Edades y Delincuencia:** Los adultos son los más afectados por la delincuencia, mientras que los más pequeños (la primera infancia) experimentan menos problemas. ¡Protejamos a nuestros ciudadanos más jóvenes!")

st.markdown("6. 🏰 **Estrato vs. Delincuencia:** Sorprendentemente, incluso un barrio de alto estrato como Cabecera del Llano comparte índices de delincuencia similares a los de un barrio de estrato más bajo, como El Centro. ¿Dónde radica la igualdad?")

st.markdown("7. 🚶‍♀️ **Caminar con Cuidado:** Caminar por algunas partes de la ciudad puede ser arriesgado. ¡Mantén tus sentidos alerta y tu seguridad en mente!")

st.markdown("En resumen, estos hallazgos sugieren la necesidad de implementar **estrategias creativas y efectivas** para reducir la incidencia de robos, proteger a nuestros ciudadanos y mantener nuestra ciudad hermosa y segura. ¡Sigamos trabajando juntos para un futuro más seguro!")
# Barra de navegación
st.sidebar.title("Navegación")
pagina_actual = st.sidebar.radio("Selecciona una página:", ["Inicio", "Acerca de", "Contacto"])

if pagina_actual == "Inicio":
    st.sidebar.write("Bienvenido a la página de inicio.")
elif pagina_actual == "Acerca de":
    st.sidebar.write("Esta es la página de información acerca de la aplicación.")
elif pagina_actual == "Contacto":
    st.sidebar.write("Puedes ponerte en contacto con nosotros aquí.")
import pandas as pd
# Cargar el DataFrame
df = pd.read_csv('92._Delitos_en_Bucaramanga_enero_2016_a_julio_de_2023.csv')  # Reemplaza 'tu_archivo.csv' con la ruta a tu archivo CSV
info_df = pd.DataFrame({
    'Nombre de la columna': df.columns,
    'No. de valores no nulos': df.count().values,
    'Tipo de datos': df.dtypes.values
})

# Mostrar el resumen en Streamlit
st.title('Información del DataFrame')
st.write('A continuación se muestra la información del DataFrame que utilizamos:')
st.write(info_df)

# p r e p r o c e s a m i e n t o


st.write("## ¿Qué hacemos en el Preprocesamiento?")

# List of preprocessing steps
preprocessing_steps = [
    "Importa las bibliotecas necesarias",
    "Monta Google Drive para acceder a los archivos",
    "Carga un conjunto de datos desde Google Drive",
    "Realiza preprocesamiento de datos, como eliminación de columnas no deseadas y codificación de etiquetas",
    "Divide el conjunto de datos en conjuntos de entrenamiento y prueba",
    "Entrena varios modelos de aprendizaje automático, incluyendo Naive Bayes, Árbol de Decisión, Bosque Aleatorio y Máquinas de Vectores de Soporte (SVM)",
    "Evalúa la precisión y realiza predicciones en el conjunto de prueba"
]

# Display the list of preprocessing steps
st.write("A continuación, se enumeran los pasos que se realizan en el preprocesamiento de datos:")
for step in preprocessing_steps:
    st.write(f"- {step}")
codigo_python = """

"""

st.write("Division de datos de Prueba y Entrenamiento:")
codigo_python = """
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1000, shuffle=True)
"""

st.write("Seleccion de las mejores caracteristicas:")

st.write("Basado en una puntuacion que mide la relacion entre las caracteristicas y las etiquetas")

codigo_python = """

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1000, shuffle=True)
"""

st.write("# ¿Como funciona lo anterior?")

codigo_python = """
selector = SelectKBest(f_classif, k=5) # Escoge las 5 mejores caracteristicas
best_train = selector.fit_transform(X_train, y_train) # Se aplica a los datos de entrenamiento
best_test = selector.transform(X_test) #El selector de train se aplica a test
print(X_train.columns)
print(X_train.columns[selector.get_support()])  # Las columnas recomendadas
#Esto imprime las columnas recomendadas por SelectKBest. selector.get_support() devuelve un vector booleano que indica qué características fueron
# seleccionadas (True) y cuáles no (False). Usando esta información, se imprimen las columnas seleccionadas del conjunto de entrenamiento original.
"""


st.code(codigo_python, language="python")
st.write("# Matriz de confusion: Bosque Aleatorio")

codigo_python = """
selector = SelectKBest(f_classif, k=5) # Escoge las 5 mejores caracteristicas
best_train = selector.fit_transform(X_train, y_train) # Se aplica a los datos de entrenamiento
best_test = selector.transform(X_test) #El selector de train se aplica a test
print(X_train.columns)
print(X_train.columns[selector.get_support()])  # Las columnas recomendadas
#Esto imprime las columnas recomendadas por SelectKBest. selector.get_support() devuelve un vector booleano que indica qué características fueron
# seleccionadas (True) y cuáles no (False). Usando esta información, se imprimen las columnas seleccionadas del conjunto de entrenamiento original.
"""

st.write("# SVM: Maquina de Vectores de Soporte")
st.write("Nuestro Kernel es Lineal")
codigo_python = """
# Entrenar un modelo de Máquinas de Vectores de Soporte (SVM)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
modeloSvm = SVC(kernel='linear')
modeloSvm.fit(X_train, y_train)
"""
st.write("¿Como sabemos que si es un buen modelo?")
st.write("Comparamos la presicion del modelo SVM, tanto en el conjunto de entrenamiento como en el de la prueba")
codigo_python = """
# Entrenar un modelo de Máquinas de Vectores de Soporte (SVM)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
modeloSvm = SVC(kernel='linear')
modeloSvm.fit(X_train, y_train)
"""


st.code(codigo_python, language="python")
#Naive Bayes
st.write("Modelo: Naive Bayes")
st.write("La informacion la manejamos con Naive Bayes")
codigo_python = """

# Entrenar un modelo Naive Bayes
from sklearn.naive_bayes import GaussianNB

modeloNb = GaussianNB()
modeloNb.fit(X_train, y_train)
# Guardar el modelo Naive Bayes en un archivo binario
jb.dump(modeloNb, "/content/drive/MyDrive/Colab Notebooks/ModeloNB.bin", compress=True)
# Precisión en el conjunto de entrenamiento
modeloNb.score(X_train, y_train) * 100
# Precisión en el conjunto de prueba
modeloNb.score(X_test, y_test) * 100
# Predicción en el conjunto de prueba
y_predict = modeloNb.predict(X_test)
print(y_test.head(20))
print(pd.DataFrame(y_predict).head(20))
# Matriz de confusión para analizar los errores de predicción
matrix = confusion_matrix(y_test, y_predict, labels=modeloNb.classes_)
displaymatrix = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=modeloNb.classes_)
displaymatrix.plot(xticks_rotation='vertical')
# Reporte de clasificación
print(classification_report(y_test, y_predict))
"""


st.code(codigo_python, language="python")
#Arbol de Desción
st.write("Arbol de Desición:")
codigo_python = """
# Entrenar un modelo de Árbol de Decisión
from sklearn import tree
modeloArbol = tree.DecisionTreeClassifier()
modeloArbol.fit(X_train, y_train)
# Precisión en el conjunto de entrenamiento
modeloArbol.score(X_train, y_train) * 100
# Precisión en el conjunto de prueba
modeloArbol.score(X_test, y_test) * 100
# Predicción en el conjunto de prueba
y_predict = modeloArbol.predict(X_test)
print(y_test.head(20))
print(pd.DataFrame(y_predict).head(20))
# Matriz de confusión para analizar los errores de predicción
matrix = confusion_matrix(y_test, y_predict, labels=modeloArbol.classes_)
displaymatrix = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=modeloArbol.classes_)
displaymatrix.plot(xticks_rotation='vertical')
# Reporte de clasificación
print(classification_report(y_test, y_predict))

# Guardar el modelo de Árbol de Decisión en un archivo binario
jb.dump(modeloArbol, "/content/drive/MyDrive/Colab Notebooks/modeloArbol.bin", compress=True)
"""


st.code(codigo_python, language="python")
#Bosque Aleatorio
st.write("Bosque Aleatorio:")
codigo_python = """
# Entrenar un modelo de Bosque Aleatorio (Random Forest)
from sklearn.ensemble import RandomForestClassifier
modeloBA = RandomForestClassifier(random_state=0)
modeloBA.fit(X_train, y_train)
# Precisión en el conjunto de entrenamiento
modeloBA.score(X_train, y_train) * 100
# Precisión en el conjunto de prueba
modeloBA.score(X_test, y_test) * 100
# Predicción en el conjunto de prueba
y_predict = modeloBA.predict(X_test)
print(y_test.head(20))
print(pd.DataFrame(y_predict).head(20))
# Matriz de confusión para analizar los errores de predicción
matrix = confusion_matrix(y_test, y_predict, labels=modeloBA.classes_)
displaymatrix = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=modeloBA.classes_)
displaymatrix.plot(xticks_rotation='vertical')
# Reporte de clasificación
print(classification_report(y_test, y_predict))
# Guardar el modelo de Bosque Aleatorio en un archivo binario
jb.dump(modeloBA, "/content/drive/MyDrive/Colab Notebooks/modeloBA.bin", compress=True)
"""


st.code(codigo_python, language="python")




