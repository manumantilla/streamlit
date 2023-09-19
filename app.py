import streamlit as st

# Título
st.title("Mi Aplicación Streamlit")

# Introducción
st.write("""
Esta es una aplicación simple de Streamlit que muestra cómo crear una página básica con Streamlit.
""")

# Imagen
st.image("https://via.placeholder.com/300", caption="Una imagen de ejemplo", use_column_width=True)

# Conclusiones
st.write("""
**Conclusión:**

Espero que esta aplicación de ejemplo te haya ayudado a comprender cómo crear una aplicación básica con Streamlit.
""")

# Barra de navegación
st.sidebar.title("Navegación")
pagina_actual = st.sidebar.radio("Selecciona una página:", ["Inicio", "Acerca de", "Contacto"])

if pagina_actual == "Inicio":
    st.sidebar.write("Bienvenido a la página de inicio.")
elif pagina_actual == "Acerca de":
    st.sidebar.write("Esta es la página de información acerca de la aplicación.")
elif pagina_actual == "Contacto":
    st.sidebar.write("Puedes ponerte en contacto con nosotros aquí.")
