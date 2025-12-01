# CEIA_2025_B5_PNL2_TP2

Curso de Especialización en Inteligencia Artificial  
Año 2025  
Bimestre 5  

## Materia: Procesamiento del Lenguaje Natural 2  

## Docentes:
* Ezequiel Guinsburg
* Abraham Rodriguez

## Alumno:
* Viñas Gustavo

# TP2 - RAG Chatbot
Este proyecto tiene como objetivo implementar técnicas avanzadas en Retrieval-Augmented Generation (RAG). El objetivo principal es la creación de un proyecto práctico, de principio a fin donde se utiliza la librería LangChain <img src="img/langchain.png" alt="Description" width="20"/> y la base de datos vectorial Pinecone <img src="img/pinecone.png" alt="Description" width="15"/>, asegurando un enfoque de vanguardia para la gestión y recuperación de datos.

### Imagen del proyecto

 ![Project Logo](img/project_image.png)

## Tabla de contenido

1. [Diagrama](#diagrama)
2. [Preparación](#preparación)
    - [Pinecone](#Pinecone)
    - [Groq](#Groq)
    - [Langchain](#Langchain)
3. [Población inicial](#población-inicial-de-bases-de-datos-vectoriales)
4. [Retrieval-Augmented Generation (RAG)](#Retrieval-Augmented-Generation-(RAG))
5. [Aplicación web](#Aplicación-web)
6. [Despliegue](#Despliegue)

## Diagrama

Diagrama básico de funcionamiento:

 ![Diagrama](img/rag-diagram.png)
Imagen de <a href="https://huggingface.co/learn/cookbook/rag_zephyr_langchain" target="_blank">rag_zephyr_langchain</a>

## Preparación
Antes de sumergirnos en la codificación del proyecto, es esencial configurar las cuentas con Pinecone y Groq. Estas cuentas servirán como base para acceder a las herramientas y APIs necesarias requeridas para el desarrollo del proyecto. Al crear cuentas en estas plataformas, se obtiene acceso a potentes tecnologías que le permitirán aprovechar capacidades de vanguardia en la gestión de bases de datos vectoriales, el procesamiento del lenguaje natural y la generación impulsada por IA.

### Pinecone
Pinecone proporciona una solución de base de datos vectorial escalable, que permite almacenar y consultar datos vectoriales de alta dimensión de manera eficiente. Al crear una cuenta de Pinecone, puede configurar su base de datos vectorial, establecer parámetros de indexación e integrarla sin problemas en el flujo de trabajo de su proyecto.

Como mínimo se deben configurar las siguientes variables de entorno:  
`PINECONE_API_KEY`  
Para la carga inicial, adicionalmente se pueden configurar:  
`PINECONE_CLOUD` (default: 'aws')  
`PINECONE_REGION` (default: 'us-east-1')  
`PINECONE_INDEX_NAME` (default: 'ceia-2025-b5-pnl2-tp2')
`PINECONE_RECREATE_INDEX` (default: False)  
Para el chatbot, adicionalmente se puede configurar:  
`PINECONE_INDEX_NAME` (default: 'ceia-2025-b5-pnl2-tp2')

### Groq
ChatGroq es la interfaz de chatbot y plataforma API de Groq que permite a los usuarios y desarrolladores acceder y utilizar los Modelos de Lenguaje Grande (LLMs) compatibles con su LPU (Unidad de Procesamiento de Lenguaje) para obtener respuestas con una latencia extremadamente baja (casi instantánea).  
Como mínimo debes tener configurado:  
`GROQ_API_KEY`

### Langchain
LangChain proporciona una biblioteca completa de herramientas y recursos para implementar técnicas avanzadas de procesamiento del lenguaje.  
Para este proyecto, no es necesario obtener una cuenta de LangChain.

## Población inicial de Bases de Datos Vectoriales

Para poblar la Base de Datos de Pinecone, tienes que ejecutar este notebook: [carga_inicial.ipynb](carga_inicial.ipynb).  
Se procesarán y cargaran los documentos existentes en la carpeta definida por la variable de entorno `DOCS_DIR` (default: `./docs`).

## Retrieval-Augmented Generation (RAG)

El código en el archivo [chatbot.py](chatbot.py) implementa RAG (Retrieval-Augmented Generation) en la versión más reciente de LangChain.

## Aplicación web

La aplicación de Streamlit se instancia en el archivo [application.py](application.py).  
El contenido es generado dinamicamente y esta embebido en el código fuente.

## Despliegue
Los requisitos para desplegar el proyecto se encuentran en el archivo [requirements.txt](requirements.txt).  
Se pueden instalar con el comando `pip install -r requirements.txt`.  
Adicionalmente el archivo [pyproject.toml](pyproject.toml) contiene la configuración del proyecto que permite instalar las dependencias con <a href="https://docs.astral.sh/uv/" target="_blank">uv</a> mediante el comando `uv sync`.

Para iniciar el chatbot ejecutar `streamlit run chatbot.py`.

## Demostración
<a href="https://youtu.be/msadd7KkFsQ" target="_blank">Video demostración</a>