"""
Chatbot con Memoria Persistente usando LangChain y Groq
=====================================================

Este archivo implementa un chatbot con interfaz web usando Streamlit que:
- Mantiene memoria de conversaciones anteriores
- Utiliza diferentes modelos de LLM a través de Groq
- Permite personalización del comportamiento del bot
- Gestiona la memoria conversacional automáticamente

Tecnologías utilizadas:
- Streamlit: Para la interfaz web
- LangChain: Para gestión de memoria y cadenas de conversación
- Groq: Como proveedor de modelos LLM
- Python: Lenguaje de programación

Autor: Clase VI - CEIA LLMIAG
Curso: Large Language Models y Generative AI

Instrucciones para ejecutar:
    streamlit run chatbot_gestionada.py

Requisitos:
    pip install streamlit groq langchain langchain-groq

Variables de entorno necesarias:
    GROQ_API_KEY: Tu clave API de Groq (obtener en https://console.groq.com)
"""

# ========================================
# IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# ========================================

import streamlit as st           # Framework para crear aplicaciones web interactivas
import os                      # Para acceso a variables de entorno
from random import randint

# Importaciones específicas de LangChain para gestión de conversaciones
from langchain_core.prompts import (
    ChatPromptTemplate,           # Template para estructurar mensajes de chat
    HumanMessagePromptTemplate,   # Template específico para mensajes humanos
    MessagesPlaceholder,          # Marcador de posición para el historial
    SystemMessagePromptTemplate,  # Template para mensajes del sistema
)
from langchain_classic.chains.conversation.memory import ConversationBufferWindowMemory  # Memoria de ventana deslizante
from langchain_groq import ChatGroq              # Integración LangChain-Groq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

def main():
    """
    Función principal de la aplicación de chatbot.
    
    Esta función coordina todos los componentes del chatbot:
    1. Configuración de la interfaz de usuario
    2. Gestión de la memoria conversacional
    3. Integración con el modelo de lenguaje
    4. Procesamiento de preguntas y respuestas
    
    Funcionalidades principales:
    - Interfaz web responsiva con Streamlit
    - Memoria de conversación con longitud configurable
    - Selección de diferentes modelos LLM
    - Personalización del prompt del sistema
    - Historial persistente durante la sesión
    """
    
    # ========================================
    # CONFIGURACIÓN INICIAL Y AUTENTICACIÓN
    # ========================================
    
    # Obtener la clave API de Groq desde las variables de entorno
    # Esto es una práctica de seguridad recomendada para no exponer credenciales en el código
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    # Verificar si la clave API está configurada
    if not groq_api_key:
        st.error("⚠️ GROQ_API_KEY no está configurada en las variables de entorno")
        st.info("💡 Configura tu clave API: export GROQ_API_KEY='tu-clave-aqui'")
        st.stop()  # Detener la ejecución si no hay clave API

    PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        st.error("⚠️ PINECONE_API_KEY no está configurada en las variables de entorno")
        st.info("💡 Configura tu clave API: export PINECONE_API_KEY='tu-clave-aqui'")
        st.stop()  # Detener la ejecución si no hay clave API

    index_name = os.environ.get('PINECONE_INDEX_NAME') or 'ceia-2025-b5-pnl2-tp2'
    namespace = "documentos"

    ### EMBEDDINGS
    embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    vectorstore = PineconeVectorStore(
        pinecone_api_key=PINECONE_API_KEY,
        index_name=index_name,
        embedding=embedding_model,
        namespace=namespace,
    )
    retriever=vectorstore.as_retriever()

    # ========================================
    # CONFIGURACIÓN DE LA INTERFAZ PRINCIPAL
    # ========================================
    
    # Configurar el título y descripción de la aplicación
    st.title("🤖 Chatbot CEIA con Memoria Persistente")
    st.markdown("""
    **¡Bienvenido al chatbot educativo!** 
    
    Este chatbot utiliza:
    - 🧠 **Memoria conversacional**: Recuerda el contexto de tu conversación
    - 🔄 **Modelos intercambiables**: Puedes elegir diferentes LLMs
    - ⚙️ **Personalización**: Configura el comportamiento del asistente
    - 🚀 **Powered by Groq**: Respuestas rápidas y precisas
    """)

    # ========================================
    # PANEL DE CONFIGURACIÓN LATERAL
    # ========================================
    
    st.sidebar.title('⚙️ Configuración del Chatbot')
    st.sidebar.markdown("---")
    
    # Input para el prompt del sistema - Define la personalidad y comportamiento del bot
    st.sidebar.subheader("🎭 Personalidad del Bot")
    system_prompt = st.sidebar.text_area(
        "Mensaje del sistema:",
        value="Eres un bot que responde preguntas sobre documentos proporcionados. "
              "Usa únicamente el contexto dado para responder."
              "Si la respuesta no está en el contexto, di: "
              "'No te puedo proporcionar la información, ya que no existe en mi base de datos.'"
              "Sé preciso y conciso.",
        height=100,
        help="Define cómo debe comportarse el chatbot."
    )

    model = "llama-3.1-8b-instant"
    st.sidebar.info(f"Modelo {model}")
    
    # Control deslizante para la longitud de memoria
    st.sidebar.subheader("🧠 Configuración de Memoria")
    conversational_memory_length = st.sidebar.slider(
        'Longitud de la memoria conversacional:', 
        min_value=1, 
        max_value=10, 
        value=5,
        help="Número de intercambios anteriores que el bot recordará. Más memoria = mayor contexto pero mayor costo computacional"
    )
    
    # Mostrar información sobre la memoria
    st.sidebar.caption(f"💭 El bot recordará los últimos {conversational_memory_length} intercambios")

    # ========================================
    # CONFIGURACIÓN DE LA MEMORIA CONVERSACIONAL
    # ========================================
    
    # Crear objeto de memoria con ventana deslizante
    # ConversationBufferWindowMemory mantiene solo los últimos k intercambios
    memory = ConversationBufferWindowMemory(
        k=conversational_memory_length,        # Número de intercambios a recordar
        memory_key="historial_chat",           # Clave para acceder al historial
        return_messages=True                   # Devolver mensajes en formato estructurado
    )
    
    # ========================================
    # GESTIÓN DEL HISTORIAL DE CONVERSACIÓN
    # ========================================
    
    # Inicializar el historial de chat en el estado de la sesión de Streamlit
    # st.session_state permite mantener datos entre ejecuciones de la aplicación
    if 'historial_chat' not in st.session_state:
        st.session_state.historial_chat = []
        st.sidebar.success("💬 Nueva conversación iniciada")
    else:
        # Si ya existe historial, cargarlo en la memoria de LangChain
        for message in st.session_state.historial_chat:
            memory.save_context(
                {'input': message['humano']},      # Mensaje del usuario
                {'output': message['IA']}          # Respuesta del chatbot
            )
        
        # Mostrar información del historial en la barra lateral
        st.sidebar.info(f"💬 Conversación con {len(st.session_state.historial_chat)} mensajes")
    
    # Botón para limpiar el historial
    if st.sidebar.button("🗑️ Limpiar Conversación"):
        st.session_state.historial_chat = []
        st.sidebar.success("✅ Conversación limpiada")
        st.rerun()  # Recargar la aplicación
    
    # ========================================
    # INTERFAZ DE ENTRADA DEL USUARIO
    # ========================================
    
    # Crear el campo de entrada para las preguntas del usuario
    st.markdown("### 💬 Haz tu pregunta:")
    user_question = st.text_input(
        "Escribe tu mensaje aquí:",
        placeholder="Por ejemplo: Que habilidades tiene Rob Otto?",
        label_visibility="collapsed"
    )


    # ========================================
    # CONFIGURACIÓN DEL MODELO DE LENGUAJE
    # ========================================
    
    # Inicializar el cliente de ChatGroq con las configuraciones seleccionadas
    try:
        groq_chat = ChatGroq(
            groq_api_key=groq_api_key,     # Clave API para autenticación
            model_name=model,              # Modelo seleccionado por el usuario
            temperature=0.7,               # Creatividad de las respuestas (0=determinista, 1=creativo)
            max_tokens=1000,               # Máximo número de tokens en la respuesta
        )
        st.sidebar.success("✅ Modelo conectado correctamente")
    except Exception as e:
        st.sidebar.error(f"❌ Error al conectar con Groq: {str(e)}")
        st.stop()

    # ========================================
    # PROCESAMIENTO DE LA PREGUNTA Y RESPUESTA
    # ========================================

    # Si el usuario ha hecho una pregunta,
    if user_question and user_question.strip():

        # Mostrar indicador de carga mientras se procesa
        with st.spinner('🤔 El chatbot está pensando...'):
            
            try:
                # ========================================
                # CONSTRUCCIÓN DEL TEMPLATE DE CONVERSACIÓN
                # ========================================
                
                # Crear un template de chat que incluye:
                # 1. Mensaje del sistema (personalidad/instrucciones)
                # 2. Historial de conversación (memoria)
                # 3. Mensaje actual del usuario
                prompt = ChatPromptTemplate.from_messages([
                    
                    # Mensaje del sistema - Define el comportamiento del chatbot
                    SystemMessagePromptTemplate.from_template(system_prompt+"\n\nContexto: {context}"),
                    
                    # Marcador de posición para el historial - Se reemplaza automáticamente
                    MessagesPlaceholder(variable_name="historial_chat"),
                    
                    # Template para el mensaje actual del usuario
                    HumanMessagePromptTemplate.from_template("{input}")
                ])
                
                # ========================================
                # CREACIÓN DE LA CADENA DE CONVERSACIÓN
                # ========================================
                question_answer_chain = create_stuff_documents_chain(groq_chat, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                def get_session_history(session_id: str):
                    # In a real application, this would fetch history from a persistent store
                    print(memory.chat_memory)
                    return memory.chat_memory # For this example, we'll return the same memory instance

                conversational_rag_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="historial_chat",
                    output_messages_key="answer",
                )

                # ========================================
                # GENERACIÓN DE LA RESPUESTA
                # ========================================
                
                response = conversational_rag_chain.invoke(
                    {"input": user_question},
                    config={
                        "configurable": {"session_id": "user1_session1" }
                    },  # constructs a key "abc123" in `store`.
                )["answer"]
                # ========================================
                # ALMACENAMIENTO Y VISUALIZACIÓN
                # ========================================
                
                # Crear un objeto mensaje para almacenar en el historial
                message = {'humano': user_question, 'IA': response}
                
                # Agregar el mensaje al historial de la sesión
                st.session_state.historial_chat.append(message)
                
                # ========================================
                # MOSTRAR LA CONVERSACIÓN
                # ========================================
                
                # Mostrar la respuesta actual destacada
                st.markdown("### 🤖 Respuesta:")
                st.markdown(f"""
                <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 4px solid #1f77b4;">
                    {response}
                </div>
                """, unsafe_allow_html=True)
                
                # Información adicional sobre la respuesta
                st.caption(f"📊 Modelo: {model} | 🧠 Memoria: {conversational_memory_length} mensajes")
                
            except Exception as e:
                raise e
                # Manejo de errores durante el procesamiento
                st.error(f"❌ Error al procesar la pregunta: {str(e)}")
                st.info("💡 Verifica tu conexión a internet y la configuración de la API")


    # ========================================
    # INFORMACIÓN ADICIONAL PARA ESTUDIANTES
    # ========================================
    
    # Panel expandible con información educativa
    with st.expander("📚 Información Técnica para Estudiantes"):
        st.markdown("""
        **¿Cómo funciona este chatbot?**
        
        1. **Memoria Conversacional**: Utiliza `ConversationBufferWindowMemory` para recordar contexto
        2. **Templates de Prompts**: Estructura los mensajes de manera consistente
        3. **Cadenas LLM**: `LLMChain` conecta el modelo con la lógica de conversación
        4. **Estado de Sesión**: Streamlit mantiene el historial durante la sesión
        5. **Integración Groq**: Acceso rápido a modelos de lenguaje modernos
        
        **Conceptos Clave:**
        - **System Prompt**: Define la personalidad del chatbot
        - **Memory Window**: Controla cuánto contexto previo se incluye
        - **Token Limits**: Gestiona el costo y velocidad de las respuestas
        - **Model Selection**: Diferentes modelos para diferentes necesidades
        
        **Arquitectura del Sistema:**
        ```
        Usuario → Streamlit → LangChain → Groq → LLM → Respuesta
                     ↓
               Session State (Memoria)
        ```
        """)
    
    # Pie de página con información del curso
    st.markdown("---")
    st.markdown("**📖 Clase VI - CEIA LLMIAG** | Ejemplo educativo de chatbot con memoria persistente")


if __name__ == "__main__":
    # Punto de entrada de la aplicación
    # Solo ejecutar main() si este archivo se ejecuta directamente
    main()
