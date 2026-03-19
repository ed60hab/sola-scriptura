from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import asyncio
import time
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone
from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError

# 1. Load config
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sola-scriptura-bible")

# 2. Initialize Clients
client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

app = FastAPI(title="Sola Scriptura RAG API")

# 3. CORS for Frontend (port 3001)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For pilot, can be restricted later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    language: str = "es"
    profile: str = "academic" # academic, devotional, pastoral

@retry(wait=wait_random_exponential(min=2, max=30), stop=stop_after_attempt(5), before=lambda rs: print(f"DEBUG: Retry attempt {rs.attempt_number} for embed"))
async def get_embedding_with_retry(text):
    try:
        # Usamos el modelo de embedding verificado por el sistema
        embed_result = await client.aio.models.embed_content(
            model="models/gemini-embedding-001",
            contents=[text],
            config={"task_type": "RETRIEVAL_QUERY"}
        )
        return embed_result.embeddings[0].values
    except Exception as e:
        print(f"DEBUG EMBED ERROR: {e}")
        raise e

@retry(wait=wait_random_exponential(min=2, max=20), stop=stop_after_attempt(5), before=lambda rs: print(f"DEBUG: Retry attempt {rs.attempt_number} for generate"))
async def generate_response_with_retry(system_prompt, user_prompt, model_name="models/gemini-2.5-flash"):
    try:
        # Usamos el cliente ASÍNCRONO
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=[user_prompt],
            config={
                "system_instruction": system_prompt,
                "temperature": 0.0,
            }
        )
        return response.text
    except Exception as e:
        print(f"DEBUG GENERATE ERROR ({model_name}): {e}")
        raise e

@app.post("/ask")
async def ask_sola_scriptura(request: QueryRequest):
    try:
        start_time = time.time()

        # 1. BÚSQUEDA VECTORIAL (Paralela a nada, pero asíncrona)
        # Consolidamos el guardarraíl en el prompt principal para ahorrar 1 llamada a la API
        # y reducir la latencia a la mitad.
        
        embed_start = time.time()
        query_vector = await get_embedding_with_retry(request.query)
        embed_time = time.time() - embed_start

        search_start = time.time()
        search_results = await asyncio.to_thread(
            index.query, 
            vector=query_vector, 
            top_k=20, 
            include_metadata=True
        )
        search_time = time.time() - search_start
        print(f"DEBUG: Search completed in {search_time:.2f}s")

        # 2. PROCESO DE CONTEXTO
        context = ""
        references = []
        for match in search_results["matches"]:
            meta = match["metadata"]
            # Robust metadata extraction
            verse_text = meta.get('full_context', meta.get('text', ''))
            simple_text = meta.get('text', verse_text[:500])
            
            context += f"[{meta.get('version', '???')}] {meta.get('book', '???')} {meta.get('chapter', 0)}:{meta.get('verse', 0)}: {verse_text}\n\n"
            references.append({
                "book": meta.get('book', 'Unknown'),
                "chapter": meta.get('chapter', 0),
                "verse": meta.get('verse', 0),
                "text": simple_text,
                "version": meta.get('version', 'Unknown')
            })

        # --- LOCALIZATION (I18N) ---
        lang = request.language.lower() if request.language else "es"
        if lang not in ["es", "en"]:
            lang = "es"

        i18n = {
            "es": {
                "common_rules": (
                    "\nNORMAS CRÍTICAS (DEBES SEGUIRLAS O FALLARÁS):\n"
                    "1. NO SABES NADA fuera de la Biblia. Ignora tu entrenamiento previo sobre historia o personajes modernos (Madrid, Rubio, Lutero, etc.).\n"
                    "2. PROHIBIDO: No menciones nombres de sistemas teológicos ni personajes post-bíblicos (Arrio, Teología Reformada).\n"
                    "3. Si el usuario pregunta por algo EXTRA-BÍBLICO, responde ÚNICAMENTE: 'Dicha persona o concepto no se encuentra en las Escrituras' y nada más.\n"
                    "4. Habla de forma DIRECTA y NATURAL. Sin jerga cristiana de relleno.\n"
                    "5. CITA SIEMPRE los versículos.\n"
                ),
                "profiles": {
                    "academic": "Eres un 'Indexador Bíblico'.\nMISIÓN: Análisis textual puro. Sin espiritualizar. Sin historia externa.\n",
                    "creyente": (
                        "Eres un 'Orientador y Analista Bíblico'.\n"
                        "MISIÓN: Dar respuestas amplias, estructuradas y orientativas al usuario. Limítate estrictamente al testimonio directo de las Escrituras y EVITA a toda costa mencionar o depender de sistemas teológicos posteriores.\n"
                        "MÉTODO DE INTERPRETACIÓN: Para responder, debes reunir primero todos los textos bíblicos claros sobre el tema y luego interpretar los pasajes difíciles a la luz de los claros. Demuestra la armonía de las Escrituras.\n"
                        "ESTRUCTURA DE RESPUESTA:\n"
                        "- Comienza con una respuesta directa que resuma el tema.\n"
                        "- Desarrolla un razonamiento bíblico estructurado por puntos numerados, citando las Escrituras y explicando su sentido (ej. '1. Jesús afirma la seguridad eterna (Juan 10:27-29)... 2. La salvación es obra de Dios (Fil. 1:6)...').\n"
                        "- Trata los textos de advertencia o pasajes difíciles explicándolos a la luz de los claros.\n"
                        "- Concluye de manera concisa resumiendo el consejo de toda la Escritura.\n"
                        "- Al final de tu respuesta, SIEMPRE ofrece opciones para explorar temas adicionales profundos que suelen pasarse por alto (ej. 'Además, podría explicarte la diferencia bíblica entre creer de verdad y la fe temporal...').\n"
                        "LENGUAJE: Natural, analítico, exhaustivo, profundo y respetuoso. Sin jerga moderna.\n"
                    ),
                    "curioso": "Eres un 'Narrador de Historias Bíblicas'.\nMisión: Contar lo que dice el texto sin añadir interpretación teológica ni datos históricos externos.\n"
                },
                "user_prompt_header": (
                    f"INSTRUCCIÓN DE FILTRADO: Analiza si el usuario pregunta por algo que NO esté en los versículos de abajo.\n"
                    f"SI el nombre o concepto es ajeno a la Biblia (como personajes modernos o figuras de la historia universal), debes responder: 'Dicho término o personaje no se encuentra en las Escrituras' y NADA MÁS.\n\n"
                    f"Versículos para indexar:\n"
                ),
                "user_prompt_query": "\nUsuario pregunta: ",
                "forbidden_keywords": [
                    "teología reformada", "lutero", "arrio", "arrianismo", "siglo iv", 
                    "constantino", "teodosio", "soli deo gloria", "sola scriptura", 
                    "solus christus", "sola gratia", "sola fide", "calvino"
                ],
                "nuclear_strip_response": f"Basado exclusivamente en el contexto de las Sagradas Escrituras, no existe información sobre '{request.query}'. Los textos sagrados no mencionan figuras o eventos ajenos al canon bíblico."
            },
            "en": {
                "common_rules": (
                    "\nCRITICAL RULES (YOU MUST FOLLOW THEM OR YOU WILL FAIL):\n"
                    "1. YOU KNOW NOTHING outside of the Bible. Ignore your previous training about history or modern characters (Madrid, Rubio, Luther, etc.).\n"
                    "2. FORBIDDEN: Do not mention names of theological systems or post-biblical characters (Arius, Reformed Theology).\n"
                    "3. If the user asks about something EXTRA-BIBLICAL, respond ONLY: 'Such person or concept is not found in the Scriptures' and nothing else.\n"
                    "4. Speak DIRECTLY and NATURALLY. No filler Christian jargon.\n"
                    "5. ALWAYS CITE the verses.\n"
                ),
                "profiles": {
                    "academic": "You are a 'Biblical Indexer'.\nMISSION: Pure textual analysis. No spiritualizing. No external history.\n",
                    "creyente": (
                        "You are a 'Biblical Counselor and Analyst'.\n"
                        "MISSION: Provide broad, structured, and guiding responses to the user. Limit yourself strictly to the direct testimony of the Scriptures and AVOID at all costs mentioning or depending on later theological systems.\n"
                        "INTERPRETATION METHOD: To respond, you must first gather all clear biblical texts on the subject and then interpret difficult passages in light of the clear ones. Demonstrate the harmony of the Scriptures.\n"
                        "RESPONSE STRUCTURE:\n"
                        "- Start with a direct answer that summarizes the topic.\n"
                        "- Develop a structured biblical reasoning by numbered points, citing the Scriptures and explaining their meaning (e.g., '1. Jesus affirms eternal security (John 10:27-29)... 2. Salvation is the work of God (Phil. 1:6)...').\n"
                        "- Deal with warning texts or difficult passages by explaining them in light of the clear ones.\n"
                        "- Conclude concisely by summarizing the counsel of all Scripture.\n"
                        "- At the end of your response, ALWAYS offer options to explore additional deep topics that are often overlooked (e.g., 'Furthermore, I could explain the biblical difference between true believing and temporary faith...').\n"
                        "LENGUAJE: Natural, analytical, exhaustive, deep, and respectful. No modern jargon.\n"
                    ),
                    "curioso": "You are a 'Biblical Storyteller'.\nMission: Tell what the text says without adding theological interpretation or external historical data.\n"
                },
                "user_prompt_header": (
                    f"FILTERING INSTRUCTION: Analyze if the user is asking for something NOT in the verses below.\n"
                    f"IF the name or concept is foreign to the Bible (such as modern characters or figures from universal history), you must respond: 'Such person or concept is not found in the Scriptures' and NOTHING ELSE.\n\n"
                    f"Verses to index:\n"
                ),
                "user_prompt_query": "\nUser asks: ",
                "forbidden_keywords": [
                    "reformed theology", "luther", "arius", "arianism", "4th century", 
                    "constantine", "theodosius", "soli deo gloria", "sola scriptura", 
                    "solus christus", "sola gratia", "sola fide", "calvin"
                ],
                "nuclear_strip_response": f"Based exclusively on the context of the Holy Scriptures, there is no information about '{request.query}'. The sacred texts do not mention figures or events foreign to the biblical canon."
            }
        }

        lang_cfg = i18n[lang]
        common_rules = lang_cfg["common_rules"]
        
        system_prompt = lang_cfg["profiles"].get(request.profile, lang_cfg["profiles"]["academic"]) + common_rules

        user_prompt = (
            lang_cfg["user_prompt_header"] + 
            context + 
            lang_cfg["user_prompt_query"] + 
            request.query
        )
        
        print(f"DEBUG: Context length: {len(context)} chars. Query: {request.query}")

        # C. Generate Answer with retry
        answer = await generate_response_with_retry(system_prompt, user_prompt)

        # 3. DETECTOR DE FUGAS (PYTHON GUARDRAIL)
        # Si la IA ignora el prompt y mete jerga o historia, limpiamos la respuesta
        forbidden_keywords = lang_cfg["forbidden_keywords"]
        
        lower_answer = answer.lower()
        if any(word in lower_answer for word in forbidden_keywords):
            print(f"DEBUG: NUCLEAR STRIP TRIGGERED for keywords: {[w for w in forbidden_keywords if w in lower_answer]}")
            answer = lang_cfg["nuclear_strip_response"]

        return {
            "answer": answer,
            "references": references,
            "metrics": {
                "total_time": round(time.time() - start_time, 2),
                "embed_time": round(embed_time, 2),
                "search_time": round(search_time, 2)
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e)
        status_code = 500
        
        # Detectar errores de cuota o reintentos agotados
        if isinstance(e, RetryError):
            last_err = e.last_attempt.exception()
            error_msg = f"Reintentos agotados tras 5 intentos. Causa última: {last_err}"
            status_code = 429 if "429" in str(last_err) else 500
        elif "429" in error_msg or "Resource has been exhausted" in error_msg:
            error_msg = f"La IA está saturada (Cuota Google Agotada). Info: {error_msg[:100]}"
            status_code = 429
            
        print(f"Error in RAG chain: {error_msg}")
        raise HTTPException(status_code=status_code, detail=error_msg)

@app.get("/health")
async def health():
    return {"status": "ok", "index": INDEX_NAME, "version": "3.1-analytical"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
