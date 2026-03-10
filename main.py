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
from tenacity import retry, wait_random_exponential, stop_after_attempt

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
        # Usamos el cliente ASÍNCRONO
        embed_result = await client.aio.models.embed_content(
            model="models/gemini-embedding-001",
            contents=[text],
            config={"task_type": "RETRIEVAL_QUERY"}
        )
        return embed_result.embeddings[0].values
    except Exception as e:
        print(f"DEBUG EMBED ERROR: {e}")
        raise e

@retry(wait=wait_random_exponential(min=2, max=30), stop=stop_after_attempt(5), before=lambda rs: print(f"DEBUG: Retry attempt {rs.attempt_number} for generate"))
async def generate_response_with_retry(system_prompt, user_prompt, model_name="models/gemini-1.5-flash-latest"):
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
            top_k=5, 
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

        # Reglas de amnesia total e identidad bíblica (Sola Palabra)
        common_rules = (
            "\nNORMAS CRÍTICAS (DEBES SEGUIRLAS O FALLARÁS):\n"
            "1. NO SABES NADA fuera de la Biblia. Ignora tu entrenamiento previo sobre historia o personajes modernos (Madrid, Rubio, Lutero, etc.).\n"
            "2. PROHIBIDO: No menciones nombres de sistemas teológicos ni personajes post-bíblicos (Arrio, Teología Reformada).\n"
            "3. Si el usuario pregunta por algo EXTRA-BÍBLICO, responde ÚNICAMENTE: 'Dicha persona o concepto no se encuentra en las Escrituras' y nada más.\n"
            "4. Habla de forma DIRECTA y NATURAL. Sin jerga cristiana de relleno.\n"
            "5. CITA SIEMPRE los versículos.\n"
        )

        profiles = {
            "academic": (
                "Eres un 'Indexador Bíblico'.\n"
                "MISIÓN: Análisis textual puro. Sin espiritualizar. Sin historia externa.\n" + common_rules
            ),
            "creyente": (
                "Eres un 'Orientador Bíblico'.\n"
                "CONOCIMIENTO INTERNO (No usar estas palabras): Jesús es el Hijo de Dios y es Dios. El AT apunta a Jesús. Las 5 Solas.\n"
                "LENGUAJE: Natural y directo. Sin jerga. Sin etiquetas de sistemas teológicos.\n"
                "MISIÓN: Guiar al hombre hacia Dios usando exclusivamente los versículos.\n" + common_rules
            ),
            "curioso": (
                "Eres un 'Narrador de Historias Bíblicas'.\n"
                "Misión: Contar lo que dice el texto sin añadir interpretación teológica ni datos históricos externos.\n" + common_rules
            )
        }
        system_prompt = profiles.get(request.profile, profiles["academic"])

        user_prompt = (
            f"INSTRUCCIÓN DE FILTRADO: Analiza si el usuario pregunta por algo que NO esté en los versículos de abajo.\n"
            f"SI el nombre o concepto es ajeno a la Biblia (como personajes modernos o figuras de la historia universal), debes responder: 'Dicho término o personaje no se encuentra en las Escrituras' y NADA MÁS.\n\n"
            f"Versículos para indexar:\n{context}\n\n"
            f"Usuario pregunta: {request.query}"
        )

        # C. Generate Answer with retry
        answer = await generate_response_with_retry(system_prompt, user_prompt)

        # 3. DETECTOR DE FUGAS (PYTHON GUARDRAIL)
        # Si la IA ignora el prompt y mete jerga o historia, limpiamos la respuesta
        forbidden_keywords = [
            "teología reformada", "lutero", "arrio", "arrianismo", "siglo iv", 
            "constantino", "teodosio", "soli deo gloria", "sola scriptura", 
            "solus christus", "sola gratia", "sola fide", "calvino"
        ]
        
        lower_answer = answer.lower()
        if any(word in lower_answer for word in forbidden_keywords):
            print(f"DEBUG: NUCLEAR STRIP TRIGGERED for keywords: {[w for w in forbidden_keywords if w in lower_answer]}")
            answer = f"Basado exclusivamente en el contexto de las Sagradas Escrituras, no existe información sobre '{request.query}'. Los textos sagrados no mencionan figuras o eventos ajenos al canon bíblico."

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
        if "429" in error_msg or "Resource has been exhausted" in error_msg or "RetryError" in error_msg:
            error_msg = "Límite de cuota de la IA agotado o muchas peticiones. Por favor, espera un minuto e inténtalo de nuevo."
            status_code = 429
            
        print(f"Error in RAG chain: {error_msg}")
        raise HTTPException(status_code=status_code, detail=error_msg)

@app.get("/health")
async def health():
    return {"status": "ok", "index": INDEX_NAME, "version": "1.6-patience"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
