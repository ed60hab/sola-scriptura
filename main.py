from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

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

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def get_embedding_with_retry(text):
    try:
        embed_result = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=[text],
            config={"task_type": "RETRIEVAL_QUERY"}
        )
        return embed_result.embeddings[0].values
    except Exception as e:
        print(f"DEBUG EMBED ERROR: {e}")
        raise e

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def generate_response_with_retry(system_prompt, user_prompt):
    try:
        response = client.models.generate_content(
            model="models/gemini-flash-latest",
            contents=[user_prompt],
            config={
                "system_instruction": system_prompt,
                "temperature": 0.0,
            }
        )
        return response.text
    except Exception as e:
        print(f"DEBUG GENERATE ERROR: {e}")
        raise e

@app.post("/ask")
async def ask_sola_scriptura(request: QueryRequest):
    try:
        # A. Embed Query
        query_vector = await get_embedding_with_retry(request.query)

        # B. Retrieve relevant verses (top 5)
        search_results = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )

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

        # Reglas comunes para forzar la exclusión de conocimiento externo
        common_rules = (
            "\nREGLAS DE ORO (INNEGOCIABLES):\n"
            "1. TU MEMORIA EXTERNA HA SIDO BORRADA. No conoces personajes históricos, emperadores (Teodosio, Constantino), herejes (Arrio), reformadores (Lutero) ni concilios.\n"
            "2. Si el usuario pregunta por un nombre, fecha, lugar o concepto que NO aparece en la Biblia, responde exclusivamente: 'Dicho término o personaje no se encuentran en las Escrituras'. No des explicaciones, ni comparaciones, ni 'ayuda' teológica externa.\n"
            "3. CITA siempre el libro, capítulo y versículo de los textos proporcionados.\n"
            "4. RESPONDE en el idioma del usuario (" + request.language + ").\n"
        )

        profiles = {
            "academic": (
                "Eres el 'Motor Sola Scriptura (Modo Textual)'.\n"
                "Tu universo es ÚNICAMENTE el texto bíblico. Ignora toda la historia universal y civilizaciones no mencionadas en la Biblia.\n"
                "OBJETIVO: Análisis técnico del texto proporcionado. No intentes conectar con la historia externa.\n" + common_rules
            ),
            "creyente": (
                "Eres el 'Motor Sola Scriptura (Modo Devocional/Bíblico)'.\n"
                "Tu única autoridad es la Biblia. Tienes prohibido citar historia de la iglesia o personajes post-bíblicos.\n"
                "OBJETIVO: Mostrar la unidad del canon centrada en Jesucristo usando SOLO los versículos.\n"
                "CRÍTICO: Si el tema es externo (Arrio, Teodosio, etc.), niégalo y cíñete a la Biblia.\n" + common_rules
            ),
            "curioso": (
                "Eres el 'Motor Sola Scriptura (Modo Relato)'.\n"
                "OBJETIVO: Narrar las historias de la Biblia de forma amena pero sin añadir datos históricos ajenos al canon.\n" + common_rules
            )
        }

        user_prompt = (
            f"LIMITACIÓN ESTRICTA: Responde SOLO usando la siguiente información de la Biblia.\n"
            f"REGLA DE RECHAZO: Si la pregunta incluye nombres o términos ajenos a estos versículos (como Teodosio, Arrio, concilios o eventos modernos), responde que dicho término no se encuentra en las Escrituras y NO lo definas.\n\n"
            f"Versículos:\n{context}\n\n"
            f"Pregunta del usuario: {request.query}"
        )

        # C. Generate Answer with retry
        answer = await generate_response_with_retry(system_prompt, user_prompt)

        return {
            "answer": answer,
            "references": references
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in RAG chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "index": INDEX_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
