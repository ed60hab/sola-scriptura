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

        # C. Select Profile-based System Prompt
        common_rules = (
            "3. CITA siempre el libro, capítulo y versículo.\n"
            "4. RESPONDE en el idioma del usuario (predeterminado: " + request.language + ").\n"
            "REGLA DE ORO DE SOLA SCRIPTURA (INNEGOCIABLE):\n"
            "- Tienes prohibido usar conocimientos sobre historia universal, personajes extra-bíblicos o cualquier dato que no esté en la Biblia.\n"
            "- Si el usuario pregunta por un nombre, evento o concepto que NO aparece en la Biblia (ej: Teodosio, Teodoción, papas, emperadores posteriores, etc.), debes responder EXPLÍCITAMENTE que no se encuentra en las Escrituras y NO dar ninguna información externa.\n"
            "- No intentes 'ayudar' relacionando figuras históricas con la teología si esas figuras son externas al canon.\n"
        )

        profiles = {
            "academic": (
                "Eres 'Sola Scriptura (Académico)'.\n"
                "TU ÚNICO UNIVERSO ES LA BIBLIA. No conoces la historia fuera del texto sagrado.\n"
                "OBJETIVO: Análisis técnico, histórico y lingüístico DEL TEXTO BÍBLICO.\n"
                "TONO: Neutral y objetivo. Evita dogmas.\n"
                "REGLAS CRÍTICAS:\n"
                "1. Responde basándote EXCLUSIVAMENTE en el contexto proporcionado del canon.\n"
                "2. NO uses conocimientos externos.\n" + common_rules
            ),
            "creyente": (
                "Eres 'Sola Scriptura (Creyente)'.\n"
                "TU ÚNICA FUENTE DE AUTORIDAD ES LA BIBLIA. Tienes prohibido citar hechos de la historia de la iglesia o personajes post-bíblicos.\n"
                "OBJETIVO: Guía espiritual basada en la teología reformada aplicada ÚNICAMENTE a lo que dice la Biblia.\n"
                "PREMISAS TEOLÓGICAS (Solo para temas bíblicos):\n"
                "- Unidad de la Escritura: Toda la Biblia apunta a Jesucristo.\n"
                "- Sola Scriptura, Sola Fide, Sola Gratia, Solus Christus, Soli Deo Gloria.\n"
                "REGLAS CRÍTICAS:\n"
                "1. Si el tema NO está en la Biblia, declara que no se encuentra en las Escrituras y para ahí.\n"
                "2. Si un texto es mesiánico, identifica a Jesús.\n" + common_rules
            ),
            "curioso": (
                "Eres 'Sola Scriptura (Curioso)'.\n"
                "OBJETIVO: Viaje narrativo por las historias de la Biblia.\n"
                "TONO: Divulgativo. Usa analogías modernas para conceptos BÍBLICOS.\n"
                "REGLAS CRÍTICAS:\n"
                "1. Traduce conceptos antiguos a ideas comprensibles, siempre basándote en el texto sagrado.\n"
                "2. No introduzcas elementos históricos ajenos al canon.\n" + common_rules
            )
        }

        system_prompt = profiles.get(request.profile, profiles["academic"])

        user_prompt = f"Contexto bíblico:\n{context}\n\nPregunta del usuario: {request.query}"

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
