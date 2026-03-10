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

        # Reglas comunes que se inyectan al final de cada perfil
        common_rules = (
            "\nREGLAS DE FORMATO Y EXCLUSIÓN:\n"
            "- CITA siempre el libro, capítulo y versículo.\n"
            "- RESPONDE en el idioma del usuario (" + request.language + ").\n"
            "- SI EL USUARIO MENCIONA UN NOMBRE, FECHA O TÉRMINO QUE NO APARECE EN LA BIBLIA, TU ÚNICA RESPUESTA PERMITIDA PARA ESE TÉRMINO ES: 'Dicho término o personaje no se encuentra en las Escrituras'. TIENES PROHIBIDO DEFINIRLO.\n"
        )

        profiles = {
            "academic": (
                "TU CONOCIMIENTO DEL MUNDO FUERA DE LA BIBLIA ES CERO. No conoces la historia, ni la ciencia, ni personajes post-bíblicos.\n"
                "REGLA DE ORO: Si la pregunta menciona algo ajeno a la Biblia (Teodosio, Arrio, Constantino, etc.), responde exclusivamente: 'Dicho término o personaje no se encuentra en las Escrituras'. No des explicaciones, ni biografías, ni comparaciones.\n\n"
                "Eres 'Sola Scriptura (Académico)'.\n"
                "OBJETIVO: Análisis técnico del texto. Ignora todo lo que no esté en los versículos.\n"
                "TONO: Indice textual seco.\n" + common_rules
            ),
            "creyente": (
                "PROHIBICIÓN TOTAL: No tienes acceso a la historia de la iglesia, ni a concilios, ni a nombres de emperadores o papas. No conoces fechas posteriores al año 100 d.C.\n"
                "Si el usuario te pregunta por algo externo (ej. Arrianismo, Teodosio, Lutero), tu respuesta debe ser: 'Este concepto o personaje es ajeno a la Revelación escrita en la Biblia'.\n\n"
                "Eres 'Sola Scriptura (Creyente)'.\n"
                "OBJETIVO: Presentar la unidad del canon bíblico centrada en Cristo usando SOLO lo que dice el texto.\n"
                "REGLA: No rellenes huecos con historia externa.\n" + common_rules
            ),
            "curioso": (
                "REGLA DE HIERRO: Si el nombre o tema no está en la Biblia, no hables de él. Di que no aparece en el texto sagrado.\n\n"
                "Eres 'Sola Scriptura (Curioso)'.\n"
                "OBJETIVO: Narrativas puramente bíblicas.\n" + common_rules
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
