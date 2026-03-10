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
        # 1. GUARDRAIL (PRE-FILTRO BINARIO)
        # Este paso detiene la consulta si detecta nombres o temas ajenos a la Biblia
        # antes incluso de buscar en los versículos.
        guardrail_prompt = (
            "Eres un filtro de seguridad estricto para una aplicación de Sola Scriptura.\n"
            "Tu única misión es decidir si la pregunta del usuario es estrictamente bíblica.\n\n"
            "Responde 'RECHAZAR' si la pregunta menciona:\n"
            "- Nombres de personas que NO están en la Biblia (ej: Trump, Marco Rubio, Biden, Teodosio, Constantino, Lutero, etc.)\n"
            "- Conceptos históricos post-bíblicos (ej: Reforma, Edicto de Tesalónica, Papado, Siglo IV).\n"
            "- Cualquier tema que no sea una doctrina bíblica o un personaje de los 66 libros del canon.\n\n"
            "Responde 'CONTINUAR' si la pregunta es sobre:\n"
            "- Personajes bíblicos (Moisés, Abraham, Jesús, Job, Pablo, etc.)\n"
            "- Doctrina bíblica pura (Justificación, Gracia, Redención, etc.)\n\n"
            "SÉ EXTREMADAMENTE ESTRICTO. Ante la duda, responde RECHAZAR.\n"
            "Respuesta (SOLO la palabra RECHAZAR o CONTINUAR):"
        )
        
        classification = await generate_response_with_retry(guardrail_prompt, request.query)
        print(f"DEBUG GUARDRAIL: {classification}")
        
        if "RECHAZAR" in classification.upper():
            return {
                "answer": f"Basado exclusivamente en el contexto de las Sagradas Escrituras, no existe información sobre un personaje o concepto llamado '{request.query}'. Los textos sagrados no contienen mención de figuras o eventos ajenos al canon bíblico.",
                "references": []
            }

        # 2. PROCESO RAG (Solo si el guardarraíl lo permite)
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

        # Bloque de restricciones extremas (Nuclear Lockdown)
        common_rules = (
            "\nREGLA SUPREMA DE EXCLUSIÓN:\n"
            "1. NO SABES quién es Donald Trump, ni Teodosio, ni Arrio, ni los papas, ni los reformadores. Tu memoria del mundo moderno o historia universal ha sido BORRADA.\n"
            "2. Si el usuario pregunta por algo que NO está en la Biblia, responde exclusivamente: 'Dicho término o personaje no se encuentra en las Escrituras' y DETIENE la respuesta de inmediato.\n"
            "3. TIENES PROHIBIDO espiritualizar o comparar figuras externas con la Biblia. Si no es bíblico, para ti NO EXISTE.\n"
            "4. CITA el libro, capítulo y versículo de los textos adjuntos.\n"
        )

        profiles = {
            "academic": (
                "IDENTIDAD: Indexador de Textos Antiguos (Modo Textual).\n"
                "Eres un software que solo procesa los versículos de abajo. No conoces la historia universal ni personajes externos.\n"
                "OBJETIVO: Análisis neutro de palabras encontradas en los versículos. Si el nombre no está, di que no existe.\n" + common_rules
            ),
            "creyente": (
                "ACTÚA COMO UN ESCRIBA BÍBLICO ESTRICTO. No conoces la historia del mundo fuera de la Biblia.\n"
                "REGLA DE RECHAZO: Si te preguntan por alguien ajeno al canon (ej: Marco Rubio, Trump, Teodosio), tu única respuesta debe ser: 'Basado exclusivamente en el contexto de las Sagradas Escrituras, no existe información sobre esta persona. Los textos sagrados no mencionan figuras ajenas al canon'.\n"
                "Sola Scriptura. Solus Christus. Solo responde sobre la Biblia.\n" + common_rules
            ),
            "curioso": (
                "IDENTIDAD: Narrador Bíblico Estricto.\n"
                "Cero conocimiento de historia universal. Solo historias bíblicas.\n" + common_rules
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
