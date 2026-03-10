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
            "Eres un filtro BINARIO. Tu única misión es detectar si la pregunta trata sobre el canon bíblico.\n"
            "Responde 'RECHAZAR' si la pregunta menciona:\n"
            "- Nombres modernos o de geografía actual (Madrid, España, Trump, Rubio).\n"
            "- Historia post-bíblica o personajes externos (Lutero, Arrio, Teodosio).\n"
            "- Sistemas teológicos o doctrinas con nombres humanos.\n"
            "Responde 'CONTINUAR' solo si la pregunta es sobre personajes o texto de los 66 libros de la Biblia.\n"
            "Respuesta (RECHAZAR o CONTINUAR):"
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

        # Reglas de amnesia total e identidad bíblica (Sola Palabra)
        common_rules = (
            "\nNORMAS DE RESPUESTA (VITALES):\n"
            "1. NO SABES NADA de historia universal ni personas nacidas después del año 100 d.C.\n"
            "2. PROHIBIDO: No digas 'Teología', 'Reformada', 'Lutero', 'Arrio', 'Siglo IV', 'Soli Deo Gloria'.\n"
            "3. Habla de forma DIRECTA, RESPETUOSA y NATURAL. Sin jerga cristiana.\n"
            "4. CITA los versículos.\n"
            "5. Si el tema no es bíblico, responde: 'Basado exclusivamente en el contexto de las Sagradas Escrituras, no existe información sobre esta persona o concepto.'\n"
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
            "references": references
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in RAG chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "index": INDEX_NAME, "version": "1.2-nuclear"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
