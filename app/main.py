import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("enterprise-rag")


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

INDEX_PATH = os.getenv(
    "INDEX_PATH",
    "data/faiss_index",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

LLM_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)

FAISS_K = int(
    os.getenv("FAISS_K", "10")
)

FINAL_K = int(
    os.getenv("FINAL_K", "3")
)

RELEVANCE_THRESHOLD = float(
    os.getenv("RELEVANCE_THRESHOLD", "1.0")
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question or query for the SAP SD enterprise assistant.",
    )

    @field_validator("question")
    @classmethod
    def validate_question_not_blank(cls, value: str) -> str:
        trimmed = value.strip()

        if not trimmed:
            raise ValueError(
                "Question cannot be empty or solely whitespace."
            )

        return trimmed


class ChatResponse(BaseModel):
    answer: str
    sources: List[int]
    model: Optional[str] = None


# ============================================================
# RAG SERVICE
# ============================================================

class RAGService:

    def __init__(self):

        logger.info(
            "Initializing RAG Service..."
        )

        # ----------------------------------------------------
        # Groq API Key
        # ----------------------------------------------------

        groq_api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not groq_api_key:
            logger.warning(
                "GROQ_API_KEY environment variable is missing. "
                "LLM calls will fail if not set."
            )

        # ----------------------------------------------------
        # Embedding Model
        # ----------------------------------------------------

        logger.info(
            f"Loading embedding model: {EMBEDDING_MODEL}"
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        # ----------------------------------------------------
        # FAISS Vector Store
        # ----------------------------------------------------

        if not os.path.exists(INDEX_PATH):

            logger.warning(
                f"FAISS index not found at '{INDEX_PATH}'. "
                "Ensure ingestion_service.py is run before "
                "queries are served."
            )

            self.vector_store = None

        else:

            logger.info(
                f"Loading FAISS index from {INDEX_PATH}..."
            )

            self.vector_store = FAISS.load_local(
                INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            logger.info(
                "FAISS index loaded successfully."
            )

        # ----------------------------------------------------
        # Cross-Encoder Reranker
        # ----------------------------------------------------

        logger.info(
            f"Loading reranker model: {RERANKER_MODEL}"
        )

        self.reranker = CrossEncoder(
            RERANKER_MODEL
        )

        # ----------------------------------------------------
        # Groq LLM
        # ----------------------------------------------------

        logger.info(
            f"Initializing ChatGroq with model: {LLM_MODEL}"
        )

        self.llm = ChatGroq(
            model=LLM_MODEL,
            temperature=0,
        )

        logger.info(
            "RAG Service initialization completed successfully."
        )

    # ========================================================
    # ENSURE FAISS INDEX IS LOADED
    # ========================================================

    def ensure_index_loaded(self):

        if self.vector_store is None:

            if os.path.exists(INDEX_PATH):

                self.vector_store = FAISS.load_local(
                    INDEX_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )

                logger.info(
                    "FAISS index loaded successfully."
                )

            else:

                raise FileNotFoundError(
                    f"FAISS index not found at '{INDEX_PATH}'. "
                    "Run ingestion first."
                )

    # ========================================================
    # RETRIEVE + RERANK
    # ========================================================

    def retrieve_and_rerank(
        self,
        query: str,
    ):

        self.ensure_index_loaded()

        # ----------------------------------------------------
        # 1. FAISS Candidate Retrieval
        # ----------------------------------------------------

        results = (
            self.vector_store.similarity_search_with_score(
                query,
                k=FAISS_K,
            )
        )

        if not results:
            return []

        # ----------------------------------------------------
        # 2. Cross-Encoder Reranking
        # ----------------------------------------------------

        pairs = [
            (query, doc.page_content)
            for doc, _ in results
        ]

        reranker_scores = self.reranker.predict(
            pairs
        )

        reranked_results = []

        for (
            (doc, faiss_score),
            reranker_score,
        ) in zip(
            results,
            reranker_scores,
        ):

            reranked_results.append(
                (
                    doc,
                    faiss_score,
                    float(reranker_score),
                )
            )

        # ----------------------------------------------------
        # 3. Sort by Reranker Score
        # ----------------------------------------------------

        reranked_results.sort(
            key=lambda item: item[2],
            reverse=True,
        )

        return reranked_results

    # ========================================================
    # EVIDENCE VALIDATION
    # ========================================================

    def validate_evidence(
        self,
        query: str,
        doc,
    ) -> bool:
        """
        Perform a simple rule-based validation to determine
        whether a retrieved document contains useful evidence
        for the user's question.
        """

        query_words = set(
            query.lower().split()
        )

        document_words = set(
            doc.page_content.lower().split()
        )

        if not query_words:
            return False

        matching_words = (
            query_words.intersection(
                document_words
            )
        )

        match_ratio = (
            len(matching_words)
            / len(query_words)
        )

        page = doc.metadata.get(
            "page"
        )

        logger.info(
            f"Evidence validation | "
            f"Page={page} | "
            f"Match ratio={match_ratio:.2f}"
        )

        return match_ratio >= 0.20

    # ========================================================
    # SELECT RELEVANT RESULTS
    # ========================================================

    def select_relevant_results(
        self,
        query: str,
        results,
    ):
        """
        Select strong, validated and non-duplicate
        evidence from reranked results.
        """

        if not results:
            return []

        # ----------------------------------------------------
        # DEBUG: LOG RERANKED RESULTS
        # ----------------------------------------------------

        logger.info(
            "Reranked retrieval results:"
        )

        for rank, (
            doc,
            faiss_score,
            reranker_score,
        ) in enumerate(
            results,
            start=1,
        ):

            page = doc.metadata.get(
                "page"
            )

            logger.info(
                f"Rank={rank} | "
                f"Page={page} | "
                f"FAISS={faiss_score:.4f} | "
                f"Reranker={reranker_score:.4f}"
            )

        # ----------------------------------------------------
        # 1. Apply Reranker Relevance Threshold
        # ----------------------------------------------------

        filtered_results = [
            result
            for result in results
            if result[2] >= RELEVANCE_THRESHOLD
        ]

        # ----------------------------------------------------
        # 2. Fallback if Nothing Passes Threshold
        # ----------------------------------------------------

        if not filtered_results:

            logger.info(
                "No results passed relevance threshold."
            )

            return []

        # ----------------------------------------------------
        # 3. Evidence Validation
        # ----------------------------------------------------

        validated_results = []

        for result in filtered_results:

            doc, faiss_score, reranker_score = result

            if self.validate_evidence(
                query,
                doc,
            ):

                validated_results.append(
                    result
                )

            else:

                page = doc.metadata.get(
                    "page"
                )

                logger.info(
                    f"Evidence rejected | "
                    f"Page={page} | "
                    f"Reranker={reranker_score:.4f}"
                )

        # ----------------------------------------------------
        # 4. Fallback if Evidence Validation Rejects All
        # ----------------------------------------------------

        if not validated_results:

            logger.info(
                "No results passed evidence validation."
            )

            return []

        # ----------------------------------------------------
        # 5. Remove Duplicate Pages
        # ----------------------------------------------------

        selected_results = []

        seen_pages = set()

        for result in validated_results:

            doc, faiss_score, reranker_score = result

            page = doc.metadata.get(
                "page"
            )

            # Skip duplicate page
            if page in seen_pages:
                continue

            selected_results.append(
                result
            )

            if page is not None:
                seen_pages.add(
                    page
                )

            # ------------------------------------------------
            # Stop after FINAL_K results
            # ------------------------------------------------

            if len(selected_results) >= FINAL_K:
                break

        logger.info(
            f"Selected {len(selected_results)} "
            f"validated results from "
            f"{len(results)} candidates."
        )

        return selected_results

    # ========================================================
    # ANSWER GENERATION
    # ========================================================

    def answer(
        self,
        query: str,
    ) -> dict:

        # ----------------------------------------------------
        # 1. Retrieve + Rerank
        # ----------------------------------------------------

        results = self.retrieve_and_rerank(
            query
        )

        if not results:

            return {
                "answer": (
                    "I could not find relevant information "
                    "in the document."
                ),
                "sources": [],
                "model": LLM_MODEL,
            }

        # ----------------------------------------------------
        # 2. Select Validated Evidence
        # ----------------------------------------------------

        final_results = (
            self.select_relevant_results(
                query,
                results,
            )
        )

        if not final_results:

            return {
                "answer": (
                    "I could not find enough reliable "
                    "evidence in the provided document."
                ),
                "sources": [],
                "model": LLM_MODEL,
            }

        # ----------------------------------------------------
        # 3. Build Context
        # ----------------------------------------------------

        context_parts = []

        sources = []

        for (
            doc,
            _,
            reranker_score,
        ) in final_results:

            page = doc.metadata.get(
                "page"
            )

            context_parts.append(
                f"--- SOURCE: Page {page} ---\n"
                f"{doc.page_content}\n"
            )

            if (
                page is not None
                and page not in sources
            ):

                sources.append(
                    page
                )

        context = "\n".join(
            context_parts
        )

        # ----------------------------------------------------
        # 4. Build LLM Prompt
        # ----------------------------------------------------

        prompt = f"""
You are an SAP SD enterprise document assistant.

Answer the user's question using ONLY the provided document context.

Rules:

1. Do not use outside knowledge.

2. Do not invent information.

3. If the answer cannot be found in the provided context, say:
"I could not find the answer in the provided document."

4. Give a concise and clear answer.

5. Do not create or invent source pages.

6. Use only information supported by the provided sources.

Context:

{context}

Question:

{query}

Answer:
"""

        # ----------------------------------------------------
        # 5. Generate Answer
        # ----------------------------------------------------

        response = self.llm.invoke(
            prompt
        )

        # ----------------------------------------------------
        # 6. Return API Response
        # ----------------------------------------------------

        return {
            "answer": response.content,
            "sources": sources,
            "model": LLM_MODEL,
        }


# ============================================================
# APPLICATION LIFESPAN & INSTANCE
# ============================================================

rag_service: Optional[RAGService] = None


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    global rag_service

    logger.info(
        "Application startup: Initializing services..."
    )

    rag_service = RAGService()

    yield

    logger.info(
        "Application shutdown: Cleaning up resources..."
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Enterprise RAG API",
    description=(
        "Production-ready SAP SD RAG API with "
        "FAISS Vector Search, Cross-Encoder Reranker "
        "& Groq LLM"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "ALLOWED_ORIGINS",
        "*",
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get(
    "/health",
    tags=["Health"],
)
def health_check():

    index_ready = (
        rag_service.vector_store is not None
        if rag_service
        else os.path.exists(INDEX_PATH)
    )

    return {
        "status": "healthy",
        "model": LLM_MODEL,
        "index_ready": index_ready,
    }


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
)
def chat(
    request: ChatRequest,
):

    global rag_service

    if rag_service is None:
        rag_service = RAGService()

    try:

        result = rag_service.answer(
            request.question
        )

        return result

    except FileNotFoundError as e:

        logger.error(
            f"Index error: {e}"
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Vector index is not initialized. "
                "Please ensure ingestion has run."
            ),
        )

    except Exception as e:

        logger.error(
            f"Error processing chat request: {e}",
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Failed to generate response: {str(e)}"
            ),
        )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )