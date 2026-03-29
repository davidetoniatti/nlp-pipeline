import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from nlp_inference import NLPPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "120"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the NLP pipeline once at startup."""
    logger.info("Initializing NLP pipeline...")
    pipeline = NLPPipeline()
    app.state.pipeline = pipeline
    logger.info("NLP pipeline ready.")
    try:
        yield
    finally:
        app.state.pipeline = None
        logger.info("NLP pipeline shutdown completed.")


app = FastAPI(title="NLP Inference API", lifespan=lifespan)


class InferenceRequest(BaseModel):
    """Request payload for batch inference."""

    model_config = ConfigDict(extra="forbid")

    doc_ids: list[str] = Field(min_length=1)
    texts: list[str] = Field(min_length=1)
    batch_size: int = Field(default=32, ge=1, le=256)


class EntityResponse(BaseModel):
    """Public API shape for one entity."""

    text: str
    label: str
    start: int
    end: int
    score: float


class DocResultResponse(BaseModel):
    """Public API shape for one analyzed document."""

    doc_id: str
    sentiment_score: float
    sentiment_label: str
    entities: list[EntityResponse]
    summary: Optional[str] = None
    error: Optional[str] = None


class ModelMetadataResponse(BaseModel):
    """Specific metadata for a single model."""

    name: str
    version: str
    revision: Optional[str] = None
    tokenizer: Optional[str] = None
    provider: str


class ModelVersionsResponse(BaseModel):
    """Model metadata returned with the batch."""

    sentiment: ModelMetadataResponse
    ner: ModelMetadataResponse
    summary: ModelMetadataResponse
    prompt_version: str
    prompt_hash: Optional[str] = None
    service_git_sha: str


class InferenceResponse(BaseModel):
    """Full API response for one batch."""

    results: list[DocResultResponse]
    model_versions: ModelVersionsResponse


@app.post("/analyze", response_model=InferenceResponse)
async def analyze(req: InferenceRequest, request: Request):
    """Validate input, run the pipeline, and shape the API response."""
    if len(req.doc_ids) != len(req.texts):
        raise HTTPException(status_code=400, detail="Mismatched doc_ids and texts")

    if any(not doc_id.strip() for doc_id in req.doc_ids):
        raise HTTPException(
            status_code=400, detail="doc_ids cannot contain empty values"
        )

    if any(not text.strip() for text in req.texts):
        raise HTTPException(status_code=400, detail="texts cannot contain empty values")

    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    try:
        # Bound the whole request so one slow batch does not hang forever.
        results, versions = await asyncio.wait_for(
            pipeline.process_batch(
                req.doc_ids,
                req.texts,
                batch_size=req.batch_size,
            ),
            timeout=REQUEST_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Inference timed out") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    return InferenceResponse(
        results=[
            DocResultResponse(
                doc_id=r.doc_id,
                sentiment_score=r.sentiment_score,
                sentiment_label=r.sentiment_label,
                entities=[
                    EntityResponse(
                        text=e.text,
                        label=e.label,
                        start=e.start,
                        end=e.end,
                        score=float(e.score),
                    )
                    for e in r.entities
                ],
                summary=r.summary,
                error=r.error,
            )
            for r in results
        ],
        model_versions=ModelVersionsResponse(
            sentiment=ModelMetadataResponse(
                name=versions.sentiment.name,
                version=versions.sentiment.version,
                revision=versions.sentiment.revision,
                tokenizer=versions.sentiment.tokenizer,
                provider=versions.sentiment.provider,
            ),
            ner=ModelMetadataResponse(
                name=versions.ner.name,
                version=versions.ner.version,
                revision=versions.ner.revision,
                tokenizer=versions.ner.tokenizer,
                provider=versions.ner.provider,
            ),
            summary=ModelMetadataResponse(
                name=versions.summary.name,
                version=versions.summary.version,
                revision=versions.summary.revision,
                tokenizer=versions.summary.tokenizer,
                provider=versions.summary.provider,
            ),
            prompt_version=versions.prompt_version,
            prompt_hash=versions.prompt_hash,
            service_git_sha=versions.service_git_sha,
        ),
    )


@app.get("/health")
async def health(request: Request):
    """Report whether the pipeline is initialized and ready."""
    pipeline_ready = (
        hasattr(request.app.state, "pipeline")
        and request.app.state.pipeline is not None
    )
    if not pipeline_ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return {"status": "ok"}
