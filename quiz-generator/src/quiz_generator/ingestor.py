from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from markitdown import MarkItDown
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class Paper(BaseModel):
    """A paper artifact produced by ingestion.

    V0: this is just text + basic provenance.
    """

    model_config = ConfigDict(extra="forbid")

    paper_id: str = Field(default_factory=lambda: f"paper_{uuid.uuid4().hex}")
    text: str = Field(min_length=1, description="Extracted paper text.")

    source_url: str | None = Field(
        default=None, description="Original URL if available."
    )
    title: str | None = Field(default=None, description="Optional title.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata."
    )


def ingest_text(
    *, text: str, source_url: str | None = None, title: str | None = None
) -> Paper:
    text = (text or "").strip()
    if not text:
        raise ValueError("text must be a non-empty string")
    return Paper(text=text, source_url=source_url, title=title)


def _normalize_arxiv_pdf_url(arxiv_url: str) -> str:
    """Convert common arXiv URL forms to a direct PDF URL."""

    url = arxiv_url.strip()
    if not url:
        raise ValueError("arxiv_url must be a non-empty string")

    logger.debug("Normalizing arXiv URL: %s", url)

    # Already a PDF.
    if url.endswith(".pdf"):
        return url

    # Typical forms:
    # - https://arxiv.org/abs/1706.03762
    # - https://arxiv.org/pdf/1706.03762
    # - https://arxiv.org/abs/1706.03762v5
    if "/abs/" in url:
        url = url.replace("/abs/", "/pdf/")
    if "/pdf/" in url and not url.endswith(".pdf"):
        url = url + ".pdf"

    return url


def _download_url_to_tempfile(url: str, *, suffix: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("url must be http(s)")

    logger.debug("Downloading URL to tempfile: %s", url)

    req = Request(url, headers={"User-Agent": "quiz-generator/0.1"})
    with urlopen(req) as resp:
        data = resp.read()

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return Path(tmp.name)


def ingest_arxiv_url(
    *, arxiv_url: str, title: str | None = None, keep_pdf: bool = False
) -> Paper:
    """Download an arXiv PDF and convert it to text via markitdown.

    V0 is intentionally minimal: it returns only extracted text and basic URL provenance.

    By default, the downloaded PDF is treated as a temporary artifact and is deleted after
    text extraction. Set keep_pdf=True to retain it.
    """

    pdf_url = _normalize_arxiv_pdf_url(arxiv_url)

    # Best-effort title inference from URL path.
    inferred_title: str | None = None
    if title is None:
        m = re.search(r"/pdf/([^/]+)\.pdf$", pdf_url)
        if m:
            inferred_title = m.group(1)

    pdf_path = _download_url_to_tempfile(pdf_url, suffix=".pdf")

    metadata: dict[str, Any] = {"pdf_url": pdf_url}
    if keep_pdf:
        metadata["pdf_path"] = str(pdf_path)

    try:
        logger.debug("Extracting text from PDF via markitdown: %s", pdf_path)
        md = MarkItDown()
        result = md.convert(str(pdf_path))
        text = (result.text_content or "").strip()
    finally:
        if not keep_pdf:
            try:
                pdf_path.unlink(missing_ok=True)
            except OSError:
                pass

    if not text:
        raise ValueError("markitdown produced empty text")

    return Paper(
        text=text,
        source_url=arxiv_url,
        title=title or inferred_title,
        metadata=metadata,
    )
