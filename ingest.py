import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    char_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentIngestor:
    """Document Ingestion and Semantic Sliding-Window Chunking Module."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text: str) -> str:
        """Normalize whitespace and remove non-printable characters."""
        text = re.sub(r'\r\n|\r', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def chunk_document(self, doc_id: str, text: str) -> List[DocumentChunk]:
        """Split text into semantic chunks using sliding windows with sentence boundary preservation."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        # Split into approximate sentences
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        chunks: List[DocumentChunk] = []
        current_sentences: List[str] = []
        current_len = 0
        chunk_idx = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                        doc_id=doc_id,
                        content=chunk_text,
                        chunk_index=chunk_idx,
                        char_count=len(chunk_text),
                        metadata={"source": doc_id}
                    )
                )
                chunk_idx += 1
                
                # Maintain overlap by retaining trailing sentences
                overlap_len = 0
                overlap_sentences: List[str] = []
                for s in reversed(current_sentences):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_sentences = overlap_sentences
                current_len = sum(len(s) for s in current_sentences)

            current_sentences.append(sent)
            current_len += sent_len

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                    doc_id=doc_id,
                    content=chunk_text,
                    chunk_index=chunk_idx,
                    char_count=len(chunk_text),
                    metadata={"source": doc_id}
                )
            )

        return chunks
