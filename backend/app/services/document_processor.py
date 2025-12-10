"""Document processing service for PDFs and text files"""

from typing import List, Tuple
import hashlib
from pathlib import Path
import re

from PyPDF2 import PdfReader

from app.models.document import Document, DocumentChunk
from app.utils.logger import get_logger
from app.utils.config_loader import get_config

logger = get_logger()


class DocumentProcessor:
    """Process documents into chunks for embedding"""

    def __init__(self):
        config = get_config()
        self.chunk_size = config.get('vectorstore.chunk_size', 1000)
        self.chunk_overlap = config.get('vectorstore.chunk_overlap', 0.2)

    def process_pdf(self, file_path: str, document_id: str) -> Tuple[Document, List[DocumentChunk]]:
        """Process a PDF file"""
        logger.info(f"Processing PDF: {file_path}")

        # Read PDF
        reader = PdfReader(file_path)
        num_pages = len(reader.pages)

        # Extract text from each page
        page_texts = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            page_texts.append((i + 1, text))

        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)

        # Create document metadata
        document = Document(
            id=document_id,
            filename=Path(file_path).name,
            type="pdf",
            path=file_path,
            pages=num_pages,
            file_hash=file_hash
        )

        # Create chunks
        chunks = self._create_chunks_from_pages(document_id, page_texts)
        document.chunks = len(chunks)

        logger.info(f"Processed PDF: {num_pages} pages, {len(chunks)} chunks")
        return document, chunks

    def process_txt(self, file_path: str, document_id: str) -> Tuple[Document, List[DocumentChunk]]:
        """Process a text file"""
        logger.info(f"Processing TXT: {file_path}")

        # Read text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)

        # Create document metadata
        document = Document(
            id=document_id,
            filename=Path(file_path).name,
            type="txt",
            path=file_path,
            file_hash=file_hash
        )

        # Create chunks
        chunks = self._create_chunks_from_text(document_id, text)
        document.chunks = len(chunks)

        logger.info(f"Processed TXT: {len(chunks)} chunks")
        return document, chunks

    def _create_chunks_from_pages(
        self,
        document_id: str,
        page_texts: List[Tuple[int, str]]
    ) -> List[DocumentChunk]:
        """Create chunks from page texts"""
        chunks = []
        chunk_index = 0

        for page_num, text in page_texts:
            # Split page by paragraphs
            paragraphs = self._split_into_paragraphs(text)

            current_chunk = ""
            for para in paragraphs:
                # If adding this paragraph would exceed chunk size
                if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk = DocumentChunk(
                        document_id=document_id,
                        content=current_chunk.strip(),
                        chunk_index=chunk_index,
                        page=page_num,
                        metadata={"page": page_num}
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                    # Start new chunk with overlap
                    overlap_size = int(len(current_chunk) * self.chunk_overlap)
                    if overlap_size > 0:
                        current_chunk = current_chunk[-overlap_size:] + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para

            # Save remaining chunk for this page
            if current_chunk.strip():
                chunk = DocumentChunk(
                    document_id=document_id,
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    page=page_num,
                    metadata={"page": page_num}
                )
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _create_chunks_from_text(self, document_id: str, text: str) -> List[DocumentChunk]:
        """Create chunks from plain text"""
        chunks = []
        paragraphs = self._split_into_paragraphs(text)

        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk = DocumentChunk(
                    document_id=document_id,
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    metadata={}
                )
                chunks.append(chunk)
                chunk_index += 1

                # Start new chunk with overlap
                overlap_size = int(len(current_chunk) * self.chunk_overlap)
                if overlap_size > 0:
                    current_chunk = current_chunk[-overlap_size:] + "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Save remaining chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                document_id=document_id,
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                metadata={}
            )
            chunks.append(chunk)

        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)

        # Filter out empty paragraphs and clean whitespace
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # If any paragraph is too long, split by sentences
        result = []
        for para in paragraphs:
            if len(para) > self.chunk_size:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) > self.chunk_size and current:
                        result.append(current.strip())
                        current = sent
                    else:
                        current += " " + sent if current else sent
                if current.strip():
                    result.append(current.strip())
            else:
                result.append(para)

        return result

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


# Global instance
_document_processor = None


def get_document_processor() -> DocumentProcessor:
    """Get the global document processor instance"""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor
