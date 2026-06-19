"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.

    Parameters
    ----------
    settings : Settings, optional
        Application settings.

    Example
    -------
    >>> chunker = DocumentChunker()
    >>> chunks = chunker.chunk_file(
    ...     Path("data/corpus/lstm.md"),
    ...     metadata_overrides={"topic": "LSTM", "difficulty": "intermediate"}
    ... )
    >>> print(f"Produced {len(chunks)} chunks")
    """

    # Default chunking parameters — justify these in your architecture diagram.
    # chunk_size: 512 tokens balances context richness with retrieval precision.
    # chunk_overlap: 50 tokens prevents concepts that span chunk boundaries
    # from being lost entirely. A common interview question.
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """
        Load a file and split it into DocumentChunks.

        Automatically detects file type and routes to the appropriate
        loader. Applies metadata_overrides on top of auto-detected
        metadata where provided.

        Parameters
        ----------
        file_path : Path
            Absolute or relative path to the source file.
        metadata_overrides : dict, optional
            Metadata fields to set or override. Keys must match
            ChunkMetadata field names. Commonly used to set topic
            and difficulty when the file does not encode these.
        chunk_size : int
            Maximum characters per chunk.
        chunk_overlap : int
            Characters of overlap between adjacent chunks.

        Returns
        -------
        list[DocumentChunk]
            Fully prepared chunks with deterministic IDs and metadata.

        Raises
        ------
        ValueError
            If the file type is not supported.
        FileNotFoundError
            If the file does not exist at the given path.
        """
        # TODO: implement
        # 1. Validate file exists
        # 2. Route to _chunk_pdf or _chunk_markdown based on suffix
        # 3. Apply metadata_overrides
        # 4. Generate chunk_ids using VectorStoreManager.generate_chunk_id
        # 5. Return list[DocumentChunk]
        # 1. Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        logger.info(f"Chunking file: {file_path.name}")
        
        # 2. Route to specific loader
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            raw_chunks = self._chunk_pdf(file_path, chunk_size, chunk_overlap)
        elif suffix in [".md", ".markdown"]:
            raw_chunks = self._chunk_markdown(file_path, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Only .pdf and .md supported.")

        # 3. Infer base metadata from filename and apply overrides
        base_metadata = self._infer_metadata(file_path, metadata_overrides)

        # 4 & 5. Convert raw dicts to DocumentChunk objects and generate IDs
        document_chunks = []
        for raw in raw_chunks:
            text = raw.get("text", "").strip()
            if not text:
                continue
                
            chunk_meta_dict = base_metadata.to_dict()
            
            # Append markdown header context to the topic if it exists
            if "header" in raw and raw["header"]:
                chunk_meta_dict["topic"] = f"{chunk_meta_dict['topic']} - {raw['header']}"
            
            final_metadata = ChunkMetadata(**chunk_meta_dict)
            
            chunk_id = VectorStoreManager.generate_chunk_id(file_path.name, text)
            
            document_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=final_metadata
                )
            )

        logger.info(f"Produced {len(document_chunks)} chunks from {file_path.name}")
        return document_chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk multiple files in a single call.

        Used by the UI multi-file upload handler to process all
        uploaded files before passing to VectorStoreManager.ingest().

        Parameters
        ----------
        file_paths : list[Path]
            List of file paths to process.
        metadata_overrides : dict, optional
            Applied to all files. Per-file metadata should be handled
            by calling chunk_file() individually.

        Returns
        -------
        list[DocumentChunk]
            Combined chunks from all files, preserving source attribution
            in each chunk's metadata.
        """
        # TODO: implement — iterate and collect, handle per-file errors
        all_chunks = []
        for path in file_paths:
            try:
                file_chunks = self.chunk_file(path, metadata_overrides)
                all_chunks.extend(file_chunks)
            except Exception as e:
                logger.error(f"Failed to chunk {path.name}: {e}")
                
        return all_chunks

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a PDF file.

        Uses PyPDFLoader for text extraction followed by
        RecursiveCharacterTextSplitter for chunking.

        Interview talking point: PDFs from academic papers often contain
        noisy content (headers, footers, reference lists, equations as
        text). Post-processing to remove this noise improves retrieval
        quality significantly.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'page' keys before conversion
            to DocumentChunk objects.
        """
        # TODO: implement using langchain_community.document_loaders.PyPDFLoader
        # and langchain.text_splitter.RecursiveCharacterTextSplitter
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(docs)
        
        return [
            {
                "text": doc.page_content, 
                "page": doc.metadata.get("page", 0)
            } 
            for doc in split_docs
        ]

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a Markdown file.

        Uses MarkdownHeaderTextSplitter first to respect document
        structure (headers create natural chunk boundaries), then
        RecursiveCharacterTextSplitter for oversized sections.

        Interview talking point: header-aware splitting preserves
        semantic coherence better than naive character splitting —
        a concept within one section stays within one chunk.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'header' keys.
        """
        # TODO: implement using langchain.text_splitter.MarkdownHeaderTextSplitter
        content = file_path.read_text(encoding="utf-8")
        
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(content)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        final_splits = text_splitter.split_documents(md_header_splits)
        
        raw_chunks = []
        for doc in final_splits:
            # Grab the deepest available header for context
            header_context = doc.metadata.get("Header 3") or doc.metadata.get("Header 2") or doc.metadata.get("Header 1") or ""
            raw_chunks.append({
                "text": doc.page_content,
                "header": header_context
            })
            
        return raw_chunks

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """
        Infer chunk metadata from filename conventions and apply overrides.

        Filename convention (recommended to Corpus Architects):
          <topic>_<difficulty>.md or <topic>_<difficulty>.pdf
          e.g. lstm_intermediate.md, alexnet_advanced.pdf

        If the filename does not follow this convention, defaults are
        applied and the Corpus Architect must provide overrides manually.

        Parameters
        ----------
        file_path : Path
            Source file path used to infer topic and difficulty.
        overrides : dict, optional
            Explicit metadata values that take precedence over inference.

        Returns
        -------
        ChunkMetadata
            Populated metadata object.
        """
        # TODO: implement filename parsing + override merging
        # Bonus topics: SOM, BoltzmannMachine, GAN → set is_bonus=True
        filename = file_path.stem.lower()
        
        topic = "General ML"
        difficulty = "intermediate"
        is_bonus = False
        
        parts = filename.split("_")
        if len(parts) >= 2:
            topic = parts[0].upper()
            difficulty = parts[1].lower()
        else:
            topic = parts[0].upper()
            
        bonus_topics = ["SOM", "BOLTZMANN", "GAN", "AUTOENCODER"]
        if any(b in topic for b in bonus_topics):
            is_bonus = True

        metadata_dict = {
            "topic": topic,
            "difficulty": difficulty,
            "type": "concept_explanation",
            "source": file_path.name,
            "related_topics": [],
            "is_bonus": is_bonus
        }
        
        if overrides:
            metadata_dict.update(overrides)
            
        return ChunkMetadata(**metadata_dict)