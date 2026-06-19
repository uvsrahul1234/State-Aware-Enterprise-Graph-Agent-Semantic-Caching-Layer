"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        # TODO: implement
        # 1. Ensure Path(self._settings.chroma_db_path).mkdir(parents=True, exist_ok=True)
        # 2. chromadb.PersistentClient(path=self._settings.chroma_db_path)
        # 3. client.get_or_create_collection(
        #        name=self._settings.chroma_collection_name,
        #        metadata={"hnsw:space": "cosine"}   # cosine similarity
        #    )
        # 4. Log successful initialisation with collection name and item count
        try:
            # 1. Ensure directory exists
            db_path = Path(self._settings.chroma_db_path)
            db_path.mkdir(parents=True, exist_ok=True)

            # 2. Initialize persistent client
            self._client = chromadb.PersistentClient(
                path=str(db_path),
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            # 3. Get or create collection with explicitly set cosine similarity
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # 4. Log successful initialization
            count = self._collection.count()
            logger.info(f"Vector Store initialized at {db_path} | Collection: {self._settings.chroma_collection_name} | Items: {count}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}")

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        # TODO: implement
        # self._collection.get(ids=[chunk_id])
        # Return True if the result contains the ID, False otherwise
        try:
            result = self._collection.get(ids=[chunk_id])
            # Check if the 'ids' list inside the result dictionary actually has items
            return len(result.get("ids", [])) > 0
        except Exception as e:
            logger.error(f"Error checking for duplicate chunk {chunk_id}: {e}")
            return False

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so re-ingestion of modified content updates
        existing chunks rather than raising an error.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        # TODO: implement
        # result = IngestionResult()
        # For each chunk:
        #   - check_duplicate(chunk.chunk_id) → if True, result.skipped += 1, continue
        #   - embed chunk.chunk_text using self._embeddings.embed_documents([chunk.chunk_text])
        #   - self._collection.upsert(
        #         ids=[chunk.chunk_id],
        #         embeddings=[embedding],
        #         documents=[chunk.chunk_text],
        #         metadatas=[chunk.metadata.to_dict()]
        #     )
        #   - result.ingested += 1
        # Log summary and return result
        result = IngestionResult(ingested=0, skipped=0)
        error_count = 0  # We will track errors locally instead
        
        if not chunks:
            return result

        # Batch processing to prevent memory overflow on large documents
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            ids_to_insert = []
            texts_to_insert = []
            metadatas_to_insert = []
            
            for chunk in batch:
                try:
                    # Idempotency check: Skip if already ingested
                    if self.check_duplicate(chunk.chunk_id):
                        result.skipped += 1
                        continue
                        
                    ids_to_insert.append(chunk.chunk_id)
                    texts_to_insert.append(chunk.chunk_text)
                    metadatas_to_insert.append(chunk.metadata.to_dict())
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")
                    error_count += 1
            
            if texts_to_insert:
                try:
                    embeddings = self._embeddings.embed_documents(texts_to_insert)
                    
                    self._collection.upsert(
                        ids=ids_to_insert,
                        embeddings=embeddings,
                        documents=texts_to_insert,
                        metadatas=metadatas_to_insert
                    )
                    result.ingested += len(ids_to_insert)
                except Exception as e:
                    logger.error(f"Failed to upsert batch to ChromaDB: {e}")
                    error_count += len(ids_to_insert)
                    
        logger.info(f"Ingestion complete. Ingested: {result.ingested}, Skipped: {result.skipped}, Errors: {error_count}")
        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        # TODO: implement
        # k = k or self._settings.retrieval_k
        # Build where_filter dict from topic_filter and difficulty_filter if provided
        # Embed query_text using self._embeddings.embed_query(query_text)
        # self._collection.query(
        #     query_embeddings=[query_embedding],
        #     n_results=k,
        #     where=where_filter,      # None if no filters
        #     include=["documents", "metadatas", "distances"]
        # )
        # Convert distances to similarity scores: score = 1 - distance (for cosine)
        # Filter out chunks below self._settings.similarity_threshold
        # Return list of RetrievedChunk objects sorted by score descending
        k = k or self._settings.retrieval_k
        
        # Build ChromaDB where-filter
        conditions = []
        if topic_filter:
            conditions.append({"topic": topic_filter})
        if difficulty_filter:
            conditions.append({"difficulty": difficulty_filter})
            
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}
        else:
            where_filter = None

        try:
            # Generate embedding for the query
            query_embedding = self._embeddings.embed_query(query_text)
            
            # Query the database
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieved_chunks = []
            
            if not results["ids"] or not results["ids"][0]:
                return retrieved_chunks
                
            for i in range(len(results["ids"][0])):
                # 
                # ChromaDB returns Euclidean/Cosine distance.
                # For cosine space, similarity score = 1 - distance.
                distance = results["distances"][0][i]
                similarity_score = 1.0 - distance
                
                # Apply the hallucination guard rail
                if similarity_score >= self._settings.similarity_threshold:
                    chunk = RetrievedChunk(
                        chunk_id=results["ids"][0][i],
                        chunk_text=results["documents"][0][i],
                        metadata=ChunkMetadata(**results["metadatas"][0][i]),
                        score=similarity_score
                    )
                    retrieved_chunks.append(chunk)
                    
            # Sort highest score first
            retrieved_chunks.sort(key=lambda x: x.score, reverse=True)
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.

        Used by the UI to populate the document viewer panel.

        Returns
        -------
        list[dict]
            Each item contains: source (str), topic (str), chunk_count (int).
        """
        # TODO: implement
        # Query all metadata from the collection
        # Group by metadata["source"] and count chunks per source
        # Return sorted list of dicts
        try:
            results = self._collection.get(include=["metadatas"])
            if not results["metadatas"]:
                return []
                
            doc_counts = {}
            doc_topics = {}
            
            for meta in results["metadatas"]:
                source = meta.get("source", "Unknown")
                topic = meta.get("topic", "Unknown")
                
                doc_counts[source] = doc_counts.get(source, 0) + 1
                doc_topics[source] = topic
                
            output = [
                {"source": src, "topic": doc_topics[src], "chunk_count": count}
                for src, count in doc_counts.items()
            ]
            
            return sorted(output, key=lambda x: x["source"])
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        # TODO: implement
        # self._collection.get(where={"source": source}, include=["documents", "metadatas"])
        # Reconstruct DocumentChunk objects from results
        try:
            results = self._collection.get(
                where={"source": source}, 
                include=["documents", "metadatas"]
            )
            
            chunks = []
            for i in range(len(results["ids"])):
                chunks.append(DocumentChunk(
                    chunk_id=results["ids"][i],
                    chunk_text=results["documents"][i],
                    metadata=ChunkMetadata(**results["metadatas"][i])
                ))
            return chunks
        except Exception as e:
            logger.error(f"Failed to get chunks for document {source}: {e}")
            return []

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        # TODO: implement
        try:
            results = self._collection.get(include=["metadatas"])
            total_chunks = len(results["ids"])
            
            if total_chunks == 0:
                return {
                    "total_chunks": 0,
                    "topics": [],
                    "sources": [],
                    "bonus_topics_present": False
                }
                
            sources = set()
            topics = set()
            bonus = False
            
            for meta in results["metadatas"]:
                sources.add(meta.get("source"))
                topics.add(meta.get("topic"))
                if meta.get("is_bonus", False):
                    bonus = True
                    
            return {
                "total_chunks": total_chunks,
                "topics": sorted(list(topics)),
                "sources": sorted(list(sources)),
                "bonus_topics_present": bonus
            }
        except Exception as e:
            logger.error(f"Failed to fetch collection stats: {e}")
            return {}

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        # TODO: implement
        # self._collection.delete(where={"source": source})
        try:
            items_to_delete = self._collection.get(where={"source": source})
            count = len(items_to_delete["ids"])
            
            if count > 0:
                self._collection.delete(where={"source": source})
                logger.info(f"Deleted {count} chunks for document: {source}")
                
            return count
        except Exception as e:
            logger.error(f"Failed to delete document {source}: {e}")
            return 0
