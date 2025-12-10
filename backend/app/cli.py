"""CLI commands for RPG Assistant"""

import sys
import argparse
import asyncio
from pathlib import Path

from app.utils.config_loader import get_config
from app.utils.logger import setup_logger, get_logger
from app.services.vector_store import create_vector_store
from app.services.document_processor import get_document_processor
from app.services.embedding_service import get_embedding_service
from app.services.llm_service import create_llm_provider


def validate_config():
    """Validate configuration files"""
    print("Validating configuration...")

    try:
        config = get_config()
        print("✓ Configuration file loaded successfully")

        # Check required sections
        required_sections = ['app', 'llm', 'vectorstore', 'embedding']
        for section in required_sections:
            if config.get_section(section):
                print(f"✓ Section [{section}] present")
            else:
                print(f"✗ Section [{section}] missing")
                return False

        # Check data directories
        directories = [
            'data/uploads',
            'data/notes',
            'data/images',
            'data/conversations',
            'data/metadata',
            'logs'
        ]

        for directory in directories:
            path = Path(directory)
            if path.exists():
                print(f"✓ Directory {directory} exists")
            else:
                print(f"⚠ Directory {directory} does not exist (will be created on startup)")

        print("\n✓ Configuration validation passed")
        return True

    except Exception as e:
        print(f"\n✗ Configuration validation failed: {e}")
        return False


async def health_check():
    """Perform health check"""
    print("Performing health check...\n")

    # Setup logger
    config = get_config()
    setup_logger(log_level=config.get('app.log_level', 'INFO'))

    all_healthy = True

    # Check LLM provider
    print("Checking LLM provider...")
    try:
        provider = create_llm_provider()
        healthy = await provider.health_check()

        if healthy:
            print("✓ LLM provider is reachable")
        else:
            print("✗ LLM provider is not reachable")
            all_healthy = False
    except Exception as e:
        print(f"✗ LLM provider error: {e}")
        all_healthy = False

    # Check vector store
    print("\nChecking vector store...")
    try:
        store = create_vector_store()
        count = store.get_count()
        print(f"✓ Vector store initialized ({count} chunks)")
    except Exception as e:
        print(f"✗ Vector store error: {e}")
        all_healthy = False

    # Check embedding service
    print("\nChecking embedding service...")
    try:
        embedding_service = get_embedding_service()
        test_embedding = embedding_service.embed_text("test")
        print(f"✓ Embedding service working (dimension: {len(test_embedding)})")
    except Exception as e:
        print(f"✗ Embedding service error: {e}")
        all_healthy = False

    if all_healthy:
        print("\n✓ All health checks passed")
        return 0
    else:
        print("\n✗ Some health checks failed")
        return 1


async def reindex_documents(incremental: bool = False):
    """Reindex all documents or incrementally index new/changed documents"""
    print(f"Starting {'incremental' if incremental else 'full'} reindex...")

    config = get_config()
    setup_logger(log_level=config.get('app.log_level', 'INFO'))
    logger = get_logger()

    # Get services
    processor = get_document_processor()
    embedding_service = get_embedding_service()
    vector_store = create_vector_store()

    if not incremental:
        # Clear existing index
        print("Clearing existing index...")
        vector_store.clear()

    # Process uploaded documents
    upload_dir = Path(config.get('documents.upload_directory', 'data/uploads'))
    if upload_dir.exists():
        files = list(upload_dir.glob('*.*'))
        print(f"Found {len(files)} files in upload directory")

        for file_path in files:
            if file_path.suffix.lower() in ['.pdf', '.txt']:
                print(f"Processing {file_path.name}...")

                try:
                    document_id = file_path.stem.split('_')[0]  # Extract ID from filename

                    if file_path.suffix.lower() == '.pdf':
                        document, chunks = processor.process_pdf(str(file_path), document_id)
                    else:
                        document, chunks = processor.process_txt(str(file_path), document_id)

                    # Generate embeddings
                    texts = [chunk.content for chunk in chunks]
                    embeddings = embedding_service.embed_batch(texts)

                    # Add embeddings to chunks
                    for chunk, embedding in zip(chunks, embeddings):
                        chunk.embedding = embedding
                        chunk.metadata['filename'] = file_path.name

                    # Store in vector database
                    vector_store.add_documents(chunks)

                    print(f"  ✓ Indexed {len(chunks)} chunks")

                except Exception as e:
                    print(f"  ✗ Error processing {file_path.name}: {e}")

    # Process notes
    notes_dir = Path(config.get('documents.notes_directory', 'data/notes'))
    if notes_dir.exists():
        notes = list(notes_dir.glob('*.txt'))
        print(f"\nFound {len(notes)} notes")

        for note_path in notes:
            print(f"Processing {note_path.name}...")

            try:
                document_id = f"note_{note_path.name}"
                document, chunks = processor.process_txt(str(note_path), document_id)

                # Generate embeddings
                texts = [chunk.content for chunk in chunks]
                embeddings = embedding_service.embed_batch(texts)

                # Add embeddings
                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding
                    chunk.metadata['filename'] = note_path.name
                    chunk.metadata['type'] = 'note'

                # Store in vector database
                vector_store.add_documents(chunks)

                print(f"  ✓ Indexed {len(chunks)} chunks")

            except Exception as e:
                print(f"  ✗ Error processing {note_path.name}: {e}")

    total_chunks = vector_store.get_count()
    print(f"\n✓ Reindexing complete. Total chunks: {total_chunks}")
    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="RPG Assistant CLI")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Validate config command
    subparsers.add_parser('validate-config', help='Validate configuration files')

    # Health check command
    subparsers.add_parser('health', help='Perform health check')

    # Reindex command
    reindex_parser = subparsers.add_parser('reindex', help='Reindex documents')
    reindex_parser.add_argument(
        '--full',
        action='store_true',
        help='Perform full reindex (clear existing index)'
    )
    reindex_parser.add_argument(
        '--incremental',
        action='store_true',
        help='Perform incremental reindex (keep existing index)'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'validate-config':
        result = validate_config()
        return 0 if result else 1

    elif args.command == 'health':
        return asyncio.run(health_check())

    elif args.command == 'reindex':
        incremental = args.incremental or not args.full
        return asyncio.run(reindex_documents(incremental))

    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
