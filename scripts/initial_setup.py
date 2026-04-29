"""
Initial Setup Script — First-time project initialization.

This script:
1. Creates all required directories
2. Validates configuration (.env file)
3. Checks for PDF documents
4. Builds the vector database
5. Runs a test query to verify everything works

Usage:
    python scripts/initial_setup.py
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_env_file() -> bool:
    """Check if .env file exists with required API key."""
    from src.utils.config import BASE_DIR

    env_path = BASE_DIR / ".env"
    env_example = BASE_DIR / ".env.example"

    if not env_path.exists():
        print("❌ No .env file found!")
        print(f"   Creating .env from .env.example...")
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_path)
            print(f"   ✅ Created .env file at {env_path}")
            print("   ⚠️  Please edit .env and add your GROQ_API_KEY")
            print("   Get a free key at: https://console.groq.com/keys")
            return False
        else:
            print("   ❌ .env.example not found either!")
            return False

    # Check if API key is set
    from src.utils.config import settings
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        print("⚠️  GROQ_API_KEY not set in .env file")
        print("   Get a free key at: https://console.groq.com/keys")
        print(f"   Edit: {env_path}")
        return False

    print("✅ Configuration validated")
    return True


def check_pdfs() -> bool:
    """Check if PDF documents exist in the data directory."""
    from src.utils.config import settings

    pdf_dir = settings.raw_pdf_dir
    pdfs = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []

    if not pdfs:
        print(f"\n📄 No PDF files found in {pdf_dir}")
        print("   To get started, download 485 visa documents from:")
        print("   https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485")
        print(f"\n   Place PDF files in: {pdf_dir}")
        return False

    print(f"✅ Found {len(pdfs)} PDF files")
    for pdf in pdfs:
        print(f"   - {pdf.name}")
    return True


def build_vectorstore() -> bool:
    """Build the vector database from PDF documents."""
    try:
        from src.ingestion.pdf_loader import PDFLoader
        from src.ingestion.text_chunker import TextChunker
        from src.ingestion.vectorstore_manager import VectorStoreManager
        from src.utils.logger import logger

        print("\n📚 Building vector database...")

        # Step 1: Load PDFs
        print("   Step 1/3: Extracting text from PDFs...")
        loader = PDFLoader()
        pages = loader.load_all_pdfs()

        if not pages:
            print("   ❌ No content extracted from PDFs")
            return False
        print(f"   ✅ Extracted {len(pages)} pages")

        # Step 2: Chunk text
        print("   Step 2/3: Chunking text...")
        chunker = TextChunker()
        chunks = chunker.chunk_document(pages)
        print(f"   ✅ Created {len(chunks)} chunks")

        # Step 3: Add to vectorstore
        print("   Step 3/3: Building embeddings and storing in ChromaDB...")
        vs = VectorStoreManager()
        vs.add_chunks(chunks)
        stats = vs.get_collection_stats()
        print(f"   ✅ Vector database built: {stats['total_chunks']} chunks")

        return True

    except Exception as e:
        print(f"   ❌ Error building vectorstore: {e}")
        return False


def setup_database() -> bool:
    """Create SQLite database tables."""
    try:
        from src.utils.db_manager import DatabaseManager

        print("\n🗄️  Setting up change tracking database...")
        db = DatabaseManager()
        db.create_tables()
        print("✅ Database tables created")
        return True

    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        return False


def test_query() -> bool:
    """Run a test query to verify the RAG pipeline works."""
    try:
        from src.retrieval.retriever import Retriever

        print("\n🔍 Running test query...")
        retriever = Retriever()
        results = retriever.retrieve(
            query="What are the requirements for the 485 visa?",
            n_results=3,
        )

        if results.total_found > 0:
            print(f"✅ Test query returned {results.total_found} results")
            print(f"   Top result: {results.results[0].source} "
                  f"(Page {results.results[0].page_number}, "
                  f"Relevance: {results.results[0].relevance_score:.1%})")
            return True
        else:
            print("⚠️  Test query returned no results")
            return False

    except Exception as e:
        print(f"⚠️  Test query failed: {e}")
        return False


def main():
    """Run the full setup process."""
    print("=" * 60)
    print("🛂 485 Visa Intelligence System — Initial Setup")
    print("=" * 60)

    # Step 1: Check configuration
    print("\n📋 Step 1: Checking configuration...")
    env_ok = check_env_file()

    # Step 2: Create directories
    print("\n📁 Step 2: Creating directories...")
    from src.utils.config import ensure_directories
    ensure_directories()
    print("✅ Directories created")

    # Step 3: Check for PDFs
    print("\n📄 Step 3: Checking for PDF documents...")
    pdfs_ok = check_pdfs()

    # Step 4: Build vectorstore (only if PDFs exist)
    vs_ok = False
    if pdfs_ok:
        vs_ok = build_vectorstore()
    else:
        print("⏭️  Skipping vectorstore build (no PDFs)")

    # Step 5: Setup database
    print("\n🗄️  Step 4: Setting up change tracking database...")
    db_ok = setup_database()

    # Step 6: Test query (only if vectorstore was built)
    if vs_ok:
        test_query()

    # Summary
    print("\n" + "=" * 60)
    print("📋 Setup Summary")
    print("=" * 60)
    print(f"  Configuration: {'✅' if env_ok else '❌ (edit .env file)'}")
    print(f"  PDF Documents:  {'✅' if pdfs_ok else '❌ (add PDFs to data/raw/pdfs/)'}")
    print(f"  Vector Store:   {'✅' if vs_ok else '⏭️  (pending PDFs)'}")
    print(f"  Database:       {'✅' if db_ok else '❌'}")

    if env_ok and pdfs_ok and vs_ok and db_ok:
        print("\n🎉 Everything is set up! Run the app:")
        print("   streamlit run app/streamlit_app.py")
    elif not env_ok:
        print("\n📝 Next steps:")
        print("   1. Edit .env and add your GROQ_API_KEY")
        print("   2. Re-run this script")
    elif not pdfs_ok:
        print("\n📝 Next steps:")
        print("   1. Download 485 visa PDFs from immi.homeaffairs.gov.au")
        print("   2. Place them in data/raw/pdfs/")
        print("   3. Re-run this script")
    else:
        print("\n📝 Check the errors above and re-run this script.")

    print()


if __name__ == "__main__":
    main()