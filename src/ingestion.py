import os
import shutil
import time

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.utils import get_config


def run_ingestion():
    config = get_config()

    # 1. Przygotowanie ścieżek
    DATA_PATH = "data/"  # PDFy od NASA, ESA, UNOOSA
    CHROMA_PATH = config["chroma_path"]

    # Czyszczenie starej bazy, żeby nie dublować danych
    if os.path.exists(CHROMA_PATH):
        print(f"Usuwanie starej bazy w {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)

    # 2. Ładowanie dokumentów
    print("Ładowanie dokumentów z folderu data/...")
    loader = DirectoryLoader(
        DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )

    raw_documents = loader.load()
    print(f"Załadowano {len(raw_documents)} stron dokumentacji.")

    # 3. Podział tekstu na mniejsze fragmenty (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        length_function=len,
        add_start_index=True,
    )

    print("Dzielenie dokumentów na fragmenty...")
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Utworzono {len(chunks)} fragmentów wiedzy.")

    # 4. Inicjalizacja modelu Embeddingów (Google)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=config["google_api_key"]
    )

    # 5. Budowa bazy wektorowej ChromaDB z BATCHINGIEM
    # Free tier Google: 100 requests/minute, więc przetwarzamy w małych paczkach
    BATCH_SIZE = 20  # Ilość dokumentów na raz
    DELAY_BETWEEN_BATCHES = 15  # Sekund między paczkami (aby nie przekroczyć limitu)
    
    print(f"\nBudowanie bazy ChromaDB w {CHROMA_PATH}...")
    print(f"📦 Batch size: {BATCH_SIZE} | ⏱️ Opóźnienie: {DELAY_BETWEEN_BATCHES}s")
    print(f"⚠️ Free tier API - to może potrwać ~{len(chunks) // BATCH_SIZE * DELAY_BETWEEN_BATCHES // 60 + 1} minut\n")

    # Pierwsza paczka - tworzy bazę
    first_batch = chunks[:BATCH_SIZE]
    vectorstore = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"✓ Paczka 1/{(len(chunks) - 1) // BATCH_SIZE + 1} - {len(first_batch)} fragmentów")
    
    # Pozostałe paczki - dodajemy do istniejącej bazy
    remaining_chunks = chunks[BATCH_SIZE:]
    total_batches = (len(remaining_chunks) - 1) // BATCH_SIZE + 1 if remaining_chunks else 0
    
    for i in range(0, len(remaining_chunks), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 2  # +2 bo pierwsza paczka już przetworzona
        batch = remaining_chunks[i:i + BATCH_SIZE]
        
        # Opóźnienie przed kolejną paczką (aby nie przekroczyć limitu API)
        print(f"⏳ Czekam {DELAY_BETWEEN_BATCHES}s przed kolejną paczką...")
        time.sleep(DELAY_BETWEEN_BATCHES)
        
        try:
            vectorstore.add_documents(batch)
            print(f"✓ Paczka {batch_num}/{total_batches + 1} - {len(batch)} fragmentów")
        except Exception as e:
            print(f"❌ Błąd w paczce {batch_num}: {e}")
            print("⏳ Czekam 60s i próbuję ponownie...")
            time.sleep(60)
            try:
                vectorstore.add_documents(batch)
                print(f"✓ Paczka {batch_num}/{total_batches + 1} - ponowna próba udana")
            except Exception as e2:
                print(f"❌ Trwały błąd: {e2}")
                raise

    print("\n" + "=" * 40)
    print("✅ INGESTION ZAKOŃCZONE SUKCESEM!")
    print(f"📊 Zindeksowano: {len(chunks)} fragmentów")
    print(f"📁 Lokalizacja bazy: {CHROMA_PATH}")
    print("=" * 40)


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
        print(
            "Utworzono folder 'data/'. Wrzuć tam swoje pliki PDF i uruchom ponownie."
        )
    else:
        run_ingestion()
