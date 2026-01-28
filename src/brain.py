import os
from operator import itemgetter

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

from src.utils import get_config

_CACHED_CHAIN = None
_VECTORSTORE = None
use_groq = True  # False, aby używać modeli Google zamiast Groq
cot_system_template = """Jesteś OrbitGuide - asystentem pomagającym firmom i startupom w zrozumieniu prawa kosmicznego.

{chat_history}

Dokumenty źródłowe:
{context}

Pytanie: {question}

Zasady odpowiedzi:
- NIE PRZEDSTAWIAJ SIĘ - przejdź od razu do odpowiedzi
- Pisz po polsku z poprawną gramatyką
- Jeśli pytanie nawiązuje do poprzedniej rozmowy, wykorzystaj kontekst
- Zakładaj, że użytkownik to przedstawiciel firmy planującej misję kosmiczną
- Tłumacz język prawniczy na praktyczne wskazówki
- Jeśli pytanie nie dotyczy prawa kosmicznego, powiedz krótko że specjalizujesz się tylko w tej dziedzinie
- Jeśli to powitanie, odpowiedz krótko
- Opieraj się tylko na dokumentach, nie zmyślaj

Odpowiedź:"""

COT_PROMPT = PromptTemplate(
    template=cot_system_template, input_variables=["context", "question", "chat_history"]
)


def get_resources():
    """Inicjalizuje i cache'uje bazę oraz embeddingi"""
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    config = get_config()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config["embedding_model"], google_api_key=config["google_api_key"]
    )

    _VECTORSTORE = Chroma(
        persist_directory=config["chroma_path"], embedding_function=embeddings
    )
    return _VECTORSTORE


def get_rag_chain():
    global _CACHED_CHAIN
    if _CACHED_CHAIN is not None:
        return _CACHED_CHAIN

    config = get_config()

    if not os.path.exists(config["chroma_path"]):
        raise FileNotFoundError("Brak bazy. Uruchom najpierw ingestion.")

    # Embeddingi takie same jak przy tworzeniu bazy
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="retrieval_query",
        google_api_key=config["google_api_key"],
    )

    vectorstore = Chroma(
        persist_directory=config["chroma_path"],
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": config["retrieval_k"]})

    # Inicjalizacja LLM (Groq lub Google)
    if not use_groq:
        llm = ChatGoogleGenerativeAI(
            model=config["llm_model"],
            google_api_key=config["google_api_key"],
            temperature=config["temperature"],
        )
    else:
        llm = ChatGroq(
            model_name=config["llm_model"],
            temperature=config["temperature"],
            max_tokens=1024,
        )

    # --- FUNKCJA EKSPANSJI ZAPYTANIA ---
    def get_expanded_docs_and_context(query_dict):
        """
        Zwraca słownik z:
        - 'context': połączony tekst dokumentów (dla LLM)
        - 'docs': lista dokumentów z metadanymi (dla źródeł)
        """
        question = query_dict["question"]

        # Prompt do generowania wariantów pytań (PL + EN dla lepszego dopasowania do dokumentów)
        expansion_prompt = f"""Jesteś ekspertem od wyszukiwania w dokumentacji prawnej NASA/ESA/UNOOSA.
        
Pytanie użytkownika: {question}

Wygeneruj dokładnie 3 alternatywne warianty tego pytania:
1. Wariant techniczny po polsku (używając terminologii prawniczej)
2. Wariant po angielsku (dokumenty źródłowe są głównie w języku angielskim)
3. Wariant po angielsku z kluczowymi terminami (space law, treaty, liability, registration, etc.)

Zwróć TYLKO 3 warianty, każdy w nowej linii, bez numeracji ani dodatkowych wyjaśnień."""

        try:
            response = llm.invoke(expansion_prompt)
            expanded_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            expanded_queries = expanded_text.strip().split("\n")
        except Exception as e:
            print(f"⚠️ Problem z ekspansją: {e}")
            expanded_queries = []

        all_docs = []
        # Szukamy dla pytania oryginalnego ORAZ wariantów
        search_queries = [question] + [q.strip() for q in expanded_queries if q.strip()]

        for q in search_queries:
            docs = retriever.invoke(q)
            all_docs.extend(docs)

        # Usuwanie duplikatów (zachowując dokumenty z metadanymi)
        unique_contents = set()
        final_docs = []
        for doc in all_docs:
            if doc.page_content not in unique_contents:
                unique_contents.add(doc.page_content)
                final_docs.append(doc)
        
        # Ograniczenie do max 5 dokumentów aby nie przekroczyć limitu tokenów
        final_docs = final_docs[:5]

        context_text = "\n\n".join(doc.page_content for doc in final_docs)
        
        return {
            "context": context_text,
            "docs": final_docs
        }

    # --- GLÓWNY PIPELINE ---
    def run_chain(query_dict):
        """Pipeline który zwraca odpowiedź LLM wraz z dokumentami źródłowymi."""
        result = get_expanded_docs_and_context(query_dict)
        context = result["context"]
        docs = result["docs"]
        
        # Pobierz historię rozmowy (jeśli jest)
        chat_history = query_dict.get("chat_history", "")
        
        # Generowanie odpowiedzi przez LLM
        prompt_text = COT_PROMPT.format(
            context=context, 
            question=query_dict["question"],
            chat_history=chat_history
        )
        response = llm.invoke(prompt_text)
        answer = response.content if hasattr(response, "content") else str(response)
        
        return {
            "answer": answer,
            "context": context,
            "docs": docs
        }

    _CACHED_CHAIN = run_chain
    return _CACHED_CHAIN


def get_astro_answer(query_text, messages=None):
    """
    Główna funkcja do przetwarzania pytań użytkownika.
    Zwraca odpowiedź, źródła i confidence - wszystko spójne z jednego wyszukiwania.
    
    Args:
        query_text: Pytanie użytkownika
        messages: Lista poprzednich wiadomości [{"role": "user/assistant", "content": "..."}]
    """
    # Formatowanie historii rozmowy (ostatnie 3 wymiany)
    chat_history = ""
    if messages and len(messages) > 0:
        # Bierzemy ostatnie 6 wiadomości (3 wymiany user+assistant)
        recent = messages[-6:]
        history_lines = []
        for m in recent:
            role = "Użytkownik" if m["role"] == "user" else "Asystent"
            # Skracamy długie odpowiedzi
            content = m["content"][:200] + "..." if len(m["content"]) > 200 else m["content"]
            history_lines.append(f"{role}: {content}")
        if history_lines:
            chat_history = "Historia rozmowy:\n" + "\n".join(history_lines)
    
    # Detekcja powitań i small-talk - nie potrzebujemy źródeł
    greetings = [
        "hej", "cześć", "siema", "witaj", "dzień dobry", "dobry wieczór",
        "hello", "hi", "hey", "co słychać", "jak się masz", "co tam",
        "co robisz", "jak leci", "co u ciebie", "siemka", "yo", "elo",
        "witam", "dzięki", "dziękuję", "ok", "okej", "super", "fajnie"
    ]
    query_lower = query_text.lower().strip()
    
    is_greeting = any(query_lower.startswith(g) or query_lower == g for g in greetings)
    
    # Dodatkowa detekcja: bardzo krótkie pytania bez słów kluczowych to prawdopodobnie small-talk
    space_keywords = ["satelit", "kosm", "orbi", "rakiet", "rejestr", "traktat", "unoosa", "esa", "nasa", "prawo"]
    has_space_keyword = any(kw in query_lower for kw in space_keywords)
    is_short_non_space = len(query_lower.split()) <= 3 and not has_space_keyword
    
    is_small_talk = is_greeting or is_short_non_space
    
    chain = get_rag_chain()
    result = chain({"question": query_text, "chat_history": chat_history})
    
    # Jeśli to powitanie/small-talk - nie pokazuj źródeł
    if is_small_talk:
        return {
            "answer": result["answer"],
            "sources": [],
            "confidence": 100,
            "is_greeting": True,
        }
    
    docs = result.get("docs", [])
    
    if not docs:
        return {
            "answer": "Brak danych w bazie dokumentacji.",
            "sources": [],
            "confidence": 0,
        }

    # Obliczanie źródeł z metadanych dokumentów (te same dokumenty które LLM widział)
    detailed_sources = []
    for doc in docs:
        name = doc.metadata.get("source", "Nieznany plik")
        page = doc.metadata.get("page", 0) + 1
        # Usuwamy ścieżkę, zostawiamy tylko nazwę pliku
        if "/" in name:
            name = name.split("/")[-1]
        detailed_sources.append({
            "text": f"📄 {name} (str. {page})",
            "score": 100  # Dokumenty już są przefiltrowane jako najbardziej trafne
        })

    # Confidence na podstawie liczby znalezionych dokumentów
    # Więcej unikalnych dokumentów = większa pewność
    unique_sources = len(set(d["text"] for d in detailed_sources))
    confidence = min(100, unique_sources * 20)  # 5 unikalnych źródeł = 100%

    return {
        "answer": result["answer"],
        "sources": detailed_sources,
        "confidence": confidence,
    }



def quick_chat():
    # Kody kolorów do terminala
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"

    print(f"{HEADER}\n ORBITGUIDE - EXPERT EVALUATION{ENDC}")
    print("Działasz jako: Lead Dev / Math Specialist")
    print("-" * 60)

    try:
        while True:
            query = input(f"\n{BOLD}Ty:{ENDC} ")
            if query.lower() in ["q", "exit"]:
                break

            print(
                f"{YELLOW}Analizuję trajektorię zapytania i przeszukuję bazę wektorową...{ENDC}",
                end="\r",
            )

            # Wywołujemy logikę RAG
            data = get_astro_answer(query)

            # Czyścimy linię ładowania
            print(" " * 80, end="\r")

            # Kolorowanie statusu
            if data["confidence"] > 80:
                status_color = GREEN
                status_text = "PEWNY"
            elif data["confidence"] > 60:
                status_color = YELLOW
                status_text = "ŚREDNI"
            else:
                status_color = RED
                status_text = "NIEPEWNY"

            print(
                f"\n🤖 AstroGuide [{status_color}{status_text}{ENDC} - {data['confidence']:.1f}%]:"
            )

            # --- PARSOWANIE CHAIN OF THOUGHT ---
            raw_response = data["answer"]

            # Sprawdzamy, czy model wygenerował sekcję odpowiedzi końcowej
            split_keywords = ["Odpowiedź:", "Podsumowując:", "Wnioski:", "Answer:"]
            split_idx = -1

            for keyword in split_keywords:
                idx = raw_response.rfind(keyword)
                if idx != -1:
                    split_idx = idx
                    break

            if split_idx != -1:
                thinking_process = raw_response[:split_idx].strip()
                final_answer = raw_response[split_idx:].strip()

                print(f"{BLUE}PROCES MYŚLOWY:{ENDC}")
                print(f"{BLUE}{thinking_process}{ENDC}")
                print("-" * 30)
                print(f"{BOLD}{final_answer}{ENDC}")
            else:
                print(raw_response)

            # --- ANALIZA ŹRÓDEŁ ---
            print(f"\n{HEADER}{'=' * 20} ANALIZA MATEMATYCZNA ŹRÓDEŁ {'=' * 20}{ENDC}")
            if not data["sources"]:
                print(f"{RED}Brak źródeł spełniających kryteria.{ENDC}")

            for i, src in enumerate(data["sources"], 1):
                # Wyświetlamy trafność każdego chunka
                print(f"[{i}] {src['text']} | Trafność: {src['score']}%")

            print("-" * 60)

    except Exception as e:
        print(f"{RED}Błąd krytyczny systemu: {e}{ENDC}")


if __name__ == "__main__":
    quick_chat()
