# OrbitGuide

**Asystent ds. Prawa Kosmicznego oparty na AI**

> Projekt stworzony na hackathon BITEhack 2026 przez zespół **Umiski** | Kategoria: Sztuczna Inteligencja

---

## Opis Projektu

OrbitGuide to aplikacja wykorzystująca technologię **RAG (Retrieval Augmented Generation)** do udzielania odpowiedzi na pytania dotyczące prawa kosmicznego. System analizuje dokumenty regulacyjne organizacji takich jako NASA, ESA, UNOOSA i ITU, a następnie generuje odpowiedzi z podaniem źródeł.

### Główne funkcje:
- **Wyszukiwanie semantyczne** w bazie dokumentów prawnych
- **Generowanie odpowiedzi** przez LLM (Groq/Llama 3.1)
- **Podawanie źródeł** z numerami stron
- **Wskaźnik pewności** odpowiedzi
- **Ekspansja zapytań** (PL + EN) dla lepszego dopasowania

---

## 🏗️ Architektura

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│   Dokumenty PDF │────▶│   Ingestion      │────▶│   ChromaDB    │
│   (NASA, ESA)   │     │   (embeddingi)   │     │   (wektory)   │
└─────────────────┘     └──────────────────┘     └───────┬───────┘
                                                         │
┌─────────────────┐     ┌──────────────────┐     ┌───────▼───────┐
│   Użytkownik    │────▶│   Streamlit UI   │────▶│   Brain RAG   │
│                 │     │   (app.py)       │     │   (brain.py)  │
└─────────────────┘     └──────────────────┘     └───────┬───────┘
                                                         │
                        ┌──────────────────┐     ┌───────▼───────┐
                        │   Odpowiedź      │◀────│   LLM (Groq)  │
                        │   + źródła       │     │   Llama 3.1   │
                        └──────────────────┘     └───────────────┘
```

```

---

## 🧠 Logika Systemu

### 🔍 Wskaźnik Pewności (Confidence Score)
System ocenia pewność odpowiedzi na podstawie liczby unikalnych dokumentów źródłowych użytych do wygenerowania odpowiedzi:
- **Wysoka (100%)**: 5 lub więcej unikalnych źródeł
- **Średnia (60-80%)**: 3-4 źródła
- **Niska (<60%)**: 1-2 źródła

### 🗣️ Kontekst Rozmowy
LLM pamięta historię czatu (ostatnie 3 wymiany wiadomości), co pozwala na dopytywanie o szczegóły (np. *"A jakie są tego koszty?"* po pytaniu o rejestrację).

### 🛡️ Filtrowanie Domeny
OrbitGuide posiada wbudowane mechanizmy:
- **Small-talk detector**: Wykrywa powitania i luźne rozmowy, odpowiadając naturalnie bez przeszukiwania bazy.
- **Out-of-domain guard**: Odmawia odpowiedzi na pytania niezwiązane z prawem kosmicznym.

---

## 🛠️ Instalacja

### 1. Klonowanie repozytorium

```bash
git clone https://github.com/YOUR_USERNAME/OrbitGuide.git
cd OrbitGuide
```

### 2. Środowisko wirtualne

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# lub: venv\Scripts\activate  # Windows
```

### 3. Instalacja zależności

```bash
pip install -r requirements.txt
```

### 4. Konfiguracja

Skopiuj plik `.env.example` jako `.env` i uzupełnij klucze API:

```bash
cp .env.example .env
```

Wymagane klucze:
- `GOOGLE_API_KEY` - dla embeddingów (gemini-embedding-001)
- `GROQ_API_KEY` - dla LLM (llama-3.1-8b-instant)

---

## 🚀 Uruchomienie

### 1. Indeksowanie dokumentów (jednorazowo)

```bash
python -m src.ingestion
```

Ten krok tworzy bazę wektorową ChromaDB z dokumentów PDF w folderze `data/`.

### 2. Uruchomienie aplikacji

```bash
streamlit run app.py
```

Aplikacja będzie dostępna pod adresem: `http://localhost:8501`

---

## 📊 Ewaluacja

System zawiera moduł ewaluacji jakości odpowiedzi:

```bash
python -m src.eval
```
> **Uwaga:** Należy uruchamiać jako moduł (`python -m ...`) z głównego folderu projektu.

### Metryki:
| Metryka | Opis | Skala |
|---------|------|-------|
| **Wierność (Faithfulness)** | Czy odpowiedź wynika z kontekstu? | 1-5 |
| **Trafność (Relevancy)** | Czy odpowiedź odpowiada na pytanie? | 1-5 |
| **Obsługa poza domeną** | Czy system odmawia przy pytaniach spoza tematu? | 1-5 |

---

## 📁 Struktura Projektu

```
OrbitGuide/
├── app.py              # Interfejs Streamlit
├── requirements.txt    # Zależności Python
├── .env.example        # Wzór konfiguracji
├── .gitignore
├── README.md
├── src/
│   ├── brain.py        # Logika RAG + Chain of Thought
│   ├── ingestion.py    # Pipeline indeksowania PDF
│   ├── eval.py         # Moduł ewaluacji
│   └── utils.py        # Konfiguracja
└── data/
    ├── *.pdf           # Dokumenty źródłowe
    └── chroma_db/      # Baza wektorowa (generowana)
```

---

## 📚 Źródła Danych

Dokumenty przeanalizowane przez system:

| Dokument | Organizacja |
|----------|-------------|
| Outer Space Treaty (1967) | ONZ |
| Liability Convention | UNOOSA |
| Registration Convention | UNOOSA |
| Moon Agreement | ONZ |
| Space Debris Mitigation Guidelines | ESA |
| ITU Regulatory Procedures | ITU |
| Guidelines for Long-term Sustainability | COPUOS |

---

## 🧰 Technologie

- **LangChain** - Orkiestracja RAG
- **ChromaDB** - Baza wektorowa
- **Groq** - Inference LLM (Llama 3.1)
- **Google AI** - Embeddingi (gemini-embedding-001)
- **Streamlit** - Interfejs użytkownika

---

## 👥 Zespół Umiski

Projekt stworzony na hackathon **BITEhack 2026**.

---

## 📄 Licencja

MIT License
