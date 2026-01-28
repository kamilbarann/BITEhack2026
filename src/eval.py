import json
import os
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.brain import get_rag_chain
from src.utils import get_config

# Konfiguracja Sędziego (Groq)
config = get_config()

api_key = os.getenv("GROQ_API_KEY") or config.get("groq_api_key")

judge_llm = ChatGroq(
    model_name=config["judge_model"],
    temperature=0,
    api_key=api_key,
)


def extract_json_from_text(text):
    """Wyciąga JSON z odpowiedzi LLM."""
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: szukanie ręczne dla skali 1-5
    score_match = re.search(r'"score"\s*:\s*([1-5])', text)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)

    if score_match and reason_match:
        return {"score": int(score_match.group(1)), "reason": reason_match.group(1)}

    return {"score": 1, "reason": "Błąd parsowania odpowiedzi sędziego"}


def evaluate_faithfulness(answer, context_text):
    """
    Sprawdza Wierność (Faithfulness): Czy odpowiedź wynika TYLKO z kontekstu?
    Skala: 1-5
    """
    prompt = ChatPromptTemplate.from_template("""
    Jesteś surowym sędzią AI oceniającym systemy RAG. Oceniasz "Wierność" (Faithfulness).

    KONTEKST ŹRÓDŁOWY:
    {context}

    ODPOWIEDŹ SYSTEMU:
    {answer}

    ZADANIE: Oceń w skali 1-5, czy odpowiedź wynika TYLKO z podanego kontekstu.

    SKALA OCEN:
    5 = Doskonale - Wszystkie informacje pochodzą bezpośrednio z kontekstu
    4 = Bardzo dobrze - Prawie wszystko z kontekstu, minimalne rozszerzenia
    3 = Dobrze - Większość z kontekstu, ale są drobne dodatki z wiedzy ogólnej
    2 = Słabo - Dużo informacji spoza kontekstu, model rozszerza odpowiedź
    1 = Bardzo słabo - Model zmyśla lub ignoruje kontekst

    Odpowiedz WYŁĄCZNIE w JSON: {{"score": <1-5>, "reason": "<krótki powód po polsku>"}}
    """)

    chain = prompt | judge_llm | StrOutputParser()
    raw_response = chain.invoke({"answer": answer, "context": context_text})
    return extract_json_from_text(raw_response)


def evaluate_relevancy(question, answer):
    """
    Sprawdza Trafność (Relevancy): Czy odpowiedź odpowiada na pytanie?
    Skala: 1-5
    """
    prompt = ChatPromptTemplate.from_template("""
    Jesteś surowym sędzią AI oceniającym systemy RAG. Oceniasz "Trafność" (Relevancy).

    PYTANIE UŻYTKOWNIKA: {question}
    ODPOWIEDŹ SYSTEMU: {answer}

    ZADANIE: Oceń w skali 1-5, czy odpowiedź jest trafna i odpowiada na pytanie.

    SKALA OCEN:
    5 = Doskonale - Pełna, wyczerpująca odpowiedź na pytanie
    4 = Bardzo dobrze - Dobra odpowiedź, ale mogłaby być pełniejsza
    3 = Dobrze - Odpowiada na pytanie, ale brakuje ważnych szczegółów
    2 = Słabo - Częściowo odpowiada, dużo nieistotnych informacji
    1 = Bardzo słabo - Nie odpowiada na pytanie lub odpowiedź jest całkowicie nietrafiona

    Odpowiedz WYŁĄCZNIE w JSON: {{"score": <1-5>, "reason": "<krótki powód po polsku>"}}
    """)

    chain = prompt | judge_llm | StrOutputParser()
    raw_response = chain.invoke({"question": question, "answer": answer})
    return extract_json_from_text(raw_response)


def evaluate_out_of_domain(question, answer):
    """
    Sprawdza obsługę pytań poza domeną: Czy system poprawnie odmawia odpowiedzi?
    Skala: 1-5
    """
    prompt = ChatPromptTemplate.from_template("""
    Jesteś surowym sędzią AI. Oceniasz jak system RAG radzi sobie z pytaniami POZA swoją domeną.

    DOMENA SYSTEMU: Prawo kosmiczne, regulacje dotyczące przestrzeni kosmicznej, traktaty ONZ.
    
    PYTANIE (potencjalnie poza domeną): {question}
    ODPOWIEDŹ SYSTEMU: {answer}

    ZADANIE: Oceń w skali 1-5, czy system poprawnie obsłużył to pytanie.

    Jeśli pytanie jest POZA domeną (np. przepisy kulinarne, sport, pogoda):
    5 = System grzecznie odmówił i wyjaśnił, że pytanie jest poza jego specjalizacją
    3 = System próbował odpowiedzieć, ale zaznaczył niepewność
    1 = System odpowiedział jakby znał temat, zmyślając informacje

    Jeśli pytanie jest W domenie:
    5 = System odpowiedział merytorycznie
    1 = System niepotrzebnie odmówił

    Odpowiedz WYŁĄCZNIE w JSON: {{"score": <1-5>, "in_domain": <true/false>, "reason": "<krótki powód>"}}
    """)

    chain = prompt | judge_llm | StrOutputParser()
    raw_response = chain.invoke({"question": question, "answer": answer})
    result = extract_json_from_text(raw_response)
    
    # Zapewnij że in_domain jest w wyniku
    if "in_domain" not in result:
        result["in_domain"] = True
    
    return result


def run_evaluation():
    """Uruchamia pełną ewaluację systemu RAG."""
    print("\n" + "=" * 60)
    print("🚀 ORBITGUIDE - EWALUACJA SYSTEMU RAG")
    print("=" * 60)
    print("Sędzia: Groq/Llama3 | Skala: 1-5")
    print("-" * 60)

    # Pytania testowe - podzielone na kategorie
    test_questions = [
        # Pytania w domenie
        {"q": "Czym jest obiekt kosmiczny w świetle prawa?", "type": "in_domain"},
        {"q": "Kto odpowiada za szkody wyrządzone przez satelitę na Ziemi?", "type": "in_domain"},
        {"q": "Czy Księżyc może należeć do prywatnej firmy?", "type": "in_domain"},
        {"q": "Jakie są wymagania rejestracji satelity w UNOOSA?", "type": "in_domain"},
        # Pytania poza domeną
        {"q": "Jaki jest przepis na ciasto marchewkowe?", "type": "out_of_domain"},
        {"q": "Kto wygrał ostatnie mistrzostwa świata w piłce nożnej?", "type": "out_of_domain"},
    ]

    rag_chain = get_rag_chain()

    results = {
        "faithfulness": [],
        "relevancy": [],
        "out_of_domain": [],
    }

    for item in test_questions:
        q = item["q"]
        q_type = item["type"]
        
        print(f"\n📌 Pytanie [{q_type.upper()}]: {q}")

        try:
            response = rag_chain({"question": q})
            answer = response["answer"]
            context_text = response.get("context", "Brak kontekstu.")

            # Skróć odpowiedź do wyświetlenia
            answer_short = answer[:150] + "..." if len(answer) > 150 else answer
            print(f"   💬 Odpowiedź: {answer_short}")

            # Ocena wierności i trafności
            faith_result = evaluate_faithfulness(answer, context_text)
            rel_result = evaluate_relevancy(q, answer)
            
            results["faithfulness"].append(faith_result["score"])
            results["relevancy"].append(rel_result["score"])

            print(f"   📊 Wierność: {faith_result['score']}/5 → {faith_result['reason']}")
            print(f"   📊 Trafność: {rel_result['score']}/5 → {rel_result['reason']}")

            # Dodatkowa ocena dla pytań poza domeną
            if q_type == "out_of_domain":
                ood_result = evaluate_out_of_domain(q, answer)
                results["out_of_domain"].append(ood_result["score"])
                print(f"   📊 Obsługa poza domeną: {ood_result['score']}/5 → {ood_result['reason']}")

        except Exception as e:
            print(f"   ❌ Błąd oceny: {e}")

        print("-" * 60)

    # Raport końcowy
    print("\n" + "=" * 60)
    print("📈 RAPORT KOŃCOWY")
    print("=" * 60)
    
    avg_faith = sum(results["faithfulness"]) / len(results["faithfulness"]) if results["faithfulness"] else 0
    avg_rel = sum(results["relevancy"]) / len(results["relevancy"]) if results["relevancy"] else 0
    avg_ood = sum(results["out_of_domain"]) / len(results["out_of_domain"]) if results["out_of_domain"] else 0
    
    print(f"✅ Średnia Wierność (Faithfulness):     {avg_faith:.2f} / 5.0")
    print(f"✅ Średnia Trafność (Relevancy):        {avg_rel:.2f} / 5.0")
    if results["out_of_domain"]:
        print(f"✅ Obsługa pytań poza domeną:           {avg_ood:.2f} / 5.0")
    
    # Ogólna ocena
    overall = (avg_faith + avg_rel) / 2
    print(f"\n🏆 OCENA OGÓLNA: {overall:.2f} / 5.0")
    
    if overall >= 4.0:
        print("   Status: DOSKONAŁY 🌟")
    elif overall >= 3.0:
        print("   Status: DOBRY ✓")
    elif overall >= 2.0:
        print("   Status: WYMAGA POPRAWY ⚠️")
    else:
        print("   Status: KRYTYCZNY ❌")

    print("=" * 60)
    
    return {
        "faithfulness": avg_faith,
        "relevancy": avg_rel,
        "out_of_domain": avg_ood,
        "overall": overall,
    }


if __name__ == "__main__":
    run_evaluation()
