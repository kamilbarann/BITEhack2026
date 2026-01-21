import json
import os
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.brain import get_rag_chain
from src.utils import get_config

# 1. Konfiguracja Sędziego (Groq)
config = get_config()

# Klucz API (z env lub configu)
api_key = os.getenv("GROQ_API_KEY") or config.get("groq_api_key")

judge_llm = ChatGroq(
    model_name=config["judge_model"],
    temperature=0,
    api_key=api_key,
)


def extract_json_from_text(text):
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: szukanie ręczne
    score_match = re.search(r'"score"\s*:\s*([01])', text)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)

    if score_match and reason_match:
        return {"score": int(score_match.group(1)), "reason": reason_match.group(1)}

    return {"score": 0, "reason": "Błąd parsowania odpowiedzi sędziego"}


def evaluate_faithfulness(answer, context_text):
    """
    Sprawdza Wierność: Czy odpowiedź wynika z kontekstu?
    """
    prompt = ChatPromptTemplate.from_template("""
    Jesteś surowym sędzią AI. Oceniasz "Wierność" (Faithfulness).

    KONTEKST:
    {context}

    ODPOWIEDŹ SYSTEMU:
    {answer}

    Zadanie: Czy odpowiedź wynika TYLKO z kontekstu? 
    1 = TAK (Wszystko jest w tekście)
    0 = NIE (Model zmyśla lub używa wiedzy zewnętrznej)

    Odpowiedz WYŁĄCZNIE w JSON: {{"score": <0 lub 1>, "reason": "<krótki powód>"}}
    """)

    chain = prompt | judge_llm | StrOutputParser()
    raw_response = chain.invoke({"answer": answer, "context": context_text})
    return extract_json_from_text(raw_response)


def evaluate_relevancy(question, answer):
    """
    Sprawdza Trafność: Czy odpowiedź jest na temat?
    """
    prompt = ChatPromptTemplate.from_template("""
    Jesteś surowym sędzią AI. Oceniasz "Trafność" (Relevancy).

    PYTANIE: {question}
    ODPOWIEDŹ: {answer}

    Zadanie: Czy to jest odpowiedź na zadane pytanie?
    1 = TAK
    0 = NIE

    Odpowiedz WYŁĄCZNIE w JSON: {{"score": <0 lub 1>, "reason": "<krótki powód>"}}
    """)

    chain = prompt | judge_llm | StrOutputParser()
    raw_response = chain.invoke({"question": question, "answer": answer})
    return extract_json_from_text(raw_response)


def run_evaluation():
    print("\nSTART EWALUACJI (Sędzia: Groq/Llama3)")
    print("-" * 50)

    # Zestaw pytań testowych
    test_questions = [
        "Czym jest obiekt kosmiczny w świetle prawa?",
        "Kto odpowiada za szkody wyrządzone przez satelitę na Ziemi?",
        "Czy Księżyc może należeć do prywatnej firmy?",
        "Jaki jest przepis na ciasto marchewkowe?",  # Test negatywny
    ]

    rag_chain = get_rag_chain()

    total_faithfulness = 0
    total_relevancy = 0

    for q in test_questions:
        print(f"Pytanie: {q}")

        try:
            # 2. Uruchomienie Brain
            response = rag_chain.invoke({"question": q})

            answer = response["answer"]

            context_text = response["context"]

            # Zabezpieczenie na wypadek pustego kontekstu
            if not context_text:
                context_text = "Brak kontekstu (pusty string)."

            # 3. Ocena Sędziego
            faith_result = evaluate_faithfulness(answer, context_text)
            rel_result = evaluate_relevancy(q, answer)

            print(f"   Odpowiedź: {answer}...")
            print(
                f"   Wierność: {faith_result['score']} -> {faith_result['reason']}"
            )
            print(f"   Trafność: {rel_result['score']} -> {rel_result['reason']}")

            total_faithfulness += faith_result["score"]
            total_relevancy += rel_result["score"]

        except Exception as e:
            print(f"   Błąd oceny: {e}")

        print("-" * 50)

    # 4. Raport
    avg_faith = total_faithfulness / len(test_questions) if test_questions else 0
    avg_rel = total_relevancy / len(test_questions) if test_questions else 0

    print("\nRAPORT KOŃCOWY:")
    print(f"Średnia Wierność: {avg_faith:.2f} / 1.0")
    print(f"Średnia Trafność: {avg_rel:.2f} / 1.0")


if __name__ == "__main__":
    run_evaluation()
