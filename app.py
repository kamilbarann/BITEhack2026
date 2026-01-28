import streamlit as st
from src.brain import get_astro_answer, get_rag_chain

st.set_page_config(
    page_title="OrbitGuide",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# CSS
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    :root {
        --bg-primary: #0f1419;
        --bg-secondary: #192734;
        --bg-card: #22303c;
        --border: #38444d;
        --text: #ffffff;
        --text-muted: #8899a6;
        --accent: #1d9bf0;
        --success: #00ba7c;
    }
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp { background: linear-gradient(180deg, #0f1419 0%, #15202b 100%); }
    [data-testid="stAppViewContainer"] { background: transparent; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { display: none; }
    
    .header {
        background: linear-gradient(135deg, #1d9bf0 0%, #00ba7c 100%);
        padding: 28px 32px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .header h1 { color: white; font-size: 1.8em; font-weight: 600; margin: 0 0 4px 0; }
    .header p { color: rgba(255,255,255,0.85); font-size: 0.95em; margin: 0; }
    
    .status {
        background: rgba(0,186,124,0.12);
        color: #00ba7c;
        padding: 8px 14px;
        border-radius: 16px;
        font-size: 0.8em;
        display: inline-block;
        margin-top: 12px;
    }
    
    .msg-user {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        border-left: 3px solid var(--accent);
    }
    .msg-user-label { color: var(--accent); font-size: 0.75em; font-weight: 600; margin-bottom: 6px; text-transform: uppercase; }
    .msg-user-text { color: var(--text); font-size: 0.95em; line-height: 1.5; }
    
    .msg-bot {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        border-left: 3px solid var(--success);
    }
    .msg-bot-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .msg-bot-label { color: var(--success); font-size: 0.75em; font-weight: 600; text-transform: uppercase; }
    .msg-bot-text { color: var(--text); font-size: 0.95em; line-height: 1.6; }
    
    .conf { font-size: 0.7em; padding: 3px 8px; border-radius: 10px; }
    .conf-high { background: rgba(0,186,124,0.15); color: #00ba7c; }
    .conf-med { background: rgba(255,173,31,0.15); color: #ffad1f; }
    .conf-low { background: rgba(244,33,46,0.15); color: #f4212e; }
    
    .source {
        background: var(--bg-primary);
        padding: 10px 14px;
        border-radius: 6px;
        margin: 6px 0;
        border: 1px solid var(--border);
        display: flex;
        justify-content: space-between;
    }
    .source-name { color: var(--text); font-size: 0.85em; }
    .source-pages { color: var(--text-muted); font-size: 0.75em; }
    
    .empty { text-align: center; padding: 40px; color: var(--text-muted); }
    
    [data-testid="stChatInput"] > div { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 10px; }
    [data-testid="stChatInput"] input { color: var(--text) !important; }
    .streamlit-expanderHeader { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="header">
        <h1>OrbitGuide</h1>
        <p>Asystent ds. prawa kosmicznego</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Init
try:
    with st.spinner("Inicjalizacja..."):
        get_rag_chain()
    st.markdown('<div class="status">System gotowy</div>', unsafe_allow_html=True)
except Exception as e:
    st.error(f"Blad: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# History
if st.session_state.messages:
    for m in st.session_state.messages:
        if m["role"] == "user":
            st.markdown(f'<div class="msg-user"><div class="msg-user-label">Pytanie</div><div class="msg-user-text">{m["content"]}</div></div>', unsafe_allow_html=True)
        else:
            conf = m.get("confidence", 0)
            is_greet = m.get("is_greeting", False)
            if is_greet or conf == 0:
                st.markdown(f'<div class="msg-bot"><div class="msg-bot-label">OrbitGuide</div><div class="msg-bot-text">{m["content"]}</div></div>', unsafe_allow_html=True)
            else:
                if conf > 70:
                    conf_class, conf_label = "conf-high", "Wysoka"
                elif conf > 40:
                    conf_class, conf_label = "conf-med", "Srednia"
                else:
                    conf_class, conf_label = "conf-low", "Niska"
                st.markdown(f'''<div class="msg-bot">
                    <div class="msg-bot-header">
                        <span class="msg-bot-label">OrbitGuide</span>
                        <span class="conf {conf_class}">Pewnosc: {conf_label}</span>
                    </div>
                    <div class="msg-bot-text">{m["content"]}</div>
                </div>''', unsafe_allow_html=True)
else:
    st.markdown('<div class="empty">Zadaj pytanie dotyczace prawa kosmicznego</div>', unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Wpisz pytanie..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="msg-user"><div class="msg-user-label">Pytanie</div><div class="msg-user-text">{prompt}</div></div>', unsafe_allow_html=True)

    with st.spinner("Analizuje..."):
        try:
            result = get_astro_answer(prompt, st.session_state.messages)
            answer = result["answer"].replace("\\n", "\n")
            confidence = result.get("confidence", 0)
            is_greeting = result.get("is_greeting", False)

            if confidence > 70:
                conf_class, conf_label = "conf-high", "Wysoka"
            elif confidence > 40:
                conf_class, conf_label = "conf-med", "Srednia"
            else:
                conf_class, conf_label = "conf-low", "Niska"

            if is_greeting:
                st.markdown(f'<div class="msg-bot"><div class="msg-bot-label">OrbitGuide</div><div class="msg-bot-text">{answer}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'''<div class="msg-bot">
                        <div class="msg-bot-header">
                            <span class="msg-bot-label">OrbitGuide</span>
                            <span class="conf {conf_class}">Pewnosc: {conf_label}</span>
                        </div>
                        <div class="msg-bot-text">{answer}</div>
                    </div>''',
                    unsafe_allow_html=True,
                )

            # Sources
            if not is_greeting and result["sources"]:
                with st.expander("Zrodla", expanded=False):
                    grouped = {}
                    for src in result["sources"]:
                        txt = src.get("text", "") if isinstance(src, dict) else str(src)
                        if "📄" in txt:
                            parts = txt.split("(str.")
                            name = parts[0].replace("📄", "").strip()
                            page = parts[1].replace(")", "").strip() if len(parts) > 1 else "?"
                        else:
                            name, page = txt, "?"
                        name = name[:40] + "..." if len(name) > 40 else name
                        if name not in grouped:
                            grouped[name] = []
                        grouped[name].append(page)
                    
                    for doc, pages in grouped.items():
                        pages_str = ", ".join(sorted(set(pages), key=lambda x: int(x) if x.isdigit() else 0))
                        st.markdown(f'<div class="source"><span class="source-name">{doc}</span><span class="source-pages">str. {pages_str}</span></div>', unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "confidence": confidence,
                "is_greeting": is_greeting
            })

        except Exception as e:
            st.error(f"Blad: {e}")
