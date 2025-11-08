# app_chat_fixed.py — UI do Assistente ERP (Llama + FAISS) with real-time evals
from __future__ import annotations

import os
import time
import uuid
import streamlit as st

from query import load_resources, answer_with_history
from realtime_evals import ConversationEvaluator

# ===== STATE GLOBAL =====
if "run_uid" not in st.session_state:
    st.session_state.run_uid = uuid.uuid4().hex[:8]
if "history" not in st.session_state:
    st.session_state.history = []  # List of ("user"|"assistant", str) tuples
if "evaluator" not in st.session_state:
    st.session_state.evaluator = ConversationEvaluator(window_size=10)
if "cheap_mode" not in st.session_state:
    st.session_state.cheap_mode = False
if "preferred_version" not in st.session_state:
    st.session_state.preferred_version = "16.0"
if "show_metrics" not in st.session_state:
    st.session_state.show_metrics = True

# ===== Carga de recursos (cache) =====
@st.cache_resource(show_spinner="Carregando modelos e índice…")
def _bootstrap():
    return load_resources()

resources = _bootstrap()

# Configure cheap mode
if st.session_state.cheap_mode:
    os.environ["USE_RERANK"] = "0"
    os.environ["TOP_K_RETRIEVAL"] = "8"
else:
    os.environ["USE_RERANK"] = "1"

# ===== Layout =====
st.set_page_config(
    page_title="Assistente ERP (LLaMa 3 + FAISS)", 
    page_icon="💬", 
    layout="wide"
)

# Create columns: main chat and metrics
col_main, col_metrics = st.columns([3, 1])

with col_main:
    st.title("Assistente ERP (Odoo) — LLaMA + RAG")
    st.caption(f"Execução: {st.session_state.run_uid}")

with col_metrics:
    st.session_state.show_metrics = st.toggle(
        "📊 Mostrar métricas",
        value=True,
        help="Exibe métricas em tempo real da conversa"
    )

# ===== Sidebar =====
with st.sidebar:
    st.header("Opções")
    st.session_state.preferred_version = st.selectbox(
        "Versão preferida dos docs",
        options=["16.0", "17.0", "15.0"],
        index=0,
        help="Aplica boost em URLs com a versão selecionada"
    )
    st.session_state.cheap_mode = st.toggle(
        "Modo barato",
        value=False,
        help="Desliga reranker e reduz top-k"
    )
    
    if st.button("🗑️ Limpar conversa"):
        st.session_state.history = []
        st.session_state.evaluator = ConversationEvaluator(window_size=10)
        st.rerun()
    
    st.divider()
    
    # Show running statistics
    if st.session_state.show_metrics and st.session_state.evaluator:
        st.markdown(st.session_state.evaluator.format_running_stats_display())

# ===== Display conversation history =====
with col_main:
    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

# ===== Chat input =====
with col_main:
    user_msg = st.chat_input("Pergunte algo sobre o Odoo...")

if user_msg:
    # Display user message
    with col_main:
        with st.chat_message("user"):
            st.markdown(user_msg)
    
    # Add to history
    st.session_state.history.append(("user", user_msg))
    
    # Generate response
    with col_main:
        with st.chat_message("assistant"):
            with st.spinner("Buscando informações..."):
                t0 = time.time()
                
                # Call with return_contexts=True for metrics
                answer, refs, contexts = answer_with_history(
                    question=user_msg,
                    history=st.session_state.history[:-1],  # Exclude current question
                    resources=resources,
                    preferred_version=st.session_state.preferred_version,
                    return_contexts=True,
                )
                
                t1 = time.time()
                total_time = t1 - t0
                
                # Estimate retrieval vs generation time (approximate)
                retrieval_time = total_time * 0.65
                generation_time = total_time * 0.35
            
            # Display answer
            st.markdown(answer)
            
            # Show sources
            if refs:
                with st.expander("📚 Ver fontes usadas"):
                    for i, url in enumerate(refs, 1):
                        st.write(f"[{i}] {url}")
            
            # Evaluate and show metrics
            if st.session_state.show_metrics and contexts is not None:
                metrics = st.session_state.evaluator.evaluate_turn(
                    question=user_msg,
                    answer=answer if isinstance(answer, str) else str(answer),
                    contexts=contexts if isinstance(contexts, list) else [],
                    retrieval_time=retrieval_time,
                    generation_time=generation_time,
                )
                
                with st.expander("📊 Métricas desta resposta"):
                    st.markdown(st.session_state.evaluator.format_metrics_display(metrics))
                    
                    # Show warnings if any
                    if metrics.get('possibly_invented'):
                        st.warning("⚠️ Esta resposta pode conter informações não fundamentadas nas fontes.")
                    if metrics['num_contexts'] == 0:
                        st.info("ℹ️ Nenhum contexto relevante foi encontrado.")
            
            st.caption(f"⏱️ {total_time:.1f}s")
    
    # Add assistant response to history
    st.session_state.history.append(("assistant", answer if isinstance(answer, str) else str(answer)))
    
    # Update metrics display in sidebar
    if st.session_state.show_metrics:
        with col_metrics:
            # Force refresh of the running stats
            st.markdown(st.session_state.evaluator.format_running_stats_display())
