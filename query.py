# query.py - FIXED VERSION for LLaMA 3.1 8B
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, math, time
from typing import List, Tuple, Dict, Any, Optional, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

# ---------------------------
# Constantes e defaults
# ---------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

# ----- Retrieval / ranking -----
TOP_K_RETRIEVAL = _env_int("TOP_K_RETRIEVAL", 12)
TOP_K_FINAL     = _env_int("TOP_K_FINAL", 4)

PREFERRED_VERSION_DEFAULT = os.getenv("PREFERRED_VERSION_DEFAULT", "16.0")

# Híbrido (RRF) e diversidade (MMR)
RRF_K       = _env_int("RRF_K", 60)
ENABLE_MMR  = os.getenv("ENABLE_MMR", "1") not in ("0", "false", "False")
MMR_LAMBDA  = _env_float("MMR_LAMBDA", 0.65)
MMR_FINAL_K = _env_int("MMR_FINAL_K", 6)

# Context window management - IMPROVED
MAX_HISTORY_TURNS = _env_int("MAX_HISTORY_TURNS", 10)  # 5 full exchanges
MAX_SNIPPET_CHARS = _env_int("MAX_SNIPPET_CHARS", 1800)  # More context

# Reranker (opcional)
USE_RERANK               = os.getenv("USE_RERANK", "1") not in ("0", "false", "False")
RERANK_TOP_CANDIDATES    = _env_int("RERANK_TOP_CANDIDATES", 20)
RERANK_MODEL_CANDIDATES  = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
]

# ----- Caminhos / modelos -----
EMBEDDINGS_DIM = 1024
INDEX_PATH     = os.getenv("INDEX_PATH", "faiss_index.bin")
TEXTS_PATH     = os.getenv("TEXTS_PATH", "texts.json")
META_PATH      = os.getenv("META_PATH",  "metadatas.json")
DOCS_PATH      = os.getenv("DOCS_PATH",  "docs.json")

EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-m3")

# LLM local (llama.cpp) - CONFIGURED FOR YOUR SYSTEM
from llama_cpp import Llama
# Update this path to where you downloaded the model
LLM_PATH = os.getenv("LLM_PATH", r"C:\Users\USER\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
LLM_CTX  = _env_int("LLM_CTX", 8192)

# =========================
# Compat/Imports
# =========================

try:
    from sentence_transformers import CrossEncoder
    _HAS_CROSS = True
except Exception:
    CrossEncoder = None
    _HAS_CROSS = False

def _device_hint() -> str:
    try:
        import torch
        import torch.cuda
        return "cuda" if getattr(torch.cuda, "is_available", lambda: False)() else "cpu"
    except Exception:
        return "cpu"

# Aliases para compatibilidade
TOP_K = TOP_K_RETRIEVAL
TITLE_BOOST = 0.0

# =========================
# Cache / singletons
# =========================
_LLAMA_SINGLETON: Optional[Llama] = None
_CACHED_RESOURCES: Optional[Dict[str, Any]] = None

# =========================
# NEW: Improved history handling
# =========================

def _coerce_to_str(x: Any) -> str:
    """Extrai texto de diferentes formatos."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        for y in reversed(x):
            if isinstance(y, str):
                return y
            if isinstance(y, dict) and "content" in y:
                return str(y["content"])
    if isinstance(x, dict):
        for k in ("content", "text", "message", "msg", "body"):
            if isinstance(x.get(k), str):
                return x[k]
    return str(x)


def _format_history(history_raw: List[Tuple[str, str]], max_turns: int = 10) -> str:
    """
    FIXED: Format history preserving speaker roles.
    history_raw: List of (role, message) tuples where role is "user" or "assistant"
    """
    if not history_raw:
        return ""
    
    formatted = []
    for role, msg in history_raw[-max_turns:]:
        msg = str(msg).strip() if msg else ""
        if not msg:
            continue
        
        role_str = str(role).lower()
        if role_str in ("user", "usuário", "human"):
            formatted.append(f"Usuário: {msg}")
        elif role_str in ("assistant", "bot", "ai"):
            formatted.append(f"Assistente: {msg}")
        else:
            formatted.append(f"{role}: {msg}")
    
    if not formatted:
        return ""
    
    return "Histórico da conversa:\n" + "\n".join(formatted) + "\n\n"


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for Portuguese"""
    return len(text) // 4


# =========================
# Loaders
# =========================

def get_encoder() -> SentenceTransformer:
    return SentenceTransformer(EMB_MODEL, device=_device_hint())


def _read_json_any(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        pass
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                items.append(json.loads(ln))
            except Exception:
                raise ValueError(f"Arquivo não é JSON válido nem JSONL: {path}")
    return items


def load_index(index_path=INDEX_PATH, meta_path=META_PATH, docs_path=DOCS_PATH):
    """Lê FAISS e carrega metas/textos."""
    index = faiss.read_index(index_path)

    metas: list = []
    texts: list = []
    raw = None

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

    if isinstance(raw, dict):
        metas = raw.get("metas", []) if isinstance(raw.get("metas", []), list) else []
        if "texts" in raw and isinstance(raw["texts"], list):
            texts = raw["texts"]
    elif isinstance(raw, list):
        metas = raw

    if not texts and os.path.exists(docs_path):
        with open(docs_path, "r", encoding="utf-8") as fh:
            maybe_texts = json.load(fh)
        if isinstance(maybe_texts, list):
            texts = maybe_texts

    # Normaliza metas para dicts
    normalized = []
    for m in metas:
        if isinstance(m, dict):
            normalized.append(m)
        elif isinstance(m, str):
            normalized.append({"url": m})
        else:
            normalized.append({})
    metas = normalized

    # Sincroniza tamanhos
    n = index.ntotal
    if len(metas) < n:
        metas.extend([{}] * (n - len(metas)))
    if len(texts) < n:
        texts.extend([""] * (n - len(texts)))

    return index, texts[:n], metas[:n]


def load_llm(path=LLM_PATH, ctx=LLM_CTX):
    """
    OPTIMIZED FOR RTX 4060 8GB with LLaMA 3.1 8B
    """
    global _LLAMA_SINGLETON
    if _LLAMA_SINGLETON is None:
        print(f"🔄 Loading LLaMA 3.1 8B from: {path}")
        
        _LLAMA_SINGLETON = Llama(
            model_path=path,
            n_ctx=ctx,              # 8192 tokens context
            n_batch=512,            # Batch size
            n_threads=8,            # CPU threads (adjust for your Ryzen 7 5700)
            n_gpu_layers=33,        # GPU layers for 8B model on 8GB VRAM
                                    # 33 layers = almost entire model on GPU
                                    # Adjust: 0=CPU only, 43=all layers (may not fit)
            verbose=False,          # Set True to see loading details
        )
        
        print("✅ LLaMA 3.1 8B loaded successfully!")
        
        # Optional: Show GPU info
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        except:
            pass
            
    return _LLAMA_SINGLETON


def load_resources():
    global _CACHED_RESOURCES
    if _CACHED_RESOURCES is not None:
        return _CACHED_RESOURCES

    print("📚 Carregando recursos...")
    enc = get_encoder()
    index, texts, metas = load_index()
    llm = load_llm()

    # BM25 opcional
    bm25 = None
    if texts:
        try:
            tokenized = [t.lower().split() for t in texts]
            bm25 = BM25Okapi(tokenized)
        except Exception:
            pass

    # Reranker opcional
    reranker = None
    if USE_RERANK and _HAS_CROSS:
        for candidate in RERANK_MODEL_CANDIDATES:
            try:
                reranker = CrossEncoder(candidate, max_length=512, device=_device_hint())
                break
            except Exception:
                continue

    _CACHED_RESOURCES = {
        "encoder": enc,
        "index": index,
        "texts": texts,
        "metas": metas,
        "llm": llm,
        "bm25": bm25,
        "reranker": reranker,
    }
    print("✅ Recursos carregados")
    return _CACHED_RESOURCES


# =========================
# Retrieval functions (keep existing - just copying the working version)
# =========================

def retrieve_contexts(
    q: str,
    index,
    encoder,
    metas: List[Dict],
    texts: List[str],
    top_k: int = TOP_K_RETRIEVAL,
    preferred_version: str = PREFERRED_VERSION_DEFAULT,
    bm25=None,
    apply_mmr: bool = True,
) -> List[Tuple[str, Dict, float]]:
    """Retrieval with semantic + BM25 hybrid and optional MMR."""
    
    # Semantic search
    q_vec = encoder.encode([f"query: {q}"], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    D, I = index.search(q_vec, top_k * 2)
    
    sem_results = []
    for dist, idx in zip(D[0], I[0]):
        if 0 <= idx < len(texts):
            sem_results.append((idx, float(dist)))
    
    # BM25
    bm25_results = []
    if bm25 is not None:
        try:
            q_tokens = q.lower().split()
            scores = bm25.get_scores(q_tokens)
            top_bm25 = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
            bm25_results = [(idx, score) for idx, score in top_bm25 if score > 0]
        except Exception:
            pass
    
    # RRF fusion
    def rrf_score(rank: int, k: int = RRF_K) -> float:
        return 1.0 / (k + rank)
    
    combined = {}
    for rank, (idx, _) in enumerate(sem_results):
        combined[idx] = combined.get(idx, 0.0) + rrf_score(rank)
    
    for rank, (idx, _) in enumerate(bm25_results):
        combined[idx] = combined.get(idx, 0.0) + rrf_score(rank) * 0.5
    
    # Version boost
    for idx in combined:
        meta = metas[idx] if idx < len(metas) else {}
        url = meta.get("url", "")
        if preferred_version in url:
            combined[idx] *= 1.15
    
    # Sort by combined score
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # MMR diversification
    if apply_mmr and ENABLE_MMR:
        selected_idx = []
        candidate_idx = [idx for idx, _ in ranked]
        
        if candidate_idx:
            cand_vecs = []
            for idx in candidate_idx:
                try:
                    vec = index.reconstruct(int(idx))
                    cand_vecs.append(vec)
                except Exception:
                    cand_vecs.append(np.zeros(EMBEDDINGS_DIM, dtype="float32"))
            cand_vecs = np.array(cand_vecs)
            
            selected_idx.append(candidate_idx[0])
            
            while len(selected_idx) < min(MMR_FINAL_K, len(candidate_idx)):
                best_score = -float('inf')
                best_idx_pos = -1
                
                for i, idx in enumerate(candidate_idx):
                    if idx in selected_idx:
                        continue
                    
                    rel = combined[idx]
                    max_sim = 0.0
                    for sel_idx in selected_idx:
                        sel_pos = candidate_idx.index(sel_idx)
                        sim = np.dot(cand_vecs[i], cand_vecs[sel_pos])
                        max_sim = max(max_sim, sim)
                    
                    mmr = MMR_LAMBDA * rel - (1 - MMR_LAMBDA) * max_sim
                    
                    if mmr > best_score:
                        best_score = mmr
                        best_idx_pos = i
                
                if best_idx_pos >= 0:
                    selected_idx.append(candidate_idx[best_idx_pos])
                else:
                    break
            
            ranked = [(idx, combined[idx]) for idx in selected_idx]
    
    # Format results
    results = []
    for idx, score in ranked:
        if 0 <= idx < len(texts):
            txt = texts[idx]
            meta = metas[idx] if idx < len(metas) else {}
            results.append((txt, meta, score))
    
    return results


# =========================
# Query processing
# =========================

def needs_clarification_and_fuse(question: str, history: List[Tuple[str, str]]) -> Tuple[str, bool]:
    """Check if query needs clarification."""
    q = question.strip()
    
    if len(q) < 15 and not history:
        return q, True
    
    if any(term in q.lower() for term in ["isso", "isto", "aquilo", "esse", "esta", "nisso"]):
        if not history:
            return q, True
    
    return q, False


# =========================
# FIXED: Unified generate_llama function
# =========================

def generate_llama(
    llm: Llama,
    question: str,
    contexts: List[Tuple[str, Dict, float]],
    history_raw: List[Tuple[str, str]],
    max_tokens: int = 384,
) -> str:
    """
    FIXED: Unified generation with chat template priority.
    Optimized for LLaMA 3.1 8B Instruct.
    """
    # Try chat completion first (BEST for Instruct models)
    if hasattr(llm, "create_chat_completion"):
        try:
            messages = build_messages(question, contexts, history_raw)
            out = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                top_p=0.9,
                repeat_penalty=1.15,
            )
            return out["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"⚠️ Chat completion failed: {e}, falling back")
    
    # Fallback to completion
    prompt = build_prompt(question, contexts, history_raw, tok=None)
    defaults = dict(
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.15,
        stop=[
            "\nPergunta:", "Pergunta:",
            "\nQuestion:", "Question:",
            "\nContexto:", "Contexto:",
            "\nFonte:", "Fontes:",
            "<|im_end|>", "</s>", "<|eot_id|>",  # LLaMA 3.1 stop tokens
        ],
    )
    
    out = llm(prompt, max_tokens=max_tokens, **defaults)
    text = out["choices"][0]["text"].strip()
    
    # Sanitize artifacts
    bad_markers = ("Status da ivy", "DataSource", "interpreted context", "dropdown menu")
    if any(marker.lower() in text.lower() for marker in bad_markers):
        lines = []
        for line in text.splitlines():
            if any(marker.lower() in line.lower() for marker in bad_markers):
                break
            lines.append(line)
        text = "\n".join(lines).strip()
    
    return text


# =========================
# IMPROVED: Prompt building
# =========================

def build_prompt(
    question: str,
    contexts: List[Tuple[str, Dict, float]],
    history_raw: List[Tuple[str, str]],
    tok=None,
) -> str:
    """
    IMPROVED: Better prompt with strict instructions and proper history.
    """
    if not contexts:
        return (
            "Você é um assistente especializado em Odoo. Responda em PT-BR.\n"
            "Não há informações relevantes disponíveis para responder à pergunta.\n\n"
            "Responda educadamente:\n"
            "1. Informe que não encontrou a informação\n"
            "2. Peça: módulo, versão e objetivo\n"
            "3. Seja breve (máximo 3 linhas)\n\n"
            f"Pergunta: {question.strip()}\n"
            "Resposta:"
        )
    
    # Format contexts
    ctx_lines = []
    for i, (txt, meta, score) in enumerate(contexts, start=1):
        src = (meta or {}).get("url") or "desconhecido"
        title = (meta or {}).get("title") or ""
        
        snippet = txt if isinstance(txt, str) else (txt.get("text") or txt.get("content") or "")
        snippet = snippet.strip()
        
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[:MAX_SNIPPET_CHARS] + "..."
        
        header = f"[{i}]" + (f" {title}" if title else "")
        ctx_lines.append(f"{header}\nFonte: {src}\n{snippet}")
    
    ctx_block = "\n\n".join(ctx_lines)
    
    # Format history with proper roles
    hist_block = _format_history(history_raw, max_turns=MAX_HISTORY_TURNS)
    
    system = """Você é um assistente especializado em Odoo. Responda sempre em português brasileiro (PT-BR).

**REGRAS CRÍTICAS** (seguir estritamente):

1. **Use APENAS as informações do Contexto abaixo**
   - Não invente passos, menus, botões ou funcionalidades
   - Se algo não estiver no Contexto, diga claramente

2. **SEMPRE cite as fontes**
   - Use [1], [2], [3] ao mencionar informações
   - Exemplo: "Configure o CFOP no campo correspondente [1]"

3. **Seja preciso e estruturado**
   - Para processos: liste os passos numerados
   - Para configurações: mencione os campos exatos
   - Para dúvidas: admita incerteza explicitamente

4. **Mantenha coerência com o histórico**
   - Se o usuário já forneceu contexto, use-o
   - Não peça informações já fornecidas

5. **Quando a informação é incompleta**
   - Não invente: "Segundo o Contexto [1], ..."
   - Sugira: "Para mais informações sobre X, consulte a documentação oficial"

**NUNCA**:
- Invente passos não documentados
- Afirme recursos sem confirmação no Contexto
- Ignore perguntas já respondidas"""

    prompt = (
        system + "\n\n"
        + "=" * 60 + "\n"
        + "CONTEXTO:\n"
        + "=" * 60 + "\n"
        + ctx_block + "\n\n"
        + ("=" * 60 + "\n" + hist_block if hist_block else "")
        + "=" * 60 + "\n"
        + f"PERGUNTA ATUAL:\n{question.strip()}\n\n"
        + "RESPOSTA (cite as fontes [N]):\n"
    )
    
    return prompt


def build_messages(
    question: str,
    contexts: List[Tuple[str, Dict, float]],
    history_raw: List[Tuple[str, str]],
) -> List[Dict[str, str]]:
    """
    IMPROVED: Chat template version for LLaMA 3.1 8B.
    """
    if not contexts:
        sys = (
            "Você é um assistente do Odoo. Responda em PT-BR, e NUNCA invente.\n"
            "Não há contexto disponível. Peça módulo e versão do Odoo."
        )
        usr = f"Pergunta: {question.strip()}"
        return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
    
    # Build context
    lines = []
    for i, (txt, meta, _s) in enumerate(contexts, start=1):
        src = (meta or {}).get("url") or "desconhecido"
        title = (meta or {}).get("title") or ""
        
        snippet = txt if isinstance(txt, str) else (txt.get("text") or txt.get("content") or "")
        snippet = snippet.strip()
        
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[:MAX_SNIPPET_CHARS] + "..."
        
        header = f"[{i}]" + (f" {title}" if title else "")
        lines.append(f"{header}\nFonte: {src}\n{snippet}")
    
    ctx_block = "\n\n".join(lines)
    
    # History
    hist_block = _format_history(history_raw, max_turns=MAX_HISTORY_TURNS)
    
    sys = (
        "Você é um assistente especializado em Odoo. Responda em PT-BR. "
        "Use SOMENTE as informações do Contexto. SEMPRE cite as fontes [N]. "
        "Não invente passos ou funcionalidades."
    )
    
    usr = (
        f"Contexto:\n{ctx_block}\n\n"
        + (f"{hist_block}" if hist_block else "")
        + f"Pergunta: {question.strip()}\n\n"
        + "Responda de forma precisa e cite as fontes:"
    )
    
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


# =========================
# Main orchestration
# =========================

def answer_with_history(
    question: str,
    history: List[Tuple[str, str]],
    resources: Dict[str, Any],
    final_k: int = TOP_K_FINAL,
    preferred_version: str = PREFERRED_VERSION_DEFAULT,
    return_contexts: bool = False,  # NEW: for metrics
):
    """
    IMPROVED: Main answer function with option to return contexts for evaluation.
    """
    if resources is None:
        resources = load_resources()
    
    enc = resources["encoder"]
    index = resources["index"]
    texts = resources["texts"]
    metas = resources["metas"]
    llm = resources["llm"]
    
    # Check if clarification needed
    fused_q, must_ask = needs_clarification_and_fuse(question, history)
    if must_ask:
        ask = (
            "Para te ajudar **no Odoo**, preciso de **contexto do módulo**.\n\n"
            "**Em qual área isso se encaixa?** (ex.: *Vendas*, *Inventário*, *Faturamento*)\n"
            "E, se puder, **resuma em 1 frase o objetivo**."
        )
        if return_contexts:
            return ask, [], []
        return ask, []
    
    # Retrieve contexts
    ctxs = retrieve_contexts(
        q=question,
        index=index,
        encoder=enc,
        metas=metas,
        texts=texts,
        top_k=TOP_K_RETRIEVAL,
        preferred_version=preferred_version,
        bm25=resources.get("bm25"),
        apply_mmr=True,
    )
    
    # Generate answer
    if not ctxs:
        answer = generate_llama(llm, question, ctxs, history, max_tokens=220)
        if return_contexts:
            return answer, [], []
        return answer, []
    
    answer = generate_llama(llm, question, ctxs, history, max_tokens=384)
    
    # Extract references
    refs: List[str] = []
    for _txt, meta, _score in ctxs[:final_k]:
        if isinstance(meta, dict):
            url = meta.get("url")
            if url:
                refs.append(url)
    
    if return_contexts:
        return answer, refs, ctxs
    return answer, refs
