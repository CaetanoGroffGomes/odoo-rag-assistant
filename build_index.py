# build_index.py — FAISS + BGE-M3 + chunks menores/overlap maior
import json
from pathlib import Path
import math, os, random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DOCS_PATH = "docs.json"
INDEX_PATH = "faiss_index.bin"
EMB_PATH = "embeddings.npy"   # opcional
META_PATH = "metadatas.json"

# Chunking:
CHUNK_WORDS = 140
OVERLAP_WORDS = 40

# Embedding:
EMB_MODEL = "BAAI/bge-m3"  # 1024d
DEVICE = "cuda"            # "cuda" ou "cpu"
BATCH = 32
TRAIN_SIZE = 8000
PQ_M = 32
PQ_B = 8

def load_docs(path=DOCS_PATH):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} não encontrado. Rode scrape_docs.py antes.")
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    docs = []
    for d in raw:
        content = d.get("content") or d.get("text") or ""
        url = d.get("url") or d.get("meta", {}).get("url") or ""
        title = d.get("title") or ""
        if content.strip():
            docs.append({"content": content, "url": url, "title": title})
    return docs

def chunk(text: str, max_words=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    step = max(1, max_words - overlap)
    for i in range(0, len(words), step):
        yield " ".join(words[i:i+max_words])

def encode_stream(chunks, model, batch=BATCH):
    for i in range(0, len(chunks), batch):
        batch_text = [f"passage: {c}" for c in chunks[i:i+batch]]
        embs = model.encode(
            batch_text, batch_size=batch, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        yield embs

def build_faiss_ivfpq(embs_memmap_path, n_items, dim, train_size=TRAIN_SIZE):
    n_train = min(n_items, train_size)
    sample_idx = np.random.default_rng(7).choice(n_items, size=n_train, replace=False)
    Xtrain = np.memmap(embs_memmap_path, dtype="float32", mode="r", shape=(n_items, dim))[sample_idx]

    nlist = max(1, int(math.sqrt(n_items)))
    quant = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index = faiss.IndexIVFPQ(quant, dim, nlist, PQ_M, PQ_B, faiss.METRIC_INNER_PRODUCT)

    print(f"🔧 Treinando IVF-PQ | nlist={nlist}  m={PQ_M}  b={PQ_B}  treino={len(Xtrain)}")
    index.train(np.asarray(Xtrain, dtype="float32"))

    mmap = np.memmap(embs_memmap_path, dtype="float32", mode="r", shape=(n_items, dim))
    cursor = 0
    STEP = 10000
    while cursor < n_items:
        part = np.asarray(mmap[cursor:cursor+STEP])
        index.add(part)
        cursor += STEP

    index.nprobe = min(32, max(8, nlist // 20))
    return index

def main():
    docs = load_docs(DOCS_PATH)
    print(f"📂 {len(docs)} documentos")

    chunks, metas = [], []
    for d in docs:
        for ch in chunk(d["content"], max_words=CHUNK_WORDS, overlap=OVERLAP_WORDS):
            chunks.append(ch); metas.append({"url": d["url"], "title": d["title"]})
    N = len(chunks)
    print(f"🔪 {N} chunks")

    try:
        model = SentenceTransformer(EMB_MODEL, device=DEVICE)
    except Exception:
        print("⚠️ CUDA indisponível — usando CPU")
        model = SentenceTransformer(EMB_MODEL)

    DIM = 1024
    mem_path = "embeddings.memmap"
    if os.path.exists(mem_path):
        os.remove(mem_path)
    mmap = np.memmap(mem_path, dtype="float32", mode="w+", shape=(N, DIM))

    cursor = 0
    for batch_emb in encode_stream(chunks, model, batch=BATCH):
        bsz = batch_emb.shape[0]
        mmap[cursor:cursor+bsz] = batch_emb
        cursor += bsz
    del mmap

    if N < 20000:
        print(f"⚙️  Usando FAISS HNSW Flat (N={N})")
        M = 32
        index = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 96

        mmap_r = np.memmap(mem_path, dtype="float32", mode="r", shape=(N, DIM))
        STEP = 10000
        cursor = 0
        while cursor < N:
            part = np.asarray(mmap_r[cursor:cursor + STEP])
            index.add(part)
            cursor += STEP
    else:
        print(f"⚙️  Usando FAISS IVF-PQ (N={N})")
        index = build_faiss_ivfpq(mem_path, N, DIM, train_size=TRAIN_SIZE)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"texts": chunks, "metas": metas}, f, ensure_ascii=False)

    try:
        _ = index.nprobe
        print(f"✅ Index salvo em {INDEX_PATH} ({index.ntotal} vetores) | nprobe={index.nprobe}")
    except AttributeError:
        ef = getattr(index.hnsw, "efSearch", None)
        if ef is not None:
            print(f"✅ Index salvo em {INDEX_PATH} ({index.ntotal} vetores) | efSearch={ef}")
        else:
            print(f"✅ Index salvo em {INDEX_PATH} ({index.ntotal} vetores)")

if __name__ == "__main__":
    main()
