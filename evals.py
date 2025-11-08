# evals.py — Evals offline (retrieval + geração)
from __future__ import annotations
import json, math, time, argparse, re
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

from query import load_resources, answer_with_history  # usa seu pipeline

NEGA_PHRASES = [
    "não sei a partir do contexto fornecido",
    "não sei com base no contexto fornecido",
    "não sei com o contexto fornecido",
]

def _normalize_text(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-zà-ú0-9çãõâêîôûáéíóúäëïöü\- ]", " ", s)
    return s.split()

def f1_score(pred: str, golds: List[str]) -> float:
    p_toks = _normalize_text(pred)
    if not p_toks:
        return 0.0
    best = 0.0
    for g in golds:
        g_toks = _normalize_text(g)
        if not g_toks:
            continue
        common = defaultdict(int)
        for t in p_toks:
            common[t] += 1
        inter = 0
        g_cnt = defaultdict(int)
        for t in g_toks:
            g_cnt[t] += 1
        for t, c in g_cnt.items():
            inter += min(c, common.get(t, 0))
        if inter == 0:
            continue
        prec = inter / len(p_toks)
        rec  = inter / len(g_toks)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        best = max(best, f1)
    return best

def contains_negation(pred: str) -> bool:
    x = pred.lower()
    return any(phrase in x for phrase in NEGA_PHRASES)

def retrieval_metrics(hits: List[List[str]], golds: List[List[str]], k: int = 5) -> Dict[str, float]:
    n = len(hits)
    hitk = 0
    mrr  = 0.0
    for i in range(n):
        topk = hits[i][:k]
        gold = set(golds[i])
        if gold & set(topk):
            hitk += 1
        rank = math.inf
        for r, h in enumerate(topk, 1):
            if h in gold:
                rank = r; break
        mrr += 0.0 if rank == math.inf else 1.0 / rank
    return {f"hit@{k}": hitk / max(n,1), f"mrr@{k}": mrr / max(n,1)}

def run_evals(eval_path: str, topk: int = 5) -> Dict[str, float]:
    resources = load_resources()
    data = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    y_pred_texts, y_gold_texts = [], []
    y_hits, y_gold_sources, times = [], [], []

    for ex in data:
        q = ex["question"]
        gold_ans = ex.get("answers", [])
        gold_src = ex.get("gold_sources", [])
        t0 = time.time()
        ans, refs = answer_with_history(q, history=[], resources=resources)
        times.append(time.time() - t0)

        urls = []
        if isinstance(refs, list):
            for r in refs:
                if isinstance(r, dict):
                    u = r.get("url") or r.get("link") or r.get("href")
                    if u: urls.append(u)
                elif isinstance(r, (list, tuple)) and r and isinstance(r[0], str):
                    urls.append(r[0])
                elif isinstance(r, str):
                    urls.append(r)
        y_pred_texts.append(ans or "")
        y_gold_texts.append(gold_ans or [])
        y_hits.append(urls)
        y_gold_sources.append(gold_src or [])

    f1s = [f1_score(p, g) for p, g in zip(y_pred_texts, y_gold_texts)]
    f1_avg = sum(f1s) / max(len(f1s), 1)
    abst = sum(1 for p in y_pred_texts if contains_negation(p)) / max(len(y_pred_texts), 1)

    halluc = 0
    for p, gold_srcs, hits in zip(y_pred_texts, y_gold_sources, y_hits):
        if not contains_negation(p):
            if not (set(gold_srcs) & set(hits[:topk])):
                halluc += 1
    halluc_rate = halluc / max(len(y_pred_texts), 1)

    rmetrics = retrieval_metrics(y_hits, y_gold_sources, k=topk)
    lat_avg = sum(times) / max(len(times), 1)

    results = {
        "f1_avg": f1_avg,
        "abstention_rate": abst,
        "halluc_rate": halluc_rate,
        "latency_avg_sec": lat_avg,
        **rmetrics
    }
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval_set.jsonl")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()
    out = run_evals(args.eval, topk=args.topk)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
