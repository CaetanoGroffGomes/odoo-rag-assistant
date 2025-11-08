# realtime_evals.py - Real-time conversation evaluation
from typing import Dict, List, Tuple, Any
import re
from collections import deque
import time

class ConversationEvaluator:
    """
    Tracks metrics in real-time during conversations.
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.conversation_start = time.time()
        
    def evaluate_turn(
        self,
        question: str,
        answer: str,
        contexts: List[Tuple[str, Dict, float]],
        retrieval_time: float,
        generation_time: float,
    ) -> Dict[str, Any]:
        """
        Evaluate a single conversation turn.
        Returns metrics dict and updates running statistics.
        """
        metrics = {}
        
        # 1. Retrieval quality
        metrics["num_contexts"] = len(contexts)
        if contexts:
            metrics["avg_retrieval_score"] = sum(score for _, _, score in contexts) / len(contexts)
            metrics["min_retrieval_score"] = min(score for _, _, score in contexts)
        else:
            metrics["avg_retrieval_score"] = 0.0
            metrics["min_retrieval_score"] = 0.0
        
        # 2. Answer quality indicators
        metrics["answer_length"] = len(answer)
        metrics["has_sources"] = bool(re.search(r"\[(\d+)\]", answer))
        metrics["source_citations"] = len(re.findall(r"\[(\d+)\]", answer))
        
        # 3. Hallucination indicators
        uncertainty_phrases = [
            "não sei", "não tenho certeza", "não encontrei",
            "não há informação", "segundo o contexto",
        ]
        metrics["expresses_uncertainty"] = any(phrase in answer.lower() for phrase in uncertainty_phrases)
        
        # Check for invented content (heuristic)
        inventive_phrases = [
            "basta clicar", "simplesmente", "é só",  # Overconfident without sources
        ]
        metrics["possibly_invented"] = (
            any(phrase in answer.lower() for phrase in inventive_phrases)
            and metrics["source_citations"] == 0
        )
        
        # 4. Performance
        metrics["retrieval_time_sec"] = retrieval_time
        metrics["generation_time_sec"] = generation_time
        metrics["total_time_sec"] = retrieval_time + generation_time
        
        # 5. Context relevance
        if contexts:
            # Check if question terms appear in contexts
            q_terms = set(question.lower().split())
            ctx_text = " ".join([
                (txt if isinstance(txt, str) else txt.get("text", "")).lower()
                for txt, _, _ in contexts
            ])
            
            relevant_terms = sum(1 for term in q_terms if len(term) > 3 and term in ctx_text)
            metrics["context_term_overlap"] = relevant_terms / max(len(q_terms), 1)
        else:
            metrics["context_term_overlap"] = 0.0
        
        # 6. Add timestamp
        metrics["timestamp"] = time.time()
        metrics["conversation_duration_sec"] = time.time() - self.conversation_start
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_running_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics over the recent window.
        """
        if not self.metrics_history:
            return {}
        
        stats = {}
        
        # Aggregate numeric metrics
        numeric_keys = [
            "avg_retrieval_score", "answer_length", "source_citations",
            "retrieval_time_sec", "generation_time_sec", "total_time_sec",
            "context_term_overlap"
        ]
        
        for key in numeric_keys:
            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                stats[f"{key}_mean"] = sum(values) / len(values)
                stats[f"{key}_max"] = max(values)
                stats[f"{key}_min"] = min(values)
        
        # Aggregate boolean metrics
        total = len(self.metrics_history)
        stats["has_sources_rate"] = sum(1 for m in self.metrics_history if m.get("has_sources", False)) / total
        stats["uncertainty_rate"] = sum(1 for m in self.metrics_history if m.get("expresses_uncertainty", False)) / total
        stats["possible_invention_rate"] = sum(1 for m in self.metrics_history if m.get("possibly_invented", False)) / total
        
        # Quality score (heuristic)
        stats["quality_score"] = (
            stats["has_sources_rate"] * 0.4 +
            (1 - stats["possible_invention_rate"]) * 0.3 +
            min(stats.get("context_term_overlap_mean", 0), 1.0) * 0.3
        )
        
        return stats
    
    def format_metrics_display(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics for display in UI.
        """
        lines = []
        lines.append("📊 **Métricas desta resposta:**")
        lines.append(f"- Contextos recuperados: {metrics['num_contexts']}")
        lines.append(f"- Score médio de retrieval: {metrics['avg_retrieval_score']:.3f}")
        lines.append(f"- Citações de fonte: {metrics['source_citations']}")
        lines.append(f"- Tempo total: {metrics['total_time_sec']:.2f}s (retrieval: {metrics['retrieval_time_sec']:.2f}s)")
        
        if metrics['possibly_invented']:
            lines.append("- ⚠️ **Atenção**: Resposta pode conter conteúdo não fundamentado")
        
        if metrics['expresses_uncertainty']:
            lines.append("- ℹ️ Resposta expressa incerteza (bom sinal)")
        
        return "\n".join(lines)
    
    def format_running_stats_display(self) -> str:
        """
        Format running statistics for sidebar display.
        """
        stats = self.get_running_stats()
        if not stats:
            return "Sem métricas ainda"
        
        lines = []
        lines.append(f"📈 **Estatísticas (últimas {len(self.metrics_history)} respostas):**\n")
        lines.append(f"**Qualidade:**")
        lines.append(f"- Score geral: {stats['quality_score']:.1%}")
        lines.append(f"- Taxa de citação de fontes: {stats['has_sources_rate']:.1%}")
        lines.append(f"- Taxa de invenção possível: {stats['possible_invention_rate']:.1%}")
        lines.append(f"\n**Performance:**")
        lines.append(f"- Tempo médio: {stats['total_time_sec_mean']:.2f}s")
        lines.append(f"- Retrieval médio: {stats['retrieval_time_sec_mean']:.2f}s")
        lines.append(f"\n**Retrieval:**")
        lines.append(f"- Score médio: {stats['avg_retrieval_score_mean']:.3f}")
        lines.append(f"- Overlap de termos: {stats.get('context_term_overlap_mean', 0):.1%}")
        
        return "\n".join(lines)
