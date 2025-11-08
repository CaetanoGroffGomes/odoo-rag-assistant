# 🤖 Assistente RAG para Odoo (LLaMA + FAISS)

Sistema de busca e resposta inteligente (RAG - Retrieval-Augmented Generation) para documentação do Odoo, usando LLaMA 3.1 8B e FAISS para embeddings vetoriais.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37-FF4B4B.svg)](https://streamlit.io)

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Características](#-características)
- [Arquitetura](#-arquitetura)
- [Instalação](#-instalação)
- [Uso](#-uso)
- [Configuração](#-configuração)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Contribuindo](#-contribuindo)

---

## 🎯 Visão Geral

Este projeto implementa um assistente de IA especializado em responder perguntas sobre o Odoo ERP, utilizando:

- **LLaMA 3.1 8B Instruct**: Modelo de linguagem de código aberto para geração de respostas
- **FAISS**: Busca vetorial eficiente para recuperação de documentos  
- **BGE-M3**: Embeddings multilíngues de alta qualidade (1024 dimensões)
- **BM25 + Reranking**: Busca híbrida léxica + semântica
- **MMR**: Diversificação de resultados para evitar redundância

### Como Funciona

```
┌─────────────┐
│   Pergunta  │
│  do Usuário │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Busca Híbrida       │
│ (Semântica + BM25)  │ ──► FAISS Index + BM25
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Reranking + MMR     │ ──► CrossEncoder + diversificação
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ LLaMA 3.1 8B        │
│ (Geração)           │ ──► Resposta + citações [1][2][3]
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Resposta Final      │
│ + Fontes            │
└─────────────────────┘
```

---

## ✨ Características

### 🎯 Core RAG
- ✅ **Busca Híbrida**: Combina busca semântica (FAISS) com busca léxica (BM25)
- ✅ **Reranking**: CrossEncoder para melhorar relevância
- ✅ **MMR**: Diversifica resultados para evitar redundância
- ✅ **Multi-turn Context**: Mantém até 10 turnos de conversa
- ✅ **Source Citation**: Sempre cita fontes nas respostas
- ✅ **Version Preference**: Prioriza versões específicas do Odoo

### 🚀 Performance
- ⚡ **GPU Accelerated**: CUDA para inferência rápida (1-3s/resposta)
- 🎮 **Optimized for 8GB VRAM**: RTX 3060/4060/4070
- 📊 **Real-time Metrics**: Qualidade, latência, alucinações
- 🔄 **Efficient Indexing**: HNSW Flat ou IVF-PQ conforme tamanho

### 🛡️ Qualidade
- ✅ **Anti-hallucination**: Regras estritas contra invenção
- ✅ **Citation Enforcement**: Força citação de fontes
- ✅ **Confidence Scoring**: Avalia confiança da recuperação
- ✅ **Abstention Detection**: Identifica quando não há info suficiente

### 📊 Métricas
- 📈 **Real-time Eval**: Métricas por resposta e agregadas
- 🎯 **Quality Score**: Heurística de qualidade
- ⚠️ **Hallucination Detection**: Identifica possíveis alucinações
- 📉 **Performance Tracking**: Latência, tokens, VRAM

---

## 🏗️ Arquitetura

### Componentes

```
projeto/
│
├── scrape_docs.py          # Crawler assíncrono
├── build_index.py          # Constrói índice FAISS
├── query.py                # Pipeline RAG completo
├── app_chat.py             # Interface Streamlit
├── evals.py                # Avaliação offline
├── realtime_evals.py       # Métricas tempo real
│
├── modules.yml             # Mapeamento módulos Odoo
├── docs.json               # Documentação (gerado)
├── faiss_index.bin         # Índice FAISS (gerado)
└── metadatas.json          # Metadados (gerado)
```

### Pipeline

#### 1. Scraping
- Crawler assíncrono com checkpoint
- Respeita robots.txt
- Extrai texto limpo de HTML

#### 2. Indexing
- **Chunking**: ~140 palavras, overlap 40
- **Embeddings**: BGE-M3 (1024d)
- **FAISS**: HNSW Flat ou IVF-PQ
- **BM25**: Índice léxico paralelo

#### 3. Retrieval
- Busca semântica (FAISS)
- Busca léxica (BM25)
- Fusão (RRF)
- Boost de versão
- Diversificação (MMR)

#### 4. Generation
- Formata contextos + histórico
- LLaMA 3.1 8B (temp=0.2)
- Pós-processamento

---

## 💾 Instalação

### Requisitos
- Python 3.10+
- CUDA 11.8+ (para GPU)
- 16GB+ RAM
- 8GB+ VRAM (para LLaMA 8B)

### 1. Clone o Repositório
```bash
git clone https://github.com/seu-usuario/odoo-rag-assistant.git
cd odoo-rag-assistant
```

### 2. Ambiente Virtual
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 3. Instale Dependências

**GPU (recomendado):**
```bash
# PyTorch com CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# llama-cpp-python com CUDA
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# Outras dependências
pip install -r requirements.txt
```

**CPU apenas:**
```bash
pip install -r requirements.txt
```

### 4. Baixe Modelo LLaMA

**LLaMA 3.1 8B Q4_K_M (4.9 GB):**
```bash
mkdir -p models
wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -O models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

### 5. Configure

Crie `.env`:
```bash
LLM_PATH=models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
LLM_CTX=8192
MAX_HISTORY_TURNS=10
TOP_K_RETRIEVAL=12
PREFERRED_VERSION_DEFAULT=16.0
```

---

## 🚀 Uso

### 1. Scraping
```bash
python scrape_docs.py --start-url "https://www.odoo.com/documentation/16.0/pt_BR/" --max-pages 600
```

### 2. Construir Índice
```bash
python build_index.py
```

Gera:
- `faiss_index.bin`
- `embeddings.memmap`
- `metadatas.json`

### 3. Rodar App
```bash
streamlit run app_chat.py
```

Abre em `http://localhost:8501`

### 4. Avaliar
```bash
python evals.py --eval eval_set.jsonl
```

---

## ⚙️ Configuração

### Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `LLM_PATH` | - | Caminho modelo GGUF |
| `LLM_CTX` | 8192 | Janela contexto |
| `MAX_HISTORY_TURNS` | 10 | Turnos mantidos |
| `MAX_SNIPPET_CHARS` | 1800 | Chars por snippet |
| `TOP_K_RETRIEVAL` | 12 | Docs recuperados |
| `TOP_K_FINAL` | 4 | Docs finais |
| `ENABLE_MMR` | 1 | Ativar MMR |
| `USE_RERANK` | 1 | Ativar reranking |

### GPU

Em `query.py`, função `load_llm()`:

```python
n_gpu_layers=33,    # Camadas no GPU
                    # 0 = CPU
                    # 33 = ~6.5GB VRAM (8B)
                    # 40 = ~7.5GB VRAM
                    
n_threads=8,        # Threads CPU
```

---

## 📁 Estrutura

```
odoo-rag-assistant/
│
├── README.md
├── requirements.txt
├── .env.example
│
├── scrape_docs.py
├── build_index.py
├── query.py
├── app_chat.py
├── evals.py
├── realtime_evals.py
│
├── modules.yml
├── eval_set.jsonl
│
├── docs/
│   ├── INSTALLATION_GUIDE_8B.md
│   ├── QUICK_START_8B.md
│   └── rag_improvements_analysis.md
│
├── data/ (gitignore)
│   ├── docs.json
│   ├── faiss_index.bin
│   └── metadatas.json
│
└── models/ (gitignore)
    └── *.gguf
```

---

## 🐛 Troubleshooting

### Model not found
```bash
# Verifique path
ls -lh models/*.gguf

# Atualize .env
LLM_PATH=models/seu-modelo.gguf
```

### CUDA OOM
```python
# query.py linha ~195
n_gpu_layers=25,  # Reduza
```

### Respostas lentas
```python
# Aumente camadas GPU
n_gpu_layers=40,
```

### História quebrada
```python
# Formato correto:
st.session_state.history = [
    ("user", "msg"),
    ("assistant", "resp"),
]
```

---

## 🤝 Contribuindo

1. Fork o repositório
2. Crie branch (`git checkout -b feature/Amazing`)
3. Commit (`git commit -m 'Add Amazing'`)
4. Push (`git push origin feature/Amazing`)
5. Abra Pull Request

### Guidelines
- Python 3.10+ compatível
- Adicione testes
- Atualize docs
- Siga PEP 8

---

## 📝 Licença

MIT License - veja [LICENSE](LICENSE)

---

## 🙏 Agradecimentos

- **Meta AI** - LLaMA
- **Beijing Academy of AI** - BGE
- **Facebook AI Research** - FAISS
- **Odoo SA** - Documentação

---

## 📧 Contato

- Issues: [GitHub Issues](https://github.com/seu-usuario/odoo-rag-assistant/issues)
- Discussions: [GitHub Discussions](https://github.com/seu-usuario/odoo-rag-assistant/discussions)

---

## 🗺️ Roadmap

- [ ] Múltiplas línguas (ES, EN, FR)
- [ ] Fine-tuning
- [ ] API REST
- [ ] Plugin Odoo
- [ ] Suporte imagens/PDFs
- [ ] Sistema de feedback
- [ ] Dashboard analytics

---

**Feito com ❤️ para a comunidade Odoo**
