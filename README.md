# V-RAG: Hierarchical RAG System for Zuleika Dobson

Hiyerarşik parçalama yöntemiyle geliştirilmiş bir Retrieval-Augmented Generation (RAG) sistemi.

## Proje Açıklaması

Bu proje, Project Gutenberg'den "Zuleika Dobson" kitabını kullanarak hiyerarşik chunking ile bir RAG sistemi oluşturur ve NarrativeQA veri setindeki soruları yanıtlar. Sistem, RAG'sız baseline model ile karşılaştırılır ve BLEU & ROUGE metrikleri ile değerlendirilir.

## Teknik Detaylar

- **Kitap:** Zuleika Dobson by Max Beerbohm
- **Dataset:** NarrativeQA (40 test sorusu)
- **Vector Database:** Milvus Lite (disk-based)
- **Embedding Model:** all-MiniLM-L6-v2
- **LLM:** google/gemma-2-2b-it
- **Chunk Yapısı:** 2 seviyeli hiyerarşi (Parent: 512 token, Child: 256 token)
- **Metrikler:** BLEU, ROUGE-1, ROUGE-2, ROUGE-L

## Proje Yapısı

```
V-RAG/
├── data/                          # Veri dosyaları
│   ├── zuleika_dobson.txt        # Kitap metni
│   ├── questions_test.csv        # Test soruları
│   ├── questions_train.csv       # Train soruları (boş)
│   └── questions_valid.csv       # Valid soruları (boş)
├── notebooks/                     # Jupyter notebook'lar
│   └── main_rag_notebook.ipynb   # Ana Colab notebook
├── src/                          # Python modülleri
│   ├── data_loader.py            # Veri indirme
│   ├── chunker.py                # Hiyerarşik parçalama
│   ├── vector_store.py           # Milvus Lite & embedding
│   ├── rag_pipeline.py           # RAG pipeline
│   ├── baseline_model.py         # Baseline model (RAG'sız)
│   ├── metrics.py                # BLEU & ROUGE hesaplama
│   └── experiment_runner.py      # Hiperparametre grid search
├── results/                      # Sonuçlar ve metrikler
├── requirements.txt              # Gerekli kütüphaneler
└── README.md                     # Bu dosya
```

## Kurulum

### Gereksinimler

- Python 3.8+
- CUDA (opsiyonel, GPU kullanımı için)

### Kütüphaneleri Yükle

```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Google Colab'da Çalıştırma (Önerilen)

1. [notebooks/main_rag_notebook.ipynb](notebooks/main_rag_notebook.ipynb) dosyasını Google Colab'da açın
2. Runtime → Change runtime type → GPU seçin
3. Notebook'u sırayla çalıştırın

### 2. Lokal Çalıştırma

```bash
# Veri indirme
python src/data_loader.py

# Chunking ve indexing
python -c "
from chunker import HierarchicalChunker
from vector_store import VectorStore

# Chunking
chunker = HierarchicalChunker(parent_size=512, child_size=256, overlap=50)
with open('data/zuleika_dobson.txt', 'r') as f:
    text = f.read()
parent_chunks, child_chunks = chunker.chunk_text(text)

# Vector store
vs = VectorStore(db_path='./milvus_rag.db')
vs.create_collections()
vs.insert_parent_chunks(parent_chunks)
vs.insert_child_chunks(child_chunks)
"

# Experiment runner (grid search)
python src/experiment_runner.py
```

## Özellikler

### 1. Hiyerarşik Chunking

- **Parent Chunks (512 token):** Geniş context sağlar
- **Child Chunks (256 token):** Detaylı bilgi içerir
- **Overlap:** Bilgi kaybını önler

### 2. Hybrid Retrieval

- Parent ve child chunk'larda parallel arama
- Cosine similarity ile en alakalı chunk'ları bulma
- Parent-child ilişkisini koruma

### 3. Performans Değerlendirme

- **BLEU:** Çeviri kalitesi metriği
- **ROUGE-1, ROUGE-2, ROUGE-L:** Özetleme metrikleri
- RAG vs Baseline karşılaştırması

### 4. Hiperparametre Optimizasyonu

Grid search ile test edilen parametreler:
- **Chunk Size:** 128, 256, 512
- **Overlap:** 0, 25, 50, 100
- **Temperature:** 0.1, 0.2, 0.4, 0.6, 0.8

Toplam: **60 farklı konfigürasyon**

## Sonuçlar

### RAG vs Baseline Karşılaştırması

Sonuçlar `results/rag_vs_baseline.json` dosyasında saklanır.

Örnek çıktı:
```
==============================================================
MODEL KARSILASTIRMASI
==============================================================

RAG Sistemi:
  BLEU:    XX.XX
  ROUGE-1: XX.XX
  ROUGE-2: XX.XX
  ROUGE-L: XX.XX

Baseline (RAG'siz):
  BLEU:    XX.XX
  ROUGE-1: XX.XX
  ROUGE-2: XX.XX
  ROUGE-L: XX.XX

Iyilestirme (RAG - Baseline):
  BLEU:    +X.XX
  ROUGE-1: +X.XX
  ROUGE-2: +X.XX
  ROUGE-L: +X.XX
==============================================================
```

### Grid Search Sonuçları

Experiment sonuçları:
- `results/experiments/experiment_summary.csv` - Tüm deneyler
- `results/experiments/experiment_summary.json` - Detaylı sonuçlar
- Her deney için ayrı JSON dosyası

## Kaynak Kullanımı

### Google Colab (T4 GPU)
- **Chunking:** ~2-5 saniye
- **Indexing:** ~10-20 saniye
- **Inference (40 soru):** ~5-10 dakika
- **Memory:** ~2-4 GB GPU, ~4-6 GB RAM

### Colab Tavsiyeleri
1. GPU runtime kullanın
2. Grid search için Colab Pro düşünün (uzun sürer)
3. Checkpoint'ler kaydedin

## Zorluklar ve Gözlemler

### Zorluklar
1. Google Colab kaynak kısıtları (timeout riski)
2. Grid search süresi (60 deney × 5-10 dk)
3. LLM bellek kullanımı

### Gözlemler
1. Hiyerarşik chunking, context kalitesini artırıyor
2. Optimal chunk size dataset'e bağlı
3. Temperature 0.1-0.4 arası en iyi sonuçları veriyor
4. Overlap 25-50 optimal aralık

## İyileştirme Önerileri

1. **Reranking:** Cross-encoder ile retrieval kalitesini artırma
2. **Query Expansion:** Soru genişletme ile daha iyi retrieval
3. **Larger LLMs:** 7B, 13B modelleri test etme
4. **Multi-hop Reasoning:** İteratif retrieval
5. **Fine-tuning:** Domain-specific embedding modeli

## Referanslar

- [NarrativeQA Dataset](https://github.com/google-deepmind/narrativeqa)
- [Project Gutenberg - Zuleika Dobson](https://www.gutenberg.org/ebooks/1845)
- [Milvus](https://milvus.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [Gemma Models](https://ai.google.dev/gemma)

## Lisans

Bu proje eğitim amaçlıdır.
