import time
import psutil
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from chunker import HierarchicalChunker
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from metrics import MetricsEvaluator

class ExperimentRunner:
    """
    Farkli chunk size, overlap ve temperature parametreleri ile
    RAG sisteminin performansini test eden sinif
    """

    def __init__(self, book_path: str, test_questions_path: str, results_dir: str = "results"):
        self.book_path = book_path
        self.test_questions_path = test_questions_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Kitap metnini yukle
        with open(book_path, 'r', encoding='utf-8') as f:
            self.book_text = f.read()

        # Test sorularini yukle
        self.test_df = pd.read_csv(test_questions_path)

        # Evaluator olustur
        self.evaluator = MetricsEvaluator()

        # Deneyler icin process bilgisi
        self.process = psutil.Process(os.getpid())

    def get_memory_usage(self) -> float:
        """
        Mevcut memory kulanimini MB cinsinden dondur
        """
        return self.process.memory_info().rss / 1024 / 1024

    def run_single_experiment(self, child_size: int, overlap: int, temperature: float) -> Dict:
        """
        Tek bir konfigurasyonla deney calistir
        """
        print(f"\n{'='*70}")
        print(f"Deney: child_size={child_size}, overlap={overlap}, temperature={temperature}")
        print(f"{'='*70}")

        parent_size = child_size * 2
        db_path = f"./milvus_c{child_size}_o{overlap}.db"

        # Baslangic metrikleri
        start_time = time.time()
        start_memory = self.get_memory_usage()

        # 1. Chunking
        print("\n1. Chunking yapiliyor...")
        chunking_start = time.time()
        chunker = HierarchicalChunker(
            parent_size=parent_size,
            child_size=child_size,
            overlap=overlap
        )
        parent_chunks, child_chunks = chunker.chunk_text(self.book_text)
        chunking_time = time.time() - chunking_start

        chunk_stats = chunker.get_chunk_stats(parent_chunks, child_chunks)

        # 2. Vector Store olustur ve indexle
        print("\n2. Vector store olusturuluyor ve indexleniyor...")
        indexing_start = time.time()
        vs = VectorStore(db_path=db_path)
        vs.create_collections()
        vs.insert_parent_chunks(parent_chunks)
        vs.insert_child_chunks(child_chunks)
        indexing_time = time.time() - indexing_start

        # 3. RAG Pipeline ile cevapla
        print("\n3. Sorular cevaplanıyor...")
        rag = RAGPipeline(vector_store=vs)

        # Temperature'u guncelle
        rag.temperature = temperature

        questions = self.test_df['question'].tolist()

        inference_start = time.time()
        # Her soru icin ayri zamanlama
        question_times = []
        rag_results = []

        for i, question in enumerate(questions, 1):
            q_start = time.time()
            result = rag.answer_question(question, max_new_tokens=100)
            q_time = time.time() - q_start
            question_times.append(q_time)
            rag_results.append(result)
            print(f"  Soru {i}/{len(questions)} - {q_time:.2f}s")

        inference_time = time.time() - inference_start
        avg_question_time = sum(question_times) / len(question_times)

        # 4. Metrikleri hesapla
        print("\n4. Metrikler hesaplaniyor...")
        predictions = [r['answer'] for r in rag_results]
        references = []
        for _, row in self.test_df.iterrows():
            refs = [row['answer1'], row['answer2']]
            references.append(refs)

        metrics = self.evaluator.evaluate(predictions, references)

        # 5. Retrieval kalitesi
        print("\n5. Retrieval kalitesi olculuyor...")
        retrieval_scores = []
        for result in rag_results:
            # Context'ten score bilgisini cikaramiyoruz, bu yuzden basit bir metrik kullan
            # Gerçek implementasyonda retrieval score'lari saklanabilir
            retrieval_scores.append(1.0)  # Placeholder

        avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores)

        # 6. Resource kullanimi
        end_memory = self.get_memory_usage()
        total_time = time.time() - start_time

        # Vector DB boyutu
        db_size = os.path.getsize(db_path) / 1024 / 1024 if os.path.exists(db_path) else 0

        # Temizlik
        vs.close()

        # Sonuclari topla
        results = {
            'config': {
                'child_size': child_size,
                'parent_size': parent_size,
                'overlap': overlap,
                'temperature': temperature
            },
            'chunk_stats': chunk_stats,
            'metrics': metrics,
            'performance': {
                'chunking_time': chunking_time,
                'indexing_time': indexing_time,
                'inference_time': inference_time,
                'avg_question_time': avg_question_time,
                'total_time': total_time,
                'db_size_mb': db_size,
                'memory_used_mb': end_memory - start_memory,
                'avg_retrieval_score': avg_retrieval_score
            }
        }

        print(f"\nDeney tamamlandi! Toplam sure: {total_time:.2f}s")
        return results

    def run_grid_search(self, chunk_sizes: List[int], overlaps: List[int],
                       temperatures: List[float]) -> List[Dict]:
        """
        Tum parametre kombinasyonlari icin deney calistir
        """
        all_results = []
        total_experiments = len(chunk_sizes) * len(overlaps) * len(temperatures)
        current_experiment = 0

        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                for temperature in temperatures:
                    current_experiment += 1
                    print(f"\n\n{'#'*70}")
                    print(f"GENEL ILERLEME: {current_experiment}/{total_experiments}")
                    print(f"{'#'*70}")

                    try:
                        result = self.run_single_experiment(chunk_size, overlap, temperature)
                        all_results.append(result)

                        # Her deneyi kaydet
                        self.save_single_result(result)

                    except Exception as e:
                        print(f"HATA: {e}")
                        continue

        return all_results

    def save_single_result(self, result: Dict):
        """
        Tek bir deney sonucunu kaydet
        """
        config = result['config']
        filename = f"exp_c{config['child_size']}_o{config['overlap']}_t{config['temperature']}.json"
        filepath = self.results_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def save_summary(self, all_results: List[Dict], summary_filename: str = "experiment_summary"):
        """
        Tum sonuclarin ozetini CSV ve JSON olarak kaydet

        Args:
            all_results: Deney sonuclari listesi
            summary_filename: Ozet dosyalarinin adi (uzantisiz)
        """
        # DataFrame olustur
        summary_data = []
        for result in all_results:
            config = result['config']
            metrics = result['metrics']
            perf = result['performance']

            summary_data.append({
                'child_size': config['child_size'],
                'parent_size': config['parent_size'],
                'overlap': config['overlap'],
                'temperature': config['temperature'],
                'bleu': metrics['bleu'],
                'rouge1': metrics['rouge1'],
                'rouge2': metrics['rouge2'],
                'rougeL': metrics['rougeL'],
                'chunking_time': perf['chunking_time'],
                'indexing_time': perf['indexing_time'],
                'inference_time': perf['inference_time'],
                'avg_question_time': perf['avg_question_time'],
                'total_time': perf['total_time'],
                'db_size_mb': perf['db_size_mb'],
                'memory_used_mb': perf['memory_used_mb']
            })

        df = pd.DataFrame(summary_data)

        # CSV olarak kaydet
        csv_path = self.results_dir / f"{summary_filename}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nOzet CSV kaydedildi: {csv_path}")

        # JSON olarak kaydet
        json_path = self.results_dir / f"{summary_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Ozet JSON kaydedildi: {json_path}")

        # En iyi sonuclari bul
        self.print_best_results(df)

    def print_best_results(self, df: pd.DataFrame):
        """
        En iyi sonuclari yazdir
        """
        print("\n" + "="*70)
        print("EN IYI SONUCLAR")
        print("="*70)

        # En yuksek BLEU
        best_bleu = df.loc[df['bleu'].idxmax()]
        print(f"\nEn Yuksek BLEU ({best_bleu['bleu']:.2f}):")
        print(f"  child_size={best_bleu['child_size']}, overlap={best_bleu['overlap']}, temp={best_bleu['temperature']}")

        # En yuksek ROUGE-L
        best_rougeL = df.loc[df['rougeL'].idxmax()]
        print(f"\nEn Yuksek ROUGE-L ({best_rougeL['rougeL']:.2f}):")
        print(f"  child_size={best_rougeL['child_size']}, overlap={best_rougeL['overlap']}, temp={best_rougeL['temperature']}")

        # En hizli
        fastest = df.loc[df['total_time'].idxmin()]
        print(f"\nEn Hizli ({fastest['total_time']:.2f}s):")
        print(f"  child_size={fastest['child_size']}, overlap={fastest['overlap']}, temp={fastest['temperature']}")

        # En az memory
        min_memory = df.loc[df['memory_used_mb'].idxmin()]
        print(f"\nEn Az Memory ({min_memory['memory_used_mb']:.2f} MB):")
        print(f"  child_size={min_memory['child_size']}, overlap={min_memory['overlap']}, temp={min_memory['temperature']}")

        print("="*70)

if __name__ == "__main__":
    # Test parametreleri
    chunk_sizes = [128, 256, 512]
    overlaps = [0, 25, 50, 100]
    temperatures = [0.1, 0.2, 0.4, 0.6, 0.8]

    # Experiment runner olustur
    runner = ExperimentRunner(
        book_path="data/zuleika_dobson.txt",
        test_questions_path="data/questions_test.csv"
    )

    # Grid search calistir
    print("Grid search baslatiliyor...")
    print(f"Toplam {len(chunk_sizes) * len(overlaps) * len(temperatures)} deney yapilacak")

    all_results = runner.run_grid_search(chunk_sizes, overlaps, temperatures)

    # Sonuclari kaydet
    runner.save_summary(all_results)

    print("\n\nTum deneyler tamamlandi!")
