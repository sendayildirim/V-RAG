from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
import pandas as pd
import json
from typing import List, Dict
from pathlib import Path

class MetricsEvaluator:
    """
    BLEU ve ROUGE metrikleri ile model performansini degerlendiren sinif
    """

    def __init__(self):
        # ROUGE scorer olustur (ROUGE-1, ROUGE-2, ROUGE-L)
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def calculate_bleu(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        BLEU skorunu hesapla
        predictions: Model tahmini cevaplar
        references: Her soru icin referans cevaplar listesi
        """
        # sacrebleu icin referanslari transpoze et
        refs_transposed = list(zip(*references))

        # BLEU skorunu hesapla
        bleu_result = corpus_bleu(predictions, refs_transposed)

        return {
            'bleu': bleu_result.score,
            'bleu_detail': str(bleu_result)
        }

    def calculate_rouge(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        ROUGE skorlarini hesapla
        Her tahmin icin tum referanslarla karsilastir ve en yuksek skoru al
        """
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for pred, refs in zip(predictions, references):
            # Her referans icin ROUGE hesapla ve maximum'u al
            max_rouge1 = 0
            max_rouge2 = 0
            max_rougeL = 0

            for ref in refs:
                scores = self.rouge_scorer.score(ref, pred)
                max_rouge1 = max(max_rouge1, scores['rouge1'].fmeasure)
                max_rouge2 = max(max_rouge2, scores['rouge2'].fmeasure)
                max_rougeL = max(max_rougeL, scores['rougeL'].fmeasure)

            rouge1_scores.append(max_rouge1)
            rouge2_scores.append(max_rouge2)
            rougeL_scores.append(max_rougeL)

        # Ortalama skorlari hesapla
        return {
            'rouge1': sum(rouge1_scores) / len(rouge1_scores) * 100,
            'rouge2': sum(rouge2_scores) / len(rouge2_scores) * 100,
            'rougeL': sum(rougeL_scores) / len(rougeL_scores) * 100
        }

    def evaluate(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        Tum metrikleri hesapla
        """
        bleu_scores = self.calculate_bleu(predictions, references)
        rouge_scores = self.calculate_rouge(predictions, references)

        return {
            **bleu_scores,
            **rouge_scores
        }

    def compare_models(self, rag_results: List[Dict], baseline_results: List[Dict],
                      ground_truth: pd.DataFrame) -> Dict:
        """
        RAG ve Baseline modellerini karsilastir
        """
        # Tahminleri ve referanslari hazirla
        rag_predictions = [r['answer'] for r in rag_results]
        baseline_predictions = [r['answer'] for r in baseline_results]

        # Referans cevaplari hazirla (answer1 ve answer2)
        references = []
        for _, row in ground_truth.iterrows():
            refs = [row['answer1'], row['answer2']]
            references.append(refs)

        # RAG metrikleri
        print("RAG metrikleri hesaplaniyor...")
        rag_metrics = self.evaluate(rag_predictions, references)

        # Baseline metrikleri
        print("Baseline metrikleri hesaplaniyor...")
        baseline_metrics = self.evaluate(baseline_predictions, references)

        # Karsilastirma
        comparison = {
            'rag': rag_metrics,
            'baseline': baseline_metrics,
            'improvement': {
                'bleu': rag_metrics['bleu'] - baseline_metrics['bleu'],
                'rouge1': rag_metrics['rouge1'] - baseline_metrics['rouge1'],
                'rouge2': rag_metrics['rouge2'] - baseline_metrics['rouge2'],
                'rougeL': rag_metrics['rougeL'] - baseline_metrics['rougeL']
            }
        }

        return comparison

    def save_results(self, comparison: Dict, output_path: str):
        """
        Sonuclari JSON dosyasina kaydet
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        print(f"\nSonuclar kaydedildi: {output_path}")

    def print_comparison(self, comparison: Dict):
        """
        Karsilastirma sonuclarini yazdir
        """
        print("\n" + "="*60)
        print("MODEL KARSILASTIRMASI")
        print("="*60)

        print("\nRAG Sistemi:")
        print(f"  BLEU:    {comparison['rag']['bleu']:.2f}")
        print(f"  ROUGE-1: {comparison['rag']['rouge1']:.2f}")
        print(f"  ROUGE-2: {comparison['rag']['rouge2']:.2f}")
        print(f"  ROUGE-L: {comparison['rag']['rougeL']:.2f}")

        print("\nBaseline (RAG'siz):")
        print(f"  BLEU:    {comparison['baseline']['bleu']:.2f}")
        print(f"  ROUGE-1: {comparison['baseline']['rouge1']:.2f}")
        print(f"  ROUGE-2: {comparison['baseline']['rouge2']:.2f}")
        print(f"  ROUGE-L: {comparison['baseline']['rougeL']:.2f}")

        print("\nIyilestirme (RAG - Baseline):")
        print(f"  BLEU:    {comparison['improvement']['bleu']:+.2f}")
        print(f"  ROUGE-1: {comparison['improvement']['rouge1']:+.2f}")
        print(f"  ROUGE-2: {comparison['improvement']['rouge2']:+.2f}")
        print(f"  ROUGE-L: {comparison['improvement']['rougeL']:+.2f}")

        print("="*60)

if __name__ == "__main__":
    # Test
    evaluator = MetricsEvaluator()

    # Ornek veri
    predictions = ["The Duke of Dorset and Noaks"]
    references = [["The Duke of Dorset and Noaks", "Duke of Dorset and Noaks"]]

    metrics = evaluator.evaluate(predictions, references)
    print("Test Metrikleri:")
    print(f"  BLEU: {metrics['bleu']:.2f}")
    print(f"  ROUGE-1: {metrics['rouge1']:.2f}")
    print(f"  ROUGE-2: {metrics['rouge2']:.2f}")
    print(f"  ROUGE-L: {metrics['rougeL']:.2f}")
