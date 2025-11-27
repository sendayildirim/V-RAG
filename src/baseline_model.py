from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict

class BaselineModel:
    """
    RAG olmadan sadece LLM ile cevap ureten baseline model
    google/gemma-2-2b-it modeli kullanir
    """

    def __init__(self, model_name="google/gemma-2-2b-it"):
        self.model_name = model_name

        # Device ayarla (GPU varsa kullan)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Cihaz: {self.device}")

        # Model ve tokenizer yukle
        print(f"Baseline model yukleniyor: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        print("Baseline model yuklendi!")

    def create_prompt(self, question: str) -> str:
        """
        Baseline icin prompt olustur (context yok)
        """
        prompt = f"""You are a helpful assistant. Answer the question about the book "Zuleika Dobson" by Max Beerbohm.

Question: {question}

Answer:"""
        return prompt

    def generate_answer(self, question: str, max_new_tokens: int = 100) -> str:
        """
        Context olmadan sadece soru ile cevap uret
        """
        # Prompt olustur
        prompt = self.create_prompt(question)

        # Tokenize et
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Cevap uret
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Cevabi decode et
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Sadece cevap kismini al (prompt'u cikar)
        answer = full_response[len(prompt):].strip()

        return answer

    def answer_question(self, question: str, max_new_tokens: int = 100) -> Dict:
        """
        Tek bir soruyu cevapla
        """
        answer = self.generate_answer(question, max_new_tokens=max_new_tokens)

        return {
            'question': question,
            'answer': answer
        }

    def batch_answer_questions(self, questions: List[str], max_new_tokens: int = 100) -> List[Dict]:
        """
        Birden fazla soruyu cevapla
        """
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\nSoru {i}/{len(questions)} cevaplanıyor...")
            result = self.answer_question(question, max_new_tokens)
            results.append(result)

        return results

if __name__ == "__main__":
    # Test
    print("Baseline Model test...")

    # Baseline model olustur
    baseline = BaselineModel()

    # Test sorusu
    test_question = "Who are Zuleika's most prominent suitors?"
    result = baseline.answer_question(test_question)

    print(f"\nSoru: {result['question']}")
    print(f"\nCevap: {result['answer']}")
