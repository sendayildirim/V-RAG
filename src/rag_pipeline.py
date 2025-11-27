from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict
from vector_store import VectorStore

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline
    google/gemma-2-2b-it modeli kullanir
    """

    def __init__(self, vector_store: VectorStore, model_name="google/gemma-2-2b-it", temperature=0.1):
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature

        # Device ayarla (GPU varsa kullan)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Cihaz: {self.device}")

        # Model ve tokenizer yukle
        print(f"Model yukleniyor: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        print("Model yuklendi!")

    def retrieve_context(self, question: str, top_k_children: int = 5) -> str:
        """
        Soru icin en alakali chunk'lari getir ve context olustur
        """
        # Child chunk'larda ara
        _, child_results = self.vector_store.hybrid_search(
            query=question,
            top_parents=3,
            top_children=top_k_children
        )

        # Context olustur
        context_parts = []
        for i, result in enumerate(child_results, 1):
            context_parts.append(f"[{i}] {result['text']}")

        context = "\n\n".join(context_parts)
        return context

    def create_prompt(self, question: str, context: str) -> str:
        """
        RAG için prompt olustur
        """
        prompt = f"""You are a helpful assistant. Answer the question based on the given context from the book "Zuleika Dobson".

Context:
{context}

Question: {question}

Answer:"""
        return prompt

    def generate_answer(self, question: str, context: str, max_new_tokens: int = 100) -> str:
        """
        Verilen context ve soru ile cevap uret
        """
        # Prompt olustur
        prompt = self.create_prompt(question, context)

        # Tokenize et
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Cevap uret
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Cevabi decode et
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Sadece cevap kismini al (prompt'u cikar)
        answer = full_response[len(prompt):].strip()

        return answer

    def answer_question(self, question: str, top_k_children: int = 5, max_new_tokens: int = 100) -> Dict:
        """
        RAG pipeline: Retrieve + Generate
        """
        # Context'i al
        context = self.retrieve_context(question, top_k_children=top_k_children)

        # Cevap uret
        answer = self.generate_answer(question, context, max_new_tokens=max_new_tokens)

        return {
            'question': question,
            'context': context,
            'answer': answer
        }

    def batch_answer_questions(self, questions: List[str], top_k_children: int = 5,
                               max_new_tokens: int = 100) -> List[Dict]:
        """
        Birden fazla soruyu cevapla
        """
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\nSoru {i}/{len(questions)} cevaplanıyor...")
            result = self.answer_question(question, top_k_children, max_new_tokens)
            results.append(result)

        return results

if __name__ == "__main__":
    # Test
    print("RAG Pipeline test...")

    # Vector store olustur (onceden chunk'lar eklenmiş olmalı)
    vs = VectorStore(db_path="./test_milvus.db")

    # RAG pipeline olustur
    rag = RAGPipeline(vector_store=vs)

    # Test sorusu
    test_question = "Who are Zuleika's most prominent suitors?"
    result = rag.answer_question(test_question)

    print(f"\nSoru: {result['question']}")
    print(f"\nContext:\n{result['context'][:200]}...")
    print(f"\nCevap: {result['answer']}")
