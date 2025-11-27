import re
import tiktoken
from typing import List, Dict, Tuple

class HierarchicalChunker:
    """
    2 seviyeli hiyerarşik metin parçalama sınıfı
    Parent: 512 token, Child: 256 token
    """

    def __init__(self, parent_size=512, child_size=256, overlap=50):
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def clean_gutenberg_text(self, text: str) -> str:
        """
        Project Gutenberg header ve footer'larini temizle
        """
        # "*** START OF" ile "*** END OF" arasindaki metni al
        start_pattern = r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*"
        end_pattern = r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*"

        start_match = re.search(start_pattern, text, re.IGNORECASE | re.DOTALL)
        end_match = re.search(end_pattern, text, re.IGNORECASE | re.DOTALL)

        if start_match and end_match:
            text = text[start_match.end():end_match.start()]

        # Fazla boslukları ve satır atlamalarını temizle
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        return text.strip()

    def count_tokens(self, text: str) -> int:
        """
        Metindeki token sayisini hesapla
        """
        return len(self.tokenizer.encode(text))

    def split_by_tokens(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Metni belirli token sayisina gore parcala
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Overlap ile devam et
            start += (chunk_size - overlap)

        return chunks

    def create_parent_chunks(self, text: str) -> List[Dict]:
        """
        Parent chunk'lari olustur (512 token)
        """
        chunks = self.split_by_tokens(text, self.parent_size, self.overlap)

        parent_chunks = []
        for idx, chunk in enumerate(chunks):
            parent_chunks.append({
                'id': f'parent_{idx}',
                'text': chunk,
                'token_count': self.count_tokens(chunk),
                'type': 'parent',
                'index': idx
            })

        return parent_chunks

    def create_child_chunks(self, parent_chunk: Dict) -> List[Dict]:
        """
        Bir parent chunk'tan child chunk'lar olustur (256 token)
        """
        parent_text = parent_chunk['text']
        parent_id = parent_chunk['id']
        parent_idx = parent_chunk['index']

        chunks = self.split_by_tokens(parent_text, self.child_size, self.overlap // 2)

        child_chunks = []
        for idx, chunk in enumerate(chunks):
            child_chunks.append({
                'id': f'child_{parent_idx}_{idx}',
                'text': chunk,
                'token_count': self.count_tokens(chunk),
                'type': 'child',
                'parent_id': parent_id,
                'index': idx
            })

        return child_chunks

    def chunk_text(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Metni hiyerarşik olarak parcala
        Returns: (parent_chunks, child_chunks)
        """
        # Metni temizle
        clean_text = self.clean_gutenberg_text(text)

        # Parent chunk'lari olustur
        parent_chunks = self.create_parent_chunks(clean_text)

        # Her parent icin child chunk'lar olustur
        all_child_chunks = []
        for parent in parent_chunks:
            child_chunks = self.create_child_chunks(parent)
            all_child_chunks.extend(child_chunks)

        print(f"Toplam {len(parent_chunks)} parent chunk olusturuldu")
        print(f"Toplam {len(all_child_chunks)} child chunk olusturuldu")

        return parent_chunks, all_child_chunks

    def get_chunk_stats(self, parent_chunks: List[Dict], child_chunks: List[Dict]) -> Dict:
        """
        Chunk istatistiklerini hesapla
        """
        parent_tokens = [c['token_count'] for c in parent_chunks]
        child_tokens = [c['token_count'] for c in child_chunks]

        stats = {
            'parent_count': len(parent_chunks),
            'child_count': len(child_chunks),
            'parent_avg_tokens': sum(parent_tokens) / len(parent_tokens) if parent_tokens else 0,
            'child_avg_tokens': sum(child_tokens) / len(child_tokens) if child_tokens else 0,
            'parent_max_tokens': max(parent_tokens) if parent_tokens else 0,
            'child_max_tokens': max(child_tokens) if child_tokens else 0
        }

        return stats

if __name__ == "__main__":
    # Test
    chunker = HierarchicalChunker(parent_size=512, child_size=256)

    with open('data/zuleika_dobson.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    parent_chunks, child_chunks = chunker.chunk_text(text)
    stats = chunker.get_chunk_stats(parent_chunks, child_chunks)

    print("\nChunk İstatistikleri:")
    print(f"  Parent sayısı: {stats['parent_count']}")
    print(f"  Child sayısı: {stats['child_count']}")
    print(f"  Parent ortalama token: {stats['parent_avg_tokens']:.1f}")
    print(f"  Child ortalama token: {stats['child_avg_tokens']:.1f}")
