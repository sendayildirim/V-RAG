from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import numpy as np

class VectorStore:
    """
    Milvus Lite ile vektor veritabani yonetimi
    all-MiniLM-L6-v2 embedding modeli kullanir
    """

    def __init__(self, db_path="./milvus_lite.db", model_name="all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model_name = model_name

        # Embedding modelini yukle
        print(f"Embedding modeli yukleniyor: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Milvus client olustur
        self.client = MilvusClient(db_path)

        # Collection isimleri
        self.parent_collection = "parent_chunks"
        self.child_collection = "child_chunks"

    def create_collections(self):
        """
        Parent ve child chunk'lar icin collection'lari olustur
        """
        # Onceki collection'lari sil
        if self.client.has_collection(self.parent_collection):
            self.client.drop_collection(self.parent_collection)
        if self.client.has_collection(self.child_collection):
            self.client.drop_collection(self.child_collection)

        # Parent collection schema
        parent_schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True
        )
        parent_schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=100, is_primary=True)
        parent_schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        parent_schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        parent_schema.add_field(field_name="index", datatype=DataType.INT64)

        # Child collection schema
        child_schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True
        )
        child_schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=100, is_primary=True)
        child_schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        child_schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        child_schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
        child_schema.add_field(field_name="index", datatype=DataType.INT64)

        # Index parametreleri (FLAT - tam dogruluk)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="FLAT",
            metric_type="IP"  # Inner Product (cosine similarity icin)
        )

        # Collection'lari olustur
        self.client.create_collection(
            collection_name=self.parent_collection,
            schema=parent_schema,
            index_params=index_params
        )

        self.client.create_collection(
            collection_name=self.child_collection,
            schema=child_schema,
            index_params=index_params
        )

        print(f"Collection'lar olusturuldu: {self.parent_collection}, {self.child_collection}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Metinleri embedding vektorlerine donustur
        """
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        return embeddings

    def insert_parent_chunks(self, parent_chunks: List[Dict]):
        """
        Parent chunk'lari veritabanina ekle
        """
        if not parent_chunks:
            return

        # Embedding'leri olustur
        texts = [chunk['text'] for chunk in parent_chunks]
        embeddings = self.embed_texts(texts)

        # Verileri hazirla
        data = []
        for chunk, embedding in zip(parent_chunks, embeddings):
            data.append({
                'id': chunk['id'],
                'embedding': embedding.tolist(),
                'text': chunk['text'],
                'index': chunk['index']
            })

        # Veritabanina ekle
        self.client.insert(collection_name=self.parent_collection, data=data)
        print(f"{len(parent_chunks)} parent chunk eklendi")

    def insert_child_chunks(self, child_chunks: List[Dict]):
        """
        Child chunk'lari veritabanina ekle
        """
        if not child_chunks:
            return

        # Embedding'leri olustur
        texts = [chunk['text'] for chunk in child_chunks]
        embeddings = self.embed_texts(texts)

        # Verileri hazirla
        data = []
        for chunk, embedding in zip(child_chunks, embeddings):
            data.append({
                'id': chunk['id'],
                'embedding': embedding.tolist(),
                'text': chunk['text'],
                'parent_id': chunk['parent_id'],
                'index': chunk['index']
            })

        # Veritabanina ekle
        self.client.insert(collection_name=self.child_collection, data=data)
        print(f"{len(child_chunks)} child chunk eklendi")

    def search_parents(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Parent chunk'larda similarity search
        """
        # Query'yi embedding'e donustur
        query_embedding = self.embed_texts([query])[0]

        # Arama yap
        results = self.client.search(
            collection_name=self.parent_collection,
            data=[query_embedding.tolist()],
            limit=top_k,
            output_fields=["id", "text", "index"]
        )

        # Sonuclari formatla
        formatted_results = []
        for hit in results[0]:
            formatted_results.append({
                'id': hit['id'],
                'text': hit['entity']['text'],
                'index': hit['entity']['index'],
                'score': hit['distance']
            })

        return formatted_results

    def search_children(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Child chunk'larda similarity search
        """
        # Query'yi embedding'e donustur
        query_embedding = self.embed_texts([query])[0]

        # Arama yap
        results = self.client.search(
            collection_name=self.child_collection,
            data=[query_embedding.tolist()],
            limit=top_k,
            output_fields=["id", "text", "parent_id", "index"]
        )

        # Sonuclari formatla
        formatted_results = []
        for hit in results[0]:
            formatted_results.append({
                'id': hit['id'],
                'text': hit['entity']['text'],
                'parent_id': hit['entity']['parent_id'],
                'index': hit['entity']['index'],
                'score': hit['distance']
            })

        return formatted_results

    def hybrid_search(self, query: str, top_parents: int = 3, top_children: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """
        Hybrid search: Hem parent hem child chunk'larda ara
        """
        parent_results = self.search_parents(query, top_k=top_parents)
        child_results = self.search_children(query, top_k=top_children)

        return parent_results, child_results

    def get_children_of_parent(self, parent_id: str) -> List[Dict]:
        """
        Belirli bir parent'in tum child'larini getir
        """
        results = self.client.query(
            collection_name=self.child_collection,
            filter=f'parent_id == "{parent_id}"',
            output_fields=["id", "text", "index"]
        )

        return results

    def close(self):
        """
        Baglantilari kapat
        """
        self.client.close()

if __name__ == "__main__":
    # Test
    print("Vector store test baslatiliyor...")

    vs = VectorStore(db_path="./test_milvus.db")
    vs.create_collections()

    # Ornek veri
    parent_chunks = [
        {'id': 'parent_0', 'text': 'This is a test parent chunk.', 'index': 0}
    ]
    child_chunks = [
        {'id': 'child_0_0', 'text': 'This is test.', 'parent_id': 'parent_0', 'index': 0},
        {'id': 'child_0_1', 'text': 'This is chunk.', 'parent_id': 'parent_0', 'index': 1}
    ]

    vs.insert_parent_chunks(parent_chunks)
    vs.insert_child_chunks(child_chunks)

    # Arama testi
    results = vs.search_parents("test", top_k=1)
    print(f"\nArama sonucu: {len(results)} parent bulundu")

    vs.close()
    print("Test tamamlandi!")
