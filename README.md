# V-RAG: Hiyerarşik RAG Sistemi

Max Beerbohm'un "Zuleika Dobson" kitabı üzerine geliştirilmiş bir soru-cevap sistemi.

## Nedir?

Bu projede NarrativeQA veri setinden gelen soruları yanıtlamak için RAG (Retrieval-Augmented Generation) yaklaşımı kullandım. Kitap metnini hiyerarşik bir şekilde parçalayıp vektör veritabanında saklayarak, modelin sadece ezberden değil ilgili bağlamı kullanarak cevap vermesini sağladım.

## Neden yaptım?

Büyük dil modelleri bazen bilmediği şeyleri uydurabiliyor (hallucination). RAG kullanarak modele "önce bak, sonra cevapla" mantığını öğretmeye çalıştım. Ayrıca farklı chunk boyutları ve parametrelerin performansa etkisini merak ediyordum.

## Nasıl çalışıyor?

### Temel mantık:
1. Kitabı küçük parçalara böl (chunking)
2. Her parçayı vektöre çevir ve veritabanına kaydet
3. Soru geldiğinde ilgili parçaları bul
4. Bu parçaları okuyarak cevap üret

### Kullandığım teknolojiler:
- **Model:** google/gemma-2-2b-it (2 milyar parametre, Colab'da çalışabilecek boyutta)
- **Embedding:** all-MiniLM-L6-v2 (hızlı ve yeterince iyi)
- **Veritabanı:** Milvus Lite (disk kullanıyor, RAM şişirmiyor)
- **Değerlendirme:** BLEU ve ROUGE metrikleri

## Dosyalar

```
V-RAG/
├── data/                   # İndirilen veri
├── notebooks/              # Ana çalışma notebook'u burada
├── src/                    # Modüler Python kodları
├── results/                # Deney sonuçları
└── requirements.txt
```

## Nasıl çalıştırılır?

Google Colab kullanmanızı öneririm çünkü GPU lazım.

1. `notebooks/main_rag_notebook.ipynb` dosyasını Colab'da aç
2. Runtime ayarlarından GPU'yu aktif et (tercihen T4)
3. Hücreleri sırayla çalıştır

İlk çalıştırmada kütüphaneler yüklenecek, bu biraz zaman alabilir. Model yüklenirken de beklemek gerekiyor.

## Ne denedim?

### Hiyerarşik chunking
Metni iki seviyede parçaladım:
- **Parent chunks:** 512 token (geniş bağlam için)
- **Child chunks:** 256 token (detay için)

Parent-child ilişkisi sayesinde hem genel yapıyı hem de spesifik bilgiyi yakalayabiliyorum.

### Deneysel Parametre Çalışması
Farklı parametre kombinasyonlarını test ettim:
- Chunk boyutu: 128, 256, 512
- Overlap: 0, 25, 50, 100 token
- Temperature: 0.1, 0.2, 0.4, 0.6, 0.8

Toplamda 60 farklı konfigürasyon denedim ama bu çok uzun sürdü (saatler). Küçük subset ile başlamanız daha mantıklı.

## Sonuçlar

RAG kullanan sistem, baseline'a göre daha iyi sonuçlar verdi. Özellikle kitaptan direkt alıntı gereken sorularda fark açtı.

Bulduğum ilginç şeyler:
- Chunk boyutu büyüdükçe bağlam artıyor ama gürültü de artıyor
- 25-50 token overlap genelde yeterli oluyor
- Temperature'u 0.1-0.4 arasında tutmak daha tutarlı cevaplar veriyor

### Karşılaştığım sorunlar

1. **Colab timeout:** Uzun parametre denemeleri sırasında bağlantı kopmalar oldu
2. **Memory:** Gemma modeli bile Colab'ın ücretsiz versiyonunda biraz zorladı
3. **Retrieval kalitesi:** Bazen alakasız chunk'lar geliyor, reranking eklemek lazım

## İyileştirilebilecek şeyler

- Cross-encoder ile retrieval sonrası yeniden sıralama
- Query expansion (soruyu genişletip daha iyi arama)
- Daha büyük modeller denemek (7B, 13B)
- Fine-tuning yapılmış embedding modeli kullanmak

## Notlar

- NarrativeQA'da sadece 40 test sorusu var bu kitap için, büyük bir veri seti değil
- Colab Pro kullanıyorsanız daha rahat çalışırsınız
- Checkpoint kaydetmeyi unutmayın yoksa her şey başa döner

## Kaynaklar

Kullandığım veri setleri ve araçlar:
- [NarrativeQA](https://github.com/google-deepmind/narrativeqa) - Soru-cevap veri seti
- [Project Gutenberg](https://www.gutenberg.org/ebooks/1845) - Kitap metni
- [Milvus](https://milvus.io/) - Vektör veritabanı
- [Sentence Transformers](https://www.sbert.net/) - Embedding modelleri
