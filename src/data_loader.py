import requests
import pandas as pd
from pathlib import Path

class DataLoader:
    """
    Zuleika Dobson kitabini ve NarrativeQA sorularini indiren sinif
    """

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.book_url = "https://www.gutenberg.org/cache/epub/1845/pg1845.txt"
        self.qaps_url = "https://raw.githubusercontent.com/google-deepmind/narrativeqa/refs/heads/master/qaps.csv"
        self.document_id = "0cd690f600881ef37a4e36ca79e378c733636c30"

    def download_book(self):
        """
        Zuleika Dobson kitabini indir ve kaydet
        """
        print("Kitap indiriliyor...")
        response = requests.get(self.book_url)
        response.raise_for_status()

        book_path = self.data_dir / "zuleika_dobson.txt"
        with open(book_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"Kitap kaydedildi: {book_path}")
        return book_path

    def download_and_filter_questions(self):
        """
        NarrativeQA sorularini indir ve Zuleika Dobson icin filtrele
        Train, test ve valid setlerini ayri kaydet
        """
        print("Sorular indiriliyor...")
        df = pd.read_csv(self.qaps_url)

        zuleika_df = df[df['document_id'] == self.document_id].copy()

        print(f"\nToplam {len(zuleika_df)} soru bulundu")
        print(f"Test: {len(zuleika_df[zuleika_df['set'] == 'test'])}")

        test_df = zuleika_df[zuleika_df['set'] == 'test']
        test_path = self.data_dir / "questions_test.csv"
        test_df.to_csv(test_path, index=False)


        print(f"\nSorular kaydedildi:")
        print(f"  Test: {test_path}")


        return test_path

    def load_all_data(self):
        """
        Tum verileri indir ve yollarini dondur
        """
        book_path = self.download_book()
        train_path, test_path, valid_path = self.download_and_filter_questions()

        return {
            'book': book_path,
            'train': train_path,
            'test': test_path,
            'valid': valid_path
        }

if __name__ == "__main__":
    loader = DataLoader()
    paths = loader.load_all_data()
    print("\nVeri indirme tamamlandi!")
