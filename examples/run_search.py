from he_vector_db.store import HEVectorStore
from cryptography.fernet import Fernet

if __name__ == "__main__":
    # ID 암호화 키 생성 (유저 단에서 관리)
    id_key = Fernet.generate_key()
    f = Fernet(id_key)

    # SQLite DB 사용 예
    store = HEVectorStore(
        model_name='all-MiniLM-L6-v2',
        db_path='./data/vectors.db',
        id_key=id_key
    )
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming industries.",
        "Climate change poses a significant threat to global biodiversity."
    ]
    raw_ids = ['fox', 'ai', 'climate']
    store.add(texts, ids=raw_ids)
    results = store.search("What impacts climate change?", top_k=3)

    # 암호화된 ID(BLOB) 복호화
    
    # search() 결과: [(enc_id, enc_text, score), ...]
    results = store.search("What impacts climate change?", top_k=3)

    # 복호화
    decoded = [
        (
            f.decrypt(enc_id).decode(),      # ID
            f.decrypt(enc_text).decode(),    # 원문 문장
            score
        )
        for enc_id, enc_text, score in results
    ]

    print("Search Results:", decoded)

    store.close()