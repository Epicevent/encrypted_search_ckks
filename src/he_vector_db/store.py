import os
import sqlite3
import tenseal as ts
from tenseal import CKKSVector
from sentence_transformers import SentenceTransformer
from cryptography.fernet import Fernet

class HEVectorStore:
    """
    동형암호화 벡터 스토어: CKKS 스킴 사용, SQLite에 암호문, 암호화된 ID 및 암호화된 원문 저장·로드 지원
    """
    def __init__(
        self,
        model_name='all-MiniLM-L6-v2',
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 60],
        db_path=None,
        id_key: bytes = None
    ):
        # 임베딩 모델 초기화
        self.model = SentenceTransformer(model_name)
        # ID/텍스트 암호화(Fernet) 설정
        self.fernet = Fernet(id_key) if id_key else None
        # HE 컨텍스트 설정
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40
        # 비밀키 보존 후 공개 모드 전환
        secret = self.context.serialize(save_secret_key=True)
        self.context.make_context_public()
        self.context = ts.context_from(secret)
        # 저장소 초기화
        self.enc_vectors = []
        self.ids = []           # 암호화된 ID 저장 (bytes)
        self.texts = []        # 암호화된 원문 저장 (bytes)
        # SQLite DB 설정 및 로드
        self.db_path = db_path
        if db_path:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self._init_db()
            self._load()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        cur = self.conn.cursor()
        # 테이블 생성: ID, ciphertext, text_enc 컬럼
        cur.execute(
            '''
            CREATE TABLE IF NOT EXISTS vectors (
                id BLOB PRIMARY KEY,
                ciphertext BLOB,
                text_enc BLOB
            )
            '''
        )
        self.conn.commit()

    def add(self, texts, ids=None):
        embeddings = self.model.encode(texts)
        cur = self.conn.cursor() if self.db_path else None
        for idx, (raw_text, vec) in enumerate(zip(texts, embeddings)):
            raw_id = ids[idx] if ids else str(len(self.ids))
            # ID 암호화 -> bytes
            enc_id = self.fernet.encrypt(raw_id.encode()) if self.fernet else raw_id.encode()
            # 원문 텍스트 암호화 -> bytes
            enc_text = self.fernet.encrypt(raw_text.encode()) if self.fernet else raw_text.encode()
            # 벡터 암호화
            enc_vec = ts.ckks_vector(self.context, vec.tolist())
            # 메모리 보관
            self.enc_vectors.append(enc_vec)
            self.ids.append(enc_id)
            self.texts.append(enc_text)
            # DB 저장
            if cur:
                data = enc_vec.serialize()
                cur.execute(
                    'REPLACE INTO vectors (id, ciphertext, text_enc) VALUES (?, ?, ?)',
                    (enc_id, data, enc_text)
                )
        if cur:
            self.conn.commit()

    def search(self, query, top_k=5):
        q_vec = self.model.encode([query])[0]
        enc_q = ts.ckks_vector(self.context, q_vec.tolist())
        sims = []
        # 메모리에 있는 암호화된 ID, 텍스트, 벡터 사용
        for enc_id, enc_text, enc_vec in zip(self.ids, self.texts, self.enc_vectors):
            score = enc_q.dot(enc_vec).decrypt()[0]
            sims.append((enc_id, enc_text, score))
        # top-k 반환
        return sorted(sims, key=lambda x: -x[2])[:top_k]

    def _load(self):
        cur = self.conn.cursor()
        cur.execute('SELECT id, ciphertext, text_enc FROM vectors')
        for enc_id, blob, enc_text in cur.fetchall():
            enc_vec = CKKSVector.load(self.context, blob)
            self.enc_vectors.append(enc_vec)
            self.ids.append(enc_id)
            self.texts.append(enc_text)

    def close(self):
        if self.db_path:
            self.conn.close()