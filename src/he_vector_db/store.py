import os
import sqlite3
import uuid
import heapq
import tenseal as ts
from tenseal import CKKSVector
from cryptography.fernet import Fernet
import numpy as np
from tqdm import tqdm
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Tuple, Optional
import math


class HEVectorStore:
    def __init__(self,
                 context_path: str,
                 db_path :str,
                 id_key_path : str):
        if not context_path or not os.path.exists(context_path):
            raise ValueError(f"Valid context_path required, got: {context_path}")
        if not db_path or not os.path.exists(os.path.dirname(db_path)):
            raise ValueError(f"Valid db_path required, got: {db_path}")
        with open(context_path, "rb") as f:
            ctx_bytes = f.read()
        self.context = ts.context_from(ctx_bytes)
        
        # 2) 컨텍스트 검증  
        self._validate_context(expected_scale=self.context.global_scale)    
        
        # 3) Fernet 키 설정
        id_key = self.load_or_create_fernet_key(id_key_path) if id_key_path else None
        if id_key and not isinstance(id_key, bytes):
            raise TypeError(f"Invalid id_key type: {type(id_key)}; expected bytes")
        self.id_key_path = id_key_path
        self.id_key = id_key
        self.fernet = Fernet(self.id_key)
        
        # 4) DB 경로 설정
        if not db_path.endswith('.db'):
            # db_path가 폴더 경로일 경우 → 내부에 기본 파일명 붙이기
            self.db_path = os.path.join(db_path, "he_vector_store.db")
        else:
            # db_path가 정확한 파일 경로일 경우 → 그대로 사용
            self.db_path = db_path

        
        print(f"[INIT] HEVectorStore @ {self.db_path}")

        # ✅ DB 파일 경로의 상위 디렉터리 생성
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def load_or_create_fernet_key(self,key_path: str) -> bytes:
        """
        Load a Fernet symmetric key from `key_path`, or generate & save one if missing.
        Returns the raw key bytes.
        """
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(key_path), exist_ok=True)

        if os.path.exists(key_path):
            with open(key_path, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(key)

        return key

    def _validate_context(self, expected_scale: float):
        """
        Duck‐type 검증:
          - serialize 메서드가 있는지
          - global_scale 이 예상 값인지
          - secret key 포함 상태인지(serialize→context_from)
        """
        # 1) duck‐typing으로 serialize 메서드 확인
        if not hasattr(self.context, "serialize"):
            raise TypeError("Invalid context: missing serialize()")

        # 2) global_scale 검증/수정
        if getattr(self.context, "global_scale", None) != expected_scale:
            print(f"[validate] resetting scale "
                  f"{getattr(self.context, 'global_scale', None)} → {expected_scale}")
            self.context.global_scale = expected_scale

        # 3) secret key 포함 여부 확인
        try:
            # serialize with secret & reload
            raw = self.context.serialize(save_secret_key=True)
            ts.context_from(raw)
        except Exception as e:
            raise RuntimeError("Context does not include a valid secret key") from e

        print("[validate] context OK")

    def add(self, texts=None, ids=None, embeddings=None, documents=None):
        """
        텍스트 또는 주어진 embeddings/documents를 암호화하여 저장합니다.
        벡터 정규화, ID/text 암호화, DB 저장까지 포함.
        """


        # embeddings 가 없으면 모델로부터 생성
        raw_texts = documents if documents is not None else texts or []

        
        print(f"[ADD] will insert {len(embeddings)} vectors")

        cur = self.conn.cursor()
        before = self.count()
        print(f"[ADD] count before = {before}")

        for idx, vec in enumerate(embeddings):
            raw_id   = ids[idx] if ids else str(uuid.uuid4())
            raw_text = raw_texts[idx] if idx < len(raw_texts) else ""

            # Encrypt ID/text
            enc_id   = self.fernet.encrypt(raw_id.encode()) if self.fernet else raw_id.encode()
            enc_text = self.fernet.encrypt(raw_text.encode()) if self.fernet else raw_text.encode()

            # Normalize vector
            arr = np.array(vec, dtype=float)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr /= norm

            # Encrypt vector
            enc_vec = ts.ckks_vector(self.context, arr.tolist())
            blob    = enc_vec.serialize()
            # Write to DB
            cur.execute(
                'REPLACE INTO vectors (id, ciphertext, text_enc) VALUES (?, ?, ?)',
                (enc_id, blob, enc_text)
            )

        # Commit & final count
        self.conn.commit()
        after = self.count()
        print(f"[ADD] committed, count after = {after}")

    def _search_chunk(
        self,
        docs: List[Tuple[bytes, bytes, bytes]],
        enc_queries: List[Any],
        chunk_id: int
    ) -> List[List[Tuple[bytes, bytes, float]]]:
        """
        Process one chunk of documents with a tqdm progress bar.
        """
        num_q = len(enc_queries)
        partial = [[] for _ in range(num_q)]
        for enc_id, blob, enc_txt in tqdm(
            docs,
            desc=f"Chunk {chunk_id}",
            position=chunk_id,
            leave=False,
            unit="doc"
        ):
            enc_vec = CKKSVector.load(self.context, blob)
            for qi, enc_q in enumerate(enc_queries):
                raw = enc_q.dot(enc_vec).decrypt()[0]
                partial[qi].append((enc_id, enc_txt, raw))
            del enc_vec
        return partial
    

    def query(
        self,
        embeddings: List[List[float]],
        n_results: int = 5,
        max_workers: Optional[int] = None
    ) -> List[List[Tuple[bytes, bytes, float]]]:
        """
        Parallel batch HE search with per-chunk tqdm bars.
        """
        if not embeddings:
            return []

        # 1) normalize & encrypt queries
        enc_queries = []
        for vec in embeddings:
            arr = np.array(vec, dtype=float)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr /= norm
            enc_queries.append(ts.ckks_vector(self.context, arr.tolist()))

        # 2) load docs once
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur  = conn.cursor()
        cur.execute("SELECT id, ciphertext, text_enc FROM vectors")
        docs = cur.fetchall()
        conn.close()

        # 3) split into chunks
        workers   = max_workers or (os.cpu_count() or 4)
        chunk_sz  = math.ceil(len(docs) / workers)
        doc_chunks = [docs[i:i+chunk_sz] for i in range(0, len(docs), chunk_sz)]

        # 4) parallel execution with per-chunk bars
        all_scores = [[] for _ in embeddings]
        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = {
                exe.submit(self._search_chunk, chunk, enc_queries, idx): idx
                for idx, chunk in enumerate(doc_chunks)
            }
            for fut in as_completed(futures):
                part = fut.result()
                for qi, lst in enumerate(part):
                    all_scores[qi].extend(lst)

        # 5) Top-K per query
        results = []
        for scores in all_scores:
            topk = sorted(scores, key=lambda x: -x[2])[:n_results]
            results.append(topk)

        return results
    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
                id BLOB PRIMARY KEY,
                ciphertext BLOB NOT NULL,
                text_enc BLOB
            )
        ''')
        # Optional pragmas for performance
        cur.execute('PRAGMA journal_mode = WAL;')
        cur.execute('PRAGMA synchronous = NORMAL;')
        self.conn.commit()

    def count(self):
        """Return total number of stored vectors."""
        if self.conn is None:
            return 0

        cur = self.conn.cursor()
        try:
            cur.execute('SELECT COUNT(*) FROM vectors')
            row = cur.fetchone()
            return row[0] if row is not None else 0
        except sqlite3.OperationalError as e:
            # vectors 테이블이 아직 없으면 0으로 처리
            print(f"[count] Warning: {e}")
            return 0

    def get_all_ids(self):
        """Return all encrypted IDs from the store."""
        if self.conn is None:
            return []
        cur = self.conn.cursor()
        cur.execute('SELECT id FROM vectors')
        return [row[0] for row in cur.fetchall()]

    def close(self):
        """Close DB connection if open."""
        if self.conn:
            self.conn.close()

