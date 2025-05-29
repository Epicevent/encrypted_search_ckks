# ─────────────────────────────────────────
# 1) 베이스 이미지
FROM python:3.9-slim

# ─────────────────────────────────────────
# 2) 작업 디렉터리
WORKDIR /app

# ─────────────────────────────────────────
# 3) 전체 컨텍스트 복사 (.dockerignore 적용)
COPY . .

# ─────────────────────────────────────────
# 4) 가상환경 생성 & 패키지 설치
RUN python -m venv /opt/venv \
 && . /opt/venv/bin/activate \
 && pip install --upgrade pip \
 # (1) requirements 파일명 확인: 
 #    실제 리포지토리에 requirement.txt 라면 아래처럼,
 && pip install -r requirements.txt \
 # (2) 개발 모드로 설치
 && pip install -e .

# ─────────────────────────────────────────
# 5) 설치 확인 커맨드
CMD ["bash","-lc", ". /opt/venv/bin/activate && python -c \"import he_vector_db; print(he_vector_db.__version__)\""]

