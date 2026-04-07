# EICN RAG Prototype - 상세설계서 v2.0

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v1.0 | 2026-04-07 | 초기 설계 (RAG-Anything 엔진 교체 방식) |
| v2.0 | 2026-04-07 | 전면 재설계: 소스 내재화, 듀얼 임베딩, 프로젝트별 전략 선택 |

---

## 1. 프로젝트 개요

### 1.1 목적
rag-baseplate 프로젝트에 LightRAG/RAG-Anything의 지식그래프(KG) 기반 검색 및
멀티모달 분석 기능을 통합한다. 외부 패키지 의존 대신 소스를 내재화하여 자체 엔진으로 개발한다.

### 1.2 핵심 설계 결정

| 항목 | 결정 |
|------|------|
| 엔진 통합 방식 | lightrag, raganything 소스를 내재화 (패키지 설치 X) |
| 브랜딩 | lightrag, rag-anything 관련 명칭 모두 제거 |
| 폴더명 | `backend/engine/` (core, storage, multimodal) |
| 테이블 prefix | 모든 테이블에 `er_` prefix |
| 임베딩 전략 | 프로젝트별 3가지 선택: dual, splade, dense |
| 그래프 저장소 | Phase1: NetworkX/GraphML, Phase2: PostgreSQL AGE |
| 문서 파싱 | MinerU 2.0 (PDF), 기존 document_parser (비-PDF) |
| VLM/LLM | Qwen3-4B (LLM), Qwen2.5-Omni-7B (VLM), BAAI/bge-m3 (dense 임베딩) |

### 1.3 하드웨어 환경

```
GPU 0 (L40S, 48GB):
  ├─ vLLM: Qwen3-4B-Instruct     (Port 8766, ~10GB)
  └─ vLLM: BAAI/bge-m3 embedding  (Port 8001, ~3GB)

GPU 1 (L40S, 48GB):
  └─ vLLM: Qwen2.5-Omni-7B VLM   (Port 8767, ~18GB)

CPU:
  └─ SPLADE-ko-v1                 (PyTorch, ~1GB)
```

### 1.4 유지 항목 (변경 없음)
- 프론트엔드 전체 (React + TypeScript)
- SSE 스트리밍 방식 및 이벤트 형식
- 서비스 관리 스크립트 (bin/)
- 로깅 인프라


---

## 2. 소스 구조

```
rag-baseplate/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI 앱
│   │   ├── config.py                  # 설정 (프로젝트별 임베딩 전략 포함)
│   │   ├── database.py                # PostgreSQL 연결
│   │   ├── routers/
│   │   │   ├── chat.py                # 듀얼 검색 + SSE 스트리밍
│   │   │   ├── ingest.py              # 병렬 적재 파이프라인
│   │   │   ├── projects.py            # 프로젝트 CRUD
│   │   │   ├── chunks.py              # 청크 상세 조회
│   │   │   └── questions.py           # 예제 질문
│   │   ├── services/
│   │   │   ├── llm.py                 # LLM 서비스 (기존 유지)
│   │   │   ├── embedding.py           # SPLADE 임베딩 (기존 유지)
│   │   │   ├── retrieval.py           # SPLADE 검색 (기존 유지)
│   │   │   ├── graph_engine.py        # 신규: KG 엔진 래퍼
│   │   │   └── image_service.py       # 신규: 이미지 추출/저장/매핑
│   │   └── models/
│   │       └── schemas.py             # Pydantic 모델
│   │
│   ├── engine/                        # 신규: 내재화된 엔진 소스
│   │   ├── core/                      # (구 lightrag 코어)
│   │   │   ├── __init__.py
│   │   │   ├── engine.py              # GraphRAGEngine (구 LightRAG)
│   │   │   ├── base.py                # BaseVectorStorage 등 인터페이스
│   │   │   ├── operate.py             # 엔티티/관계 추출 로직
│   │   │   ├── utils.py               # EmbeddingFunc, 해시, 토크나이저
│   │   │   └── prompt.py              # KG 추출용 프롬프트
│   │   │
│   │   ├── storage/                   # (구 lightrag/kg)
│   │   │   ├── __init__.py
│   │   │   ├── pg_vector.py           # PGVectorStorage (dense + sparse 지원)
│   │   │   ├── pg_kv.py               # PGKVStorage
│   │   │   ├── pg_graph.py            # PGGraphStorage (AGE, Phase2)
│   │   │   ├── pg_doc_status.py       # PGDocStatusStorage
│   │   │   ├── networkx_graph.py      # NetworkX 그래프 (Phase1 기본)
│   │   │   └── nano_vector.py         # NanoVectorDB (로컬 개발용)
│   │   │
│   │   └── multimodal/               # (구 raganything)
│   │       ├── __init__.py
│   │       ├── pipeline.py            # MultimodalPipeline (구 RAGAnything)
│   │       ├── config.py              # PipelineConfig (구 RAGAnythingConfig)
│   │       ├── parser.py              # MinerU 파서
│   │       ├── processor.py           # 문서 처리 (구 ProcessorMixin)
│   │       ├── query.py               # 쿼리 처리 (구 QueryMixin)
│   │       ├── modal_processors.py    # 이미지/테이블/수식 프로세서
│   │       ├── context_extractor.py   # 문맥 추출
│   │       ├── prompt.py              # 멀티모달 프롬프트 (한국어)
│   │       └── utils.py               # separate_content 등
│   │
│   ├── scripts/
│   │   ├── document_parser.py         # 비-PDF 파서 (기존 유지)
│   │   ├── chunker.py                 # 텍스트 청킹 (기존 유지)
│   │   └── init_db.sql                # er_ prefix 반영 DDL
│   │
│   └── data/
│       └── extracted_images/
│
├── frontend/                          # 변경 없음
└── bin/                               # 서비스 관리 스크립트
```

---

## 3. 프로젝트별 임베딩 전략

### 3.1 설정

er_projects 테이블의 embedding_strategy 컬럼으로 프로젝트별 전략을 결정한다.

### 3.2 전략별 동작 비교

```
┌───────────┬──────────────────┬──────────────────┬──────────────────┐
│           │ dual             │ splade           │ dense            │
├───────────┼──────────────────┼──────────────────┼──────────────────┤
│ 청크 검색 │ SPLADE           │ SPLADE           │ bge-m3           │
│           │ (sparsevec)      │ (sparsevec)      │ (GraphRAGEngine) │
├───────────┼──────────────────┼──────────────────┼──────────────────┤
│ KG 엔티티 │ bge-m3           │ SPLADE           │ bge-m3           │
│ /관계 벡터│ (vector 1024)    │ (sparsevec 51200)│ (vector 1024)    │
├───────────┼──────────────────┼──────────────────┼──────────────────┤
│ KG 구축   │ LLM 기반 (동일)  │ LLM 기반 (동일)  │ LLM 기반 (동일)  │
├───────────┼──────────────────┼──────────────────┼──────────────────┤
│ KG 검색   │ bge-m3 +         │ SPLADE +         │ bge-m3 +         │
│           │ 그래프탐색        │ 그래프탐색       │ 그래프탐색       │
├───────────┼──────────────────┼──────────────────┼──────────────────┤
│ er_vec_*  │ vector(1024)     │ sparsevec(51200) │ vector(1024)     │
│ 컬럼 타입 │                  │                  │                  │
├───────────┼──────────────────┼──────────────────┼──────────────────┤
│ 필요 모델 │ SPLADE + bge-m3  │ SPLADE만         │ bge-m3만         │
├───────────┼──────────────────┼──────────────────┼──────────────────┤
│ 강점      │ 최고 검색 품질   │ 한국어 최적,     │ 그래프 검색 최적,│
│           │ 양쪽 장점 모두   │ GPU 부담 최소    │ 복합 추론 강점   │
├───────────┼──────────────────┼──────────────────┼──────────────────┤
│ 엔진 수정 │ 불필요           │ PGVectorStorage  │ 불필요           │
│           │ (기본 dense)     │ sparse 지원 필요 │ (기본 동작)      │
└───────────┴──────────────────┴──────────────────┴──────────────────┘
```


---

## 4. DB 스키마

### 4.1 기존 테이블 (er_ prefix로 rename)

```sql
-- 프로젝트
CREATE TABLE er_projects (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    company_code    VARCHAR(20) DEFAULT 'eicn',
    embedding_strategy VARCHAR(10) DEFAULT 'dual'
                    CHECK (embedding_strategy IN ('dual', 'splade', 'dense')),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 문서
CREATE TABLE er_documents (
    id              SERIAL PRIMARY KEY,
    project_id      INTEGER REFERENCES er_projects(id) ON DELETE SET NULL,
    filename        VARCHAR(255) NOT NULL,
    original_name   VARCHAR(255) NOT NULL,
    file_size       INTEGER NOT NULL,
    page_count      INTEGER NOT NULL,
    status          VARCHAR(20) DEFAULT 'processing',
    document_hint   TEXT DEFAULT '',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 청크
CREATE TABLE er_chunks (
    id              SERIAL PRIMARY KEY,
    document_id     INTEGER REFERENCES er_documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    page_number     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    content_full    TEXT,
    embedding       sparsevec(51200),       -- SPLADE (strategy: dual, splade)
    engine_chunk_id VARCHAR(64),            -- GraphRAGEngine 청크 매핑
    token_count     INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_chunk UNIQUE (document_id, chunk_index)
);

-- 이미지 자산
CREATE TABLE er_image_assets (
    id              SERIAL PRIMARY KEY,
    document_id     INTEGER REFERENCES er_documents(id) ON DELETE CASCADE,
    page_number     INTEGER NOT NULL,
    image_type      VARCHAR(30) NOT NULL DEFAULT 'extracted',
    image_path      TEXT NOT NULL,
    caption         TEXT,
    width           INTEGER,
    height          INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 청크-이미지 매핑
CREATE TABLE er_chunk_images (
    id              SERIAL PRIMARY KEY,
    chunk_id        INTEGER NOT NULL REFERENCES er_chunks(id) ON DELETE CASCADE,
    image_asset_id  INTEGER NOT NULL REFERENCES er_image_assets(id) ON DELETE CASCADE,
    CONSTRAINT unique_chunk_image UNIQUE (chunk_id, image_asset_id)
);

-- 예제 질문
CREATE TABLE er_questions (
    id              SERIAL PRIMARY KEY,
    document_id     INTEGER REFERENCES er_documents(id) ON DELETE CASCADE,
    question        TEXT NOT NULL
);

-- 인덱스
CREATE INDEX idx_er_documents_project ON er_documents(project_id);
CREATE INDEX idx_er_chunks_document ON er_chunks(document_id);
CREATE INDEX idx_er_chunks_page ON er_chunks(page_number);
CREATE INDEX idx_er_image_assets_doc_page ON er_image_assets(document_id, page_number);
CREATE INDEX idx_er_chunk_images_chunk ON er_chunk_images(chunk_id);
CREATE INDEX idx_er_chunk_images_image ON er_chunk_images(image_asset_id);
CREATE INDEX idx_er_questions_document ON er_questions(document_id);
```

### 4.2 엔진 테이블 (er_ prefix, 프로젝트별 workspace)

workspace 형식: `{company_code}_{project_id}` (예: `eicn_1`, `eicn_2`)

```sql
-- 벡터 테이블 (strategy에 따라 컬럼 타입 결정)
-- strategy=dual|dense → vector(1024)
-- strategy=splade → sparsevec(51200)

CREATE TABLE er_vec_chunks_{workspace} (
    id              VARCHAR(64) PRIMARY KEY,
    content         TEXT,
    content_vector  vector(1024),           -- 또는 sparsevec(51200)
    metadata        JSONB,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE er_vec_entities_{workspace} (
    id              VARCHAR(64) PRIMARY KEY,
    entity_name     TEXT,
    entity_type     TEXT,
    content         TEXT,
    content_vector  vector(1024),           -- 또는 sparsevec(51200)
    source_id       TEXT,
    file_path       TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE er_vec_relations_{workspace} (
    id              VARCHAR(64) PRIMARY KEY,
    src_id          TEXT,
    tgt_id          TEXT,
    content         TEXT,
    content_vector  vector(1024),           -- 또는 sparsevec(51200)
    keywords        TEXT,
    description     TEXT,
    weight          FLOAT,
    file_path       TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- KV 저장소
CREATE TABLE er_kv_store_{workspace} (
    id              VARCHAR(256) PRIMARY KEY,
    data            JSONB,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 문서 상태
CREATE TABLE er_doc_status_{workspace} (
    id              VARCHAR(256) PRIMARY KEY,
    status          TEXT,
    metadata        JSONB,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4.3 엔진 테이블 스키마 변경 가능성에 대한 참고사항

> **주의:** 엔진 테이블(er_vec_*, er_kv_store_*, er_doc_status_*)의 스키마는
> 개발 과정에서 변경될 수 있습니다.
>
> LightRAG 원본 소스를 내재화하면서 다음 사유로 스키마가 수정될 수 있습니다:
>
> 1. **컬럼 추가/변경**: 엔티티/관계에 한국어 최적화 메타데이터 추가,
>    SPLADE sparse 지원을 위한 컬럼 타입 분기 등
> 2. **인덱스 변경**: HNSW/IVFFlat 인덱스 파라미터 튜닝,
>    sparsevec용 인덱스 전략 변경
> 3. **테이블 분할/병합**: 성능 최적화를 위한 구조 변경
> 4. **JSONB 구조 변경**: KV 저장소의 내부 데이터 형식 변경
> 5. **LightRAG 업스트림 반영**: 원본 LightRAG의 스키마 변경사항 중
>    유용한 것을 선택적으로 반영
>
> 따라서 개발 초기에는 **마이그레이션 스크립트**를 준비하고,
> 엔진 테이블 DDL을 `engine/storage/` 내에서 코드로 관리하여
> 스키마 변경 시 자동 마이그레이션이 가능하도록 합니다.
>
> 기존 애플리케이션 테이블(er_projects, er_documents, er_chunks 등)의
> 스키마는 안정적으로 유지합니다.

### 4.4 데이터 저장소 구성

| 데이터 | 저장소 | 테이블/경로 |
|--------|--------|------------|
| 프로젝트/문서 메타데이터 | PostgreSQL | er_projects, er_documents |
| 청크 텍스트 + SPLADE 임베딩 | PostgreSQL | er_chunks |
| 이미지 메타데이터 | PostgreSQL | er_image_assets |
| 청크-이미지 매핑 | PostgreSQL | er_chunk_images |
| Dense 벡터 (bge-m3) | PostgreSQL pgvector | er_vec_*_{workspace} |
| Sparse 벡터 (SPLADE) | PostgreSQL pgvector | er_vec_*_{workspace} (sparsevec) |
| KV 저장소 | PostgreSQL | er_kv_store_{workspace} |
| 지식그래프 (Phase1) | 파일시스템 | {working_dir}/graph.graphml |
| 지식그래프 (Phase2) | PostgreSQL AGE | er_graph_{workspace} |
| 이미지 파일 | 파일시스템 | /assets/extracted_images/doc_{id}/ |


---

## 5. 적재 파이프라인 (Ingest)

```
POST /api/ingest (파일 업로드)
  │
  ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1단계: 파싱                                [engine/multimodal/parser.py]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  │
  │  PDF → MinerU 2.0 (subprocess)
  │    → content_list.json 출력
  │    → 추출 이미지 파일들 (img_path)
  │
  │  비-PDF (DOCX, PPTX, XLSX, HWP)
  │    → scripts/document_parser.py (기존 코드)
  │    → content_list 호환 형식으로 변환
  │
  ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2단계: 콘텐츠 분리                         [engine/multimodal/utils.py]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  │
  │  separate_content(content_list)
  │    ├─→ text_content: 텍스트 항목들 결합 (순서 보존)
  │    └─→ multimodal_items: 이미지/테이블/수식 리스트
  │          각 항목: {type, img_path, page_idx, caption, ...}
  │
  ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3단계: 멀티모달 분석 + 이미지 저장 (병렬)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  │
  │  await asyncio.gather(
  │
  │    ┌─ Task A: VLM/LLM 분석 ─────[engine/multimodal/modal_processors.py]
  │    │  이미지 → ImageModalProcessor
  │    │    → base64 인코딩 → Qwen2.5-Omni VLM
  │    │    → 엔티티 추출 + 상세 설명(캡션) 생성
  │    │  테이블 → TableModalProcessor
  │    │    → table_body + caption → Qwen3-4B LLM
  │    │    → 구조 분석 + 핵심 수치 추출
  │    │  수식 → EquationModalProcessor
  │    │    → LaTeX → Qwen3-4B LLM → 의미 해석
  │    │  출력: enhanced_descriptions[]
  │    └─────────────────────────────────────────────────
  │
  │    ┌─ Task B: 이미지 파일 저장 ──[app/services/image_service.py]
  │    │  MinerU 추출 이미지
  │    │    → /assets/extracted_images/doc_{document_id}/ 복사
  │    │  er_image_assets 테이블에 메타데이터 삽입
  │    │  출력: image_asset_ids[]
  │    └─────────────────────────────────────────────────
  │
  │  )
  │
  ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4단계: 캡션 합류 + 텍스트 청킹             [scripts/chunker.py]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  │
  │  enhanced_descriptions → 캡션 청크로 변환
  │    형식: "[시각자료 요약][페이지 N]\n{캡션 텍스트}"
  │
  │  text_content + 캡션 청크 → chunker.py
  │    ├─ 400자 청크 / 120자 오버랩
  │    ├─ [시각자료 요약] 블록 보호 범위
  │    ├─ 문장/단락 경계에서 분리
  │    └─ 페이지 번호 추적
  │
  │  결과: chunks[] (chunk_index, page_number, content, content_full)
  │
  ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5단계: 임베딩 + KG 구축 (★ 전략별 병렬 실행)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  │
  │  strategy = er_projects.embedding_strategy
  │  workspace = f"{company_code}_{project_id}"
  │
  │  ═══ strategy == "dual" ═══════════════════════════
  │  │
  │  │  await asyncio.gather(
  │  │
  │  │    ┌─ SPLADE 경로 ──────────[app/services/embedding.py]
  │  │    │  각 청크 → SPLADE-ko-v1 모델
  │  │    │    → {token_id: weight} sparse 벡터
  │  │    │    → er_chunks.embedding (sparsevec 51200)
  │  │    └─────────────────────────────────────────────
  │  │
  │  │    ┌─ Dense + KG 경로 ──────[app/services/graph_engine.py]
  │  │    │                        [engine/core/engine.py]
  │  │    │  전체 텍스트 → GraphRAGEngine.ainsert()
  │  │    │    ├─ bge-m3 임베딩 (dense, 1024차원)
  │  │    │    ├─ LLM 엔티티/관계 추출 (Qwen3-4B)
  │  │    │    ├─ → er_vec_chunks_{ws}    (vector 1024)
  │  │    │    ├─ → er_vec_entities_{ws}  (vector 1024)
  │  │    │    ├─ → er_vec_relations_{ws} (vector 1024)
  │  │    │    ├─ → er_kv_store_{ws}      (청크 텍스트)
  │  │    │    └─ → 지식그래프 (NetworkX GraphML)
  │  │    └─────────────────────────────────────────────
  │  │
  │  │  )
  │  │
  │  ═══ strategy == "splade" ═════════════════════════
  │  │
  │  │  await asyncio.gather(
  │  │
  │  │    ┌─ SPLADE 청크 임베딩 ───[app/services/embedding.py]
  │  │    │  (위 SPLADE 경로와 동일)
  │  │    └─────────────────────────────────────────────
  │  │
  │  │    ┌─ SPLADE + KG 경로 ─────[app/services/graph_engine.py]
  │  │    │                        [engine/core/engine.py (fork 수정)]
  │  │    │                        [engine/storage/pg_vector.py (fork 수정)]
  │  │    │  전체 텍스트 → GraphRAGEngine.ainsert()
  │  │    │    ├─ SPLADE 임베딩 (sparse, 51200차원)
  │  │    │    ├─ LLM 엔티티/관계 추출 (Qwen3-4B) ← 동일
  │  │    │    ├─ → er_vec_chunks_{ws}    (sparsevec 51200)
  │  │    │    ├─ → er_vec_entities_{ws}  (sparsevec 51200)
  │  │    │    ├─ → er_vec_relations_{ws} (sparsevec 51200)
  │  │    │    ├─ → er_kv_store_{ws}      (청크 텍스트)
  │  │    │    └─ → 지식그래프 (NetworkX GraphML)
  │  │    └─────────────────────────────────────────────
  │  │
  │  │  )
  │  │
  │  ═══ strategy == "dense" ══════════════════════════
  │  │
  │  │  await task_dense_kg(text_content, chunks)
  │  │    (위 Dense + KG 경로와 동일)
  │  │    ※ er_chunks.embedding 컬럼 미사용
  │  │
  │  ══════════════════════════════════════════════════
  │
  ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6단계: 청크-이미지 매핑 + 완료             [app/routers/ingest.py]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  │
  │  content_list의 page_idx + 출현 순서 기반 매핑:
  │    같은 페이지의 이미지 → 인접 텍스트 청크에 매핑
  │    → er_chunk_images 테이블에 저장
  │
  │  er_documents.status = 'completed'
  │
  ▼ 적재 완료
```


---

## 6. 쿼리 파이프라인 (Chat)

```
POST /api/chat (사용자 질문)
  │
  │  project = er_projects 조회
  │  strategy = project.embedding_strategy
  │  workspace = f"{project.company_code}_{project.id}"
  │
  ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1단계: 전략별 병렬 검색
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  │
  │  ═══ strategy == "dual" ═══════════════════════════
  │  │
  │  │  await asyncio.gather(
  │  │
  │  │    ┌─ SPLADE 검색 ─────────[app/services/retrieval.py]
  │  │    │  질문 → SPLADE 임베딩
  │  │    │  → er_chunks에서 sparsevec 유사도 검색
  │  │    │  → 어휘 폴백 (ILIKE 패턴 매칭)
  │  │    │  → 리랭킹 (어휘 겹침, 수치 일관성, 페이지 힌트)
  │  │    │  → top_k 청크 + relevance_score
  │  │    └───────────────────────────────────────────
  │  │
  │  │    ┌─ 그래프 검색 ─────────[app/services/graph_engine.py]
  │  │    │  질문 → bge-m3 임베딩
  │  │    │  → er_vec_entities_{ws} 유사도 검색 (진입점)
  │  │    │  → 지식그래프 탐색 (엔티티→관계→엔티티)
  │  │    │  → 관련 엔티티/관계 컨텍스트
  │  │    └───────────────────────────────────────────
  │  │
  │  │  )
  │  │
  │  ═══ strategy == "splade" ═════════════════════════
  │  │
  │  │  await asyncio.gather(
  │  │
  │  │    ┌─ SPLADE 청크 검색 ────(위와 동일)
  │  │    └───────────────────────────────────────────
  │  │
  │  │    ┌─ SPLADE 그래프 검색 ──[app/services/graph_engine.py]
  │  │    │  질문 → SPLADE 임베딩
  │  │    │  → er_vec_entities_{ws} (sparsevec) 유사도 검색
  │  │    │  → 지식그래프 탐색
  │  │    │  → 관련 엔티티/관계 컨텍스트
  │  │    └───────────────────────────────────────────
  │  │
  │  │  )
  │  │
  │  ═══ strategy == "dense" ══════════════════════════
  │  │
  │  │  GraphRAGEngine.aquery(question, mode="hybrid")
  │  │    질문 → bge-m3 임베딩
  │  │    → er_vec_chunks_{ws} + er_vec_entities_{ws} 검색
  │  │    → 지식그래프 탐색
  │  │    → top_k 청크 + 엔티티/관계 컨텍스트
  │  │
  │  ══════════════════════════════════════════════════
  │
  ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2단계: 결과 병합 + 이미지 조회
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  │
  │  SPLADE 청크 + 그래프 컨텍스트
  │    → 중복 제거 (같은 청크는 높은 점수 채택)
  │    → 리랭킹
  │
  │  소스 포맷 (프론트엔드 호환):
  │    [{chunk_id, page_number, content_preview, relevance_score}, ...]
  │
  │  소스 청크 → er_chunk_images → er_image_assets → 관련 이미지
  │    [{image_id, page_number, image_url, caption, ...}, ...]
  │
  ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3단계: LLM 컨텍스트 구성                   [app/services/llm.py]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  │
  │  컨텍스트 = SPLADE 청크 텍스트 + 그래프 보강 컨텍스트
  │  시스템 프롬프트 = 프로젝트별 프롬프트 (기존 유지)
  │  히스토리 = 대화 이력 (토큰 제한 내 trimming)
  │
  ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4단계: SSE 스트리밍 응답                   [app/routers/chat.py]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  │
  │  yield ":\n\n"                              (SSE flush)
  │  yield {type: "sources", data: source_list}  (검색된 청크)
  │  yield {type: "images",  data: image_list}   (관련 이미지)
  │  yield {type: "settings", data: settings}    (검색 설정)
  │  yield {type: "token",   data: chunk}        (LLM 토큰 스트리밍)
  │  yield {type: "done",    data: {...}}        (완료 + 응답시간)
  │
  ▼ 프론트엔드 표시 (변경 없음)
```


---

## 7. 엔진 소스 수정 상세

### 7.1 리네이밍 매핑

| 원본 (lightrag/raganything) | 내재화 후 | 비고 |
|----------------------------|----------|------|
| `lightrag.LightRAG` | `engine.core.engine.GraphRAGEngine` | 메인 클래스 |
| `lightrag.base.BaseVectorStorage` | `engine.core.base.BaseVectorStorage` | 인터페이스 |
| `lightrag.utils.EmbeddingFunc` | `engine.core.utils.EmbeddingFunc` | sparse 모드 추가 |
| `lightrag.operate` | `engine.core.operate` | 엔티티/관계 추출 |
| `lightrag.kg.postgres_impl` | `engine.storage.pg_vector` 등 | 파일 분리 |
| `lightrag.kg.nano_vector_db_impl` | `engine.storage.nano_vector` | 로컬 개발용 |
| `lightrag.kg.networkx_impl` | `engine.storage.networkx_graph` | 그래프 |
| `raganything.RAGAnything` | `engine.multimodal.pipeline.MultimodalPipeline` | 멀티모달 |
| `raganything.RAGAnythingConfig` | `engine.multimodal.config.PipelineConfig` | 설정 |
| `raganything.modalprocessors` | `engine.multimodal.modal_processors` | 프로세서 |
| 모든 `LIGHTRAG_` DB prefix | `ER_` 또는 `er_` | 테이블 prefix |

### 7.2 PGVectorStorage sparse 지원 수정 (핵심)

```python
# engine/storage/pg_vector.py (수정)

class PGVectorStorage(BaseVectorStorage):
    """PostgreSQL pgvector 벡터 저장소 (dense + sparse 지원)"""

    def __init__(self, embedding_mode="dense", **kwargs):
        super().__init__(**kwargs)
        self.embedding_mode = embedding_mode  # "dense" 또는 "sparse"

    async def _create_table(self, workspace: str):
        if self.embedding_mode == "sparse":
            vector_type = f"sparsevec({self.sparse_dim})"  # 51200
        else:
            vector_type = f"vector({self.embedding_func.embedding_dim})"  # 1024

        await self.db.execute(f"""
            CREATE TABLE IF NOT EXISTS er_vec_{self.table_suffix}_{workspace} (
                id VARCHAR(64) PRIMARY KEY,
                content TEXT,
                content_vector {vector_type},
                ...
            )
        """)

    async def upsert(self, data: dict):
        if self.embedding_mode == "sparse":
            # SPLADE: {token_id: weight} → pgvector sparsevec format
            for key, item in data.items():
                sparse_vec = await self.embedding_func(item["content"])
                pgvec_str = self._sparse_to_pgvector(sparse_vec)
                await self.db.execute(
                    "INSERT INTO ... VALUES (%s, %s, %s::sparsevec, ...)",
                    (key, item["content"], pgvec_str)
                )
        else:
            # Dense: 기존 로직 유지 (numpy array → vector)
            ...

    async def query(self, query: str, top_k: int):
        if self.embedding_mode == "sparse":
            sparse_vec = await self.embedding_func(query)
            pgvec_str = self._sparse_to_pgvector(sparse_vec)
            # sparsevec 내적 연산자 사용
            results = await self.db.fetch(f"""
                SELECT *, (content_vector <#> %s::sparsevec) * -1 AS score
                FROM er_vec_{self.table_suffix}_{self.workspace}
                ORDER BY content_vector <#> %s::sparsevec
                LIMIT %s
            """, pgvec_str, pgvec_str, top_k)
        else:
            # Dense: 기존 cosine similarity
            ...

    def _sparse_to_pgvector(self, sparse_dict: dict) -> str:
        """SPLADE sparse dict → pgvector sparsevec format"""
        entries = [f"{idx}:{weight:.6f}"
                   for idx, weight in sorted(sparse_dict.items())]
        return "{" + ",".join(entries) + "}/51200"
```

### 7.3 EmbeddingFunc sparse 모드 추가

```python
# engine/core/utils.py (수정)

@dataclass
class EmbeddingFunc:
    embedding_dim: int              # dense: 1024, sparse: 51200
    func: callable
    max_token_size: int = None
    embedding_mode: str = "dense"   # 추가: "dense" 또는 "sparse"

    async def __call__(self, texts, **kwargs):
        result = await self.func(texts, **kwargs)

        if self.embedding_mode == "sparse":
            # sparse: Dict[int, float] 또는 List[Dict[int, float]] 반환
            # 차원 검증 스킵 (가변 길이)
            return result
        else:
            # dense: 기존 numpy 검증 로직 유지
            ...
```

### 7.4 테이블 prefix 변경

모든 storage 모듈에서 테이블명 prefix를 교체:
- `LIGHTRAG_VDB_` → `er_vec_`
- `LIGHTRAG_KV_` → `er_kv_store_`
- `LIGHTRAG_DOC_STATUS_` → `er_doc_status_`
- `LIGHTRAG_GRAPH_` → `er_graph_`

### 7.5 엔진 테이블 스키마 변경 관리

개발 과정에서 엔진 테이블 스키마가 변경될 수 있으므로:

1. **DDL을 코드로 관리**: `engine/storage/` 내 각 모듈에서 테이블 생성 SQL을 관리
2. **버전 추적**: `er_schema_version` 테이블 또는 마이그레이션 파일로 스키마 버전 관리
3. **자동 마이그레이션**: 엔진 초기화 시 현재 스키마 버전과 코드 버전을 비교하여 자동 마이그레이션
4. **변경 가능 항목**:
   - 컬럼 추가/변경 (한국어 메타데이터, 인덱스 힌트 등)
   - 인덱스 전략 변경 (HNSW 파라미터, sparsevec 인덱스 등)
   - JSONB 내부 구조 변경
   - 테이블 분할/병합 (성능 최적화)
5. **안정 테이블**: er_projects, er_documents, er_chunks 등 앱 테이블은 스키마 안정 유지


---

## 8. 마이그레이션 및 개발 단계

### Phase 1: 엔진 내재화 + 기본 동작 (1주)
1. lightrag 소스를 `engine/core/`에 복사 + 리네이밍
2. raganything 소스를 `engine/multimodal/`에 복사 + 리네이밍
3. 모든 import 경로 수정 (lightrag.* → engine.core.*, raganything.* → engine.multimodal.*)
4. 테이블 prefix `er_` 적용
5. 기존 dense 임베딩(bge-m3)으로 엔진 동작 확인

### Phase 2: SPLADE sparse 지원 (3일)
1. EmbeddingFunc에 sparse 모드 추가
2. PGVectorStorage에 sparsevec DDL/INSERT/SELECT 추가
3. strategy="splade" 프로젝트에서 KG 구축+검색 테스트

### Phase 3: 적재 파이프라인 통합 (3일)
1. ingest.py에 MinerU 파서 연동
2. 멀티모달 분석 (VLM/LLM) 연동
3. 이미지 저장 + 청크-이미지 매핑
4. 듀얼 임베딩 병렬 실행

### Phase 4: 쿼리 파이프라인 통합 (3일)
1. chat.py에 듀얼 검색 연동
2. 그래프 검색 결과 병합
3. SSE 스트리밍 응답 (프론트엔드 호환 확인)

### Phase 5: 검증 + 최적화 (3일)
1. 데모 데이터 재적재 (6개 프로젝트)
2. 3가지 전략별 검색 품질 비교
3. 성능 벤치마크 (적재 시간, 쿼리 응답 시간)
4. 프론트엔드 E2E 테스트

---

## 9. 리스크 및 대응

| 리스크 | 영향도 | 대응 방안 |
|--------|--------|----------|
| SPLADE로 KG 진입점 검색 품질 미검증 | 높음 | Phase2에서 조기 테스트, 품질 저하 시 dual 전략 권장 |
| MinerU 파서가 비-PDF 미지원 | 중간 | 기존 document_parser.py 유지 (DOCX/PPTX/XLSX 처리) |
| 엔진 소스 리네이밍 중 import 누락 | 중간 | 단위 테스트로 모든 import 경로 검증 |
| PostgreSQL AGE 설치/설정 복잡도 | 낮음 | Phase1에서 NetworkX 사용, AGE는 Phase2로 연기 |
| LightRAG 업스트림 업데이트 반영 어려움 | 낮음 | 내재화 시점의 버전 기록, 필요 시 수동 패치 |
| 엔진 테이블 스키마 변경으로 인한 데이터 손실 | 중간 | 마이그레이션 스크립트 준비, 개발 중 워크스페이스 재생성 허용 |

---

## 10. 검증 계획

### 기능 테스트
- [ ] 문서 업로드: PDF, DOCX, PPTX, XLSX
- [ ] MinerU 파싱 + content_list 생성
- [ ] 멀티모달 분석 (이미지 캡션, 테이블 분석, 수식 해석)
- [ ] 이미지 추출 + 프론트엔드 표시
- [ ] 3가지 전략별 적재: dual, splade, dense
- [ ] 3가지 전략별 쿼리: dual, splade, dense
- [ ] SSE 스트리밍 응답 (sources, images, tokens, done)
- [ ] 프로젝트 생성/삭제 (전략 설정 포함)
- [ ] 문서 삭제 시 엔진 데이터 정리
- [ ] 대화 히스토리 유지

### 품질 테스트
- [ ] 동일 질문에 대한 3가지 전략별 응답 비교
- [ ] 수치/표 질문 정확도 (SPLADE 강점 확인)
- [ ] 복합 추론 질문 (그래프 검색 강점 확인)
- [ ] 한국어 검색 품질 (SPLADE vs bge-m3)

### 성능 테스트
- [ ] 문서 적재 시간 (전략별 비교)
- [ ] 쿼리 응답 시간 - 첫 토큰까지 (전략별 비교)
- [ ] 듀얼 모드 병렬 실행 효과 측정

---

## 11. 의존성

### 유지 패키지
```
fastapi>=0.128.0
uvicorn>=0.27.0
psycopg2-binary>=2.9.0
pydantic-settings>=2.0.0
openai>=1.0.0
python-dotenv>=1.0.0
Pillow>=10.0.0
python-docx>=1.0.0
python-pptx>=0.6.0
openpyxl>=3.1.0
torch
transformers
```

### 신규 패키지 (engine용)
```
networkx                    # 그래프 저장소 (Phase1)
nano-vectordb               # 로컬 벡터 저장소 (개발용)
tiktoken                    # 토큰 카운팅
numpy                       # 벡터 연산
```

### 제거 패키지
```
# lightrag-hku              # 내재화로 대체
# raganything                # 내재화로 대체
```

---

*문서 버전: v2.0*
*작성일: 2026-04-07*
*프로젝트: EICN RAG Prototype*
