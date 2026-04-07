# EICN RAG Prototype - 상세설계서 v1.0

## 1. 프로젝트 개요

### 1.1 목적
rag-baseplate 프로젝트의 프론트엔드/API 구조를 유지하면서, RAG 엔진을 RAG-Anything(LightRAG 기반)으로 교체하여 그래프 기반 검색 및 고도화된 멀티모달 처리 능력을 확보한다.

### 1.2 핵심 변경 사항
| 구분 | 현재 (rag-baseplate) | 변경 후 (EICN RAG) |
|------|---------------------|---------------------|
| RAG 엔진 | 직접 구현 (SPLADE + pgvector) | RAG-Anything (LightRAG) |
| 임베딩 | SPLADE-ko-v1 (sparse, 51200차원) | BAAI/bge-m3 (dense, 1024차원) |
| 벡터 저장소 | PostgreSQL pgvector (sparsevec) | LightRAG 내장 (nano-vectordb 또는 pgvector dense) |
| 지식 그래프 | 없음 | LightRAG 내장 (엔티티/관계 자동 추출) |
| 문서 파싱 | 직접 구현 (pdf_extractor + document_parser) | RAG-Anything 파서 (MinerU/PaddleOCR) |
| 청킹 | 직접 구현 (chunker.py) | LightRAG 내장 토큰 기반 청킹 |
| 멀티모달 | Qwen2.5-Omni 캡셔닝 (직접 호출) | RAG-Anything ModalProcessor (VLM 통합) |
| 검색 | SPLADE 유사도 + 어휘 폴백 + 리랭킹 | LightRAG 하이브리드 (벡터 + 그래프 + 키워드) |
| LLM | vLLM Qwen3-4B (유지) | vLLM Qwen3-4B (유지) |
| 이미지 서빙 | PostgreSQL image_assets + 정적 파일 | PostgreSQL image_assets 유지 (하이브리드) |

### 1.3 하드웨어 환경
- GPU: L40S x 2
- GPU 0: vLLM (Qwen3-4B-Instruct + bge-m3 임베딩)
- GPU 1: VLM (Qwen2.5-Omni-7B) + 기타
- OS: Linux
- DB: PostgreSQL (pgvector 확장)

### 1.4 유지 항목 (변경 없음)
- 프론트엔드 전체 (React + TypeScript)
- FastAPI 프레임워크 및 API 엔드포인트 구조
- SSE 스트리밍 방식
- 프로젝트/문서 관리 (projects, documents 테이블)
- 이미지 파일 저장 및 서빙 (/assets 마운트)
- 어드민 인증 체계
- 서비스 관리 스크립트 (bin/)
- 로깅 인프라


## 2. 시스템 아키텍처

### 2.1 전체 구조
```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React + TS)                     │
│              Port 5678 (변경 없음)                            │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ ChatUI   │ │SourceCard │ │ AdminPanel│ │ ImageLightbox │  │
│  └────┬─────┘ └───────────┘ └──────────┘ └───────────────┘  │
│       │ SSE / REST API (동일 인터페이스)                       │
├───────┼─────────────────────────────────────────────────────┤
│       ▼          Backend (FastAPI) Port 8765                 │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Routers (변경 최소화)                                    │ │
│  │  /api/chat  /api/ingest  /api/projects  /api/chunks     │ │
│  └────┬──────────┬──────────────────────────────────────---┘ │
│       │          │                                           │
│  ┌────▼────┐ ┌───▼──────────────────────────────────┐       │
│  │ LLM Svc │ │ RAGAnything Service (신규)            │       │
│  │ (유지)  │ │  ┌──────────┐  ┌──────────────────┐  │       │
│  │         │ │  │ 문서 파싱  │  │ 쿼리 (aquery)    │  │       │
│  │         │ │  │ (MinerU)  │  │ vector+graph     │  │       │
│  │         │ │  └──────────┘  └──────────────────┘  │       │
│  │         │ │  ┌──────────────────────────────────┐│       │
│  │         │ │  │ Modal Processors                 ││       │
│  │         │ │  │ Image / Table / Equation         ││       │
│  │         │ │  └──────────────────────────────────┘│       │
│  └─────────┘ └──────────────────────────────────────┘       │
│       │              │                                       │
│  ┌────▼────┐    ┌────▼────────────────────────┐              │
│  │  vLLM   │    │ LightRAG 저장소              │              │
│  │ Qwen3-4B│    │  - nano-vectordb (임베딩)   │              │
│  │ Port8766│    │  - 지식그래프 (엔티티/관계)  │              │
│  └─────────┘    │  - 문서 청크 저장           │              │
│                 └─────────────────────────────┘              │
│  ┌─────────┐    ┌─────────────────────────────┐              │
│  │ vLLM    │    │ PostgreSQL (메타데이터)      │              │
│  │ bge-m3  │    │  - projects, documents      │              │
│  │ Port8001│    │  - image_assets             │              │
│  └─────────┘    │  - chunk_images, questions  │              │
│                 └─────────────────────────────┘              │
│  ┌─────────┐                                                 │
│  │ Qwen2.5 │                                                 │
│  │ Omni-7B │                                                 │
│  │ (VLM)   │                                                 │
│  └─────────┘                                                 │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 데이터 흐름

#### 문서 적재 (Ingest)
```
파일 업로드 (/api/ingest)
  │
  ▼
[1] RAG-Anything 파서로 문서 파싱
    - MinerU: PDF → content_list (텍스트, 이미지, 테이블, 수식)
    - PaddleOCR: 스캔 문서 OCR
    - document_parser: DOCX, PPTX, XLSX (기존 코드 재사용)
  │
  ▼
[2] 이미지 추출 및 저장 (기존 파이프라인 유지)
    - 파서 출력에서 img_path 추출
    - /assets/extracted_images/{doc_id}/ 에 저장
    - PostgreSQL image_assets 테이블에 메타데이터 삽입
  │
  ▼
[3] RAG-Anything로 콘텐츠 삽입
    - 텍스트: LightRAG에 삽입 (자동 청킹 + bge-m3 임베딩 + 엔티티/관계 추출)
    - 멀티모달: ModalProcessor로 VLM 분석 → 설명 텍스트를 LightRAG에 삽입
  │
  ▼
[4] 이미지-청크 매핑
    - LightRAG 청크와 image_assets를 페이지 번호 기반으로 연결
    - chunk_images 테이블에 매핑 저장
  │
  ▼
[5] 문서 상태 업데이트 (documents.status = 'completed')
```

#### 쿼리 (Chat)
```
사용자 질문 (/api/chat)
  │
  ▼
[1] RAG-Anything aquery() 호출
    - mode="hybrid" (벡터 + 그래프 + 키워드 검색)
    - 관련 청크 + 엔티티/관계 컨텍스트 반환
  │
  ▼
[2] 소스 정보 구성
    - LightRAG 반환 결과에서 청크 ID, 페이지 번호, 관련도 추출
    - PostgreSQL에서 해당 청크의 관련 이미지 조회
  │
  ▼
[3] SSE 스트리밍 응답 (기존 형식 유지)
    - sources: 관련 청크 목록
    - images: 관련 이미지 목록
    - settings: 검색 설정
    - token: LLM 스트리밍 토큰
    - done: 완료 신호
```


## 3. 백엔드 변경 상세

### 3.1 파일별 변경 계획

#### 유지 (변경 없음)
| 파일 | 설명 |
|------|------|
| `backend/app/main.py` | FastAPI 앱 설정 - 라우터 등록만 필요시 추가 |
| `backend/app/database.py` | PostgreSQL 연결 관리 |
| `backend/app/models/schemas.py` | Pydantic 모델 |
| `backend/app/routers/projects.py` | 프로젝트 CRUD |
| `backend/app/routers/questions.py` | 예제 질문 API |
| `backend/scripts/document_parser.py` | 비-PDF 문서 파서 (DOCX, PPTX 등) |
| `frontend/` | 전체 프론트엔드 |
| `bin/` | 서비스 관리 스크립트 |

#### 수정
| 파일 | 변경 내용 |
|------|----------|
| `backend/app/config.py` | RAG-Anything 관련 설정 추가 (working_dir, bge-m3 설정 등), SPLADE 관련 설정 제거 |
| `backend/app/routers/chat.py` | retrieval.search() → raganything_service.query() 호출로 변경 |
| `backend/app/routers/ingest.py` | 기존 파이프라인 → RAG-Anything 파이프라인으로 교체, 이미지 저장 로직 유지 |
| `backend/app/routers/chunks.py` | LightRAG 청크 조회로 변경 |
| `backend/app/services/llm.py` | 시스템 프롬프트, 컨텍스트 빌딩 로직 유지. LLM 호출부 유지 |

#### 신규 생성
| 파일 | 설명 |
|------|------|
| `backend/app/services/raganything_service.py` | RAGAnything 인스턴스 관리 및 래핑 서비스 |
| `backend/app/services/image_service.py` | 이미지 추출/저장/매핑 서비스 (기존 로직 리팩토링) |

#### 삭제 (더 이상 불필요)
| 파일 | 이유 |
|------|------|
| `backend/app/services/embedding.py` | SPLADE → bge-m3 (RAG-Anything 내장) |
| `backend/app/services/retrieval.py` | pgvector 검색 → LightRAG 쿼리 |
| `backend/scripts/chunker.py` | LightRAG 내장 청킹 사용 |
| `backend/scripts/embedder.py` | RAG-Anything 내장 임베딩 사용 |

### 3.2 핵심 신규 모듈: raganything_service.py

```python
"""
RAG-Anything Service
RAGAnything 인스턴스를 FastAPI 서비스로 래핑
"""
import os
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from app.config import settings

# ── 모델 함수 (module-level, pickle-safe) ──

VLLM_BASE_URL = f"http://{settings.vllm_host}:{settings.vllm_port}/v1"
VLLM_MODEL_NAME = Path(settings.llm_model_path).name
EMBED_BASE_URL = f"http://{settings.vllm_host}:{settings.embedding_port}/v1"
EMBED_MODEL = settings.embedding_model_name  # "BAAI/bge-m3"

async def llm_model_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = None,
    **kwargs,
) -> str:
    return await openai_complete_if_cache(
        model=VLLM_MODEL_NAME,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=VLLM_BASE_URL,
        api_key="dummy",
        **kwargs,
    )

async def vision_model_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = None,
    image_data: Optional[str] = None,
    **kwargs,
) -> str:
    """VLM 호출 - Qwen2.5-Omni-7B"""
    # Qwen2.5-Omni가 vLLM에서 서빙되는 경우
    vlm_base_url = f"http://{settings.vllm_host}:{settings.vlm_port}/v1"
    vlm_model_name = Path(settings.multimodal_model_path).name
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    if image_data:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                {"type": "text", "text": prompt}
            ]
        })
    else:
        messages.append({"role": "user", "content": prompt})
    
    return await openai_complete_if_cache(
        model=vlm_model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=messages,
        base_url=vlm_base_url,
        api_key="dummy",
        **kwargs,
    )

async def embedding_func(texts: List[str]) -> List[List[float]]:
    embeddings = await openai_embed(
        texts=texts,
        model=EMBED_MODEL,
        base_url=EMBED_BASE_URL,
        api_key="dummy",
    )
    return embeddings.tolist()


class RAGAnythingService:
    """프로젝트별 RAGAnything 인스턴스를 관리하는 서비스"""
    
    _instances: Dict[int, RAGAnything] = {}
    
    @classmethod
    def _get_working_dir(cls, project_id: int) -> str:
        """프로젝트별 working directory 반환"""
        base = Path(settings.raganything_working_dir)  # 예: /data/rag_storage
        project_dir = base / f"project_{project_id}"
        project_dir.mkdir(parents=True, exist_ok=True)
        return str(project_dir)
    
    @classmethod
    async def get_instance(cls, project_id: int) -> RAGAnything:
        """프로젝트별 RAGAnything 싱글톤 인스턴스 반환"""
        if project_id not in cls._instances:
            working_dir = cls._get_working_dir(project_id)
            
            config = RAGAnythingConfig(
                working_dir=working_dir,
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
                parser="mineru",
                parse_method="auto",
            )
            
            rag = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                vision_model_func=vision_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=1024,
                    max_token_size=8192,
                    func=embedding_func,
                ),
            )
            
            cls._instances[project_id] = rag
        
        return cls._instances[project_id]
    
    @classmethod
    async def ingest_document(
        cls, project_id: int, file_path: str, **kwargs
    ) -> Dict[str, Any]:
        """문서를 RAG-Anything으로 적재"""
        rag = await cls.get_instance(project_id)
        result = await rag.process_document_complete(file_path, **kwargs)
        return result
    
    @classmethod
    async def query(
        cls,
        project_id: int,
        question: str,
        mode: str = "hybrid",
        **kwargs
    ) -> str:
        """RAG-Anything으로 질의"""
        rag = await cls.get_instance(project_id)
        result = await rag.aquery(question, mode=mode, **kwargs)
        return result
    
    @classmethod
    async def cleanup(cls, project_id: int = None):
        """인스턴스 정리"""
        if project_id:
            if project_id in cls._instances:
                await cls._instances[project_id].finalize_storages()
                del cls._instances[project_id]
        else:
            for rag in cls._instances.values():
                await rag.finalize_storages()
            cls._instances.clear()
```

### 3.3 config.py 변경사항

```python
class Settings(BaseSettings):
    # ... (기존 설정 유지) ...
    
    # ── RAG-Anything 설정 (추가) ──
    raganything_working_dir: str = "/data/rag_storage"
    
    # 임베딩 모델 (bge-m3)
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_port: int = 8001  # vLLM 임베딩 서버 포트
    embedding_dim: int = 1024
    
    # VLM (Qwen2.5-Omni)
    vlm_port: int = 8767  # VLM vLLM 서버 포트
    
    # RAG-Anything 쿼리 모드
    rag_query_mode: str = "hybrid"  # local, global, hybrid, mix
    
    # ── SPLADE 관련 설정 (제거 또는 deprecated 마킹) ──
    # splade_batch_size, splade_chunk_size, splade_overlap 등 제거
```


### 3.4 chat.py 변경사항

기존 `retrieval.search()` + `llm.generate_stream()`을 **RAG-Anything query + 직접 LLM 스트리밍**으로 교체.

**변경 전 흐름:**
```
질문 → SPLADE 임베딩 → pgvector 검색 → 리랭킹 → 컨텍스트 구성 → LLM 스트리밍
```

**변경 후 흐름:**
```
질문 → RAG-Anything aquery (벡터+그래프 검색) → 컨텍스트 획득 → LLM 스트리밍
```

**핵심 변경 코드:**
```python
# chat.py (변경 후)

from app.services.raganything_service import RAGAnythingService
from app.services.llm import get_llm_service, build_context, build_prompt, _trim_history

@router.post("/chat")
async def chat(request: ChatRequest):
    async def generate():
        start_time = time.time()
        yield ":\n\n"
        
        # Step 1: RAG-Anything 하이브리드 검색
        try:
            rag_service = RAGAnythingService()
            rag_result = await rag_service.query(
                project_id=request.project_id,
                question=request.question,
                mode=settings.rag_query_mode,
            )
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
            return
        
        # Step 2: 소스 정보 구성 (LightRAG 결과에서 추출)
        sources = extract_sources_from_rag_result(rag_result, request.project_id)
        source_list = format_sources(sources)
        yield f"data: {json.dumps({'type': 'sources', 'data': source_list})}\n\n"
        
        # Step 3: 관련 이미지 조회 (PostgreSQL에서)
        related_images = get_related_images(sources)
        yield f"data: {json.dumps({'type': 'images', 'data': related_images})}\n\n"
        
        # Step 4: LLM 스트리밍 (기존 LLM 서비스 재사용)
        llm = get_llm_service()
        # RAG-Anything이 이미 검색한 컨텍스트를 활용
        context = rag_result  # LightRAG가 반환한 종합 컨텍스트
        
        # 또는: RAG-Anything의 컨텍스트 + 기존 프롬프트 템플릿 사용
        async for chunk in llm.generate_stream(
            question=request.question,
            sources=sources,
            temperature=temperature,
            project_id=request.project_id,
            history=history_dicts,
        ):
            yield f"data: {json.dumps({'type': 'token', 'data': chunk})}\n\n"
        
        response_time_ms = int((time.time() - start_time) * 1000)
        yield f"data: {json.dumps({'type': 'done', 'data': {'response_time_ms': response_time_ms}})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

**설계 결정: 2가지 쿼리 전략 옵션**

| 옵션 | 설명 | 장점 | 단점 |
|------|------|------|------|
| A. RAG-Anything 완전 위임 | `aquery()`가 검색+LLM 생성까지 수행 | 간단, 그래프 검색 최적화 | 스트리밍 미지원, 프롬프트 커스터마이징 제한 |
| B. 하이브리드 (추천) | RAG-Anything은 검색만, LLM 스트리밍은 기존 서비스 | 스트리밍 유지, 프롬프트 자유도 | 약간의 통합 복잡도 |

**추천: 옵션 B (하이브리드)**
- RAG-Anything의 `aquery()`는 LLM 응답까지 포함하여 반환하므로, 내부 검색 결과만 활용
- LightRAG의 `_get_relevant_context()` 또는 유사 내부 메서드로 검색 결과만 추출
- 스트리밍은 기존 `LLMService.generate_stream()` 재사용

### 3.5 ingest.py 변경사항

```python
# ingest.py 핵심 변경

@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    project_id: int = Form(...),
    admin_key: str = Form(...),
    document_hint: str = Form(""),
):
    # 관리자 인증 (기존 동일)
    if admin_key != settings.admin_access_key:
        raise HTTPException(status_code=403)
    
    # 임시 파일 저장 (기존 동일)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # ── 변경된 파이프라인 ──
        
        # 1. 문서 레코드 삽입 (기존 동일)
        document_id = insert_document_record(project_id, original_name, file_size, ...)
        
        # 2. RAG-Anything으로 문서 처리 (파싱 + 청킹 + 임베딩 + 지식그래프)
        rag_service = RAGAnythingService()
        rag = await rag_service.get_instance(project_id)
        rag_result = await rag.process_document_complete(tmp_path)
        
        # 3. 이미지 추출 및 저장 (기존 파이프라인 유지 또는 MinerU 출력 활용)
        #    MinerU 파서는 img_path를 content_list에 포함하므로,
        #    파서 출력에서 이미지 경로를 추출하여 image_assets에 저장
        image_assets = extract_and_store_images(
            rag_result, document_id, project_id
        )
        
        # 4. VLM 캡셔닝 (RAG-Anything의 ImageModalProcessor가 이미 수행)
        #    → 추가 캡셔닝 불필요 (RAG-Anything이 처리)
        
        # 5. 청크-이미지 매핑
        create_chunk_image_mappings(document_id, rag_result, image_assets)
        
        # 6. 문서 상태 업데이트
        update_document_status(document_id, 'completed')
        
        return {"status": "success", "document_id": document_id, ...}
    except Exception:
        cleanup_failed_document(document_id)
        raise
```


## 4. 데이터베이스 변경

### 4.1 스키마 변경

```sql
-- ── 제거: chunks 테이블의 sparsevec 임베딩 ──
-- LightRAG가 자체적으로 벡터 저장소를 관리하므로,
-- PostgreSQL의 chunks 테이블에서 embedding 컬럼 제거

ALTER TABLE chunks DROP COLUMN IF EXISTS embedding;

-- ── 추가: lightrag_chunk_id 컬럼 ──
-- LightRAG 내부 청크 ID와의 매핑을 위해
ALTER TABLE chunks ADD COLUMN lightrag_chunk_id VARCHAR(64);
CREATE INDEX idx_chunks_lightrag_id ON chunks(lightrag_chunk_id);

-- ── 추가: documents 테이블에 document_hint 컬럼 (이미 존재할 수 있음) ──
ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_hint TEXT DEFAULT '';

-- ── 유지: 나머지 테이블 구조 동일 ──
-- projects, documents, image_assets, chunk_images, questions → 변경 없음
```

### 4.2 데이터 저장소 이원화

| 데이터 | 저장소 | 관리 주체 |
|--------|--------|----------|
| 임베딩 벡터 | LightRAG nano-vectordb (파일 기반) | RAG-Anything |
| 지식 그래프 (엔티티/관계) | LightRAG 내장 그래프 저장소 | RAG-Anything |
| 문서 청크 텍스트 | LightRAG KV 저장소 | RAG-Anything |
| 프로젝트/문서 메타데이터 | PostgreSQL | 직접 관리 |
| 이미지 메타데이터 | PostgreSQL (image_assets) | 직접 관리 |
| 청크-이미지 매핑 | PostgreSQL (chunk_images) | 직접 관리 |
| 이미지 파일 | 파일시스템 (/assets) | 직접 관리 |
| 예제 질문 | PostgreSQL (questions) | 직접 관리 |

### 4.3 프로젝트별 데이터 격리

LightRAG는 `working_dir` 기반으로 데이터를 격리함. 프로젝트별 디렉토리 구조:

```
/data/rag_storage/
├── project_1/
│   ├── vdb_entities.json        # 엔티티 벡터 DB
│   ├── vdb_relationships.json   # 관계 벡터 DB
│   ├── vdb_chunks.json          # 청크 벡터 DB
│   ├── graph_chunk_entity_relation.graphml  # 지식 그래프
│   ├── kv_store_full_docs.json  # 전체 문서 저장소
│   ├── kv_store_text_chunks.json  # 텍스트 청크
│   └── kv_store_llm_response_cache.json  # LLM 캐시
├── project_2/
│   └── ...
└── project_3/
    └── ...
```

## 5. 모델 서빙 구성

### 5.1 GPU 할당 계획

```
GPU 0 (L40S):
  - vLLM: Qwen3-4B-Instruct (Port 8766)
  - vLLM: BAAI/bge-m3 임베딩 (Port 8001)

GPU 1 (L40S):
  - vLLM: Qwen2.5-Omni-7B VLM (Port 8767)
```

### 5.2 vLLM 시작 명령

```bash
# GPU 0: LLM (Qwen3-4B)
CUDA_VISIBLE_DEVICES=0 vllm serve ${LLM_MODEL_PATH} \
    --port 8766 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.45

# GPU 0: 임베딩 (bge-m3) - 같은 GPU, 낮은 메모리 사용
CUDA_VISIBLE_DEVICES=0 vllm serve BAAI/bge-m3 \
    --task embedding \
    --port 8001 \
    --gpu-memory-utilization 0.15

# GPU 1: VLM (Qwen2.5-Omni-7B)
CUDA_VISIBLE_DEVICES=1 vllm serve ${MULTIMODAL_MODEL_PATH} \
    --port 8767 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85
```

### 5.3 모델 메모리 추정

| 모델 | 파라미터 | 예상 VRAM | GPU |
|------|---------|----------|-----|
| Qwen3-4B-Instruct | 4B | ~10 GB | GPU 0 |
| BAAI/bge-m3 | 568M | ~3 GB | GPU 0 |
| Qwen2.5-Omni-7B | 7B | ~18 GB | GPU 1 |
| **합계** | | ~31 GB | L40S 48GB x 2 = 충분 |

## 6. 프론트엔드 영향 분석

### 6.1 API 호환성

프론트엔드는 변경하지 않음. 백엔드 API 응답 형식을 동일하게 유지:

| API | 응답 형식 | 변경 여부 |
|-----|----------|----------|
| POST /api/chat (SSE) | sources, images, settings, token, done | 유지 |
| POST /api/chat/sync | answer, sources, images, settings_used | 유지 |
| GET /api/chunks/{id} | chunk detail + context + images | 유지 |
| POST /api/ingest | status, document_id, pages, chunks | 유지 |
| GET /api/projects | projects list | 유지 |

### 6.2 프론트엔드 파라미터 변경

| 파라미터 | 현재 | 변경 |
|---------|------|------|
| top_n_terms | SPLADE sparse 항목수 (64-512) | 제거 또는 무시 |
| top_k | 검색 청크 수 (1-10) | 유지 (LightRAG top_k로 전달) |
| temperature | LLM 온도 (0.0-1.0) | 유지 |

**호환성 방안:** `top_n_terms`는 백엔드에서 수신하되 무시. 프론트엔드 `ParameterControl.tsx`에서 나중에 제거하거나, RAG-Anything의 다른 파라미터(예: 쿼리 모드)로 대체 가능.


## 7. 이미지 처리 하이브리드 설계

### 7.1 문제점
RAG-Anything은 이미지를 VLM으로 분석하여 텍스트 설명만 지식그래프에 삽입할 뿐,
이미지 파일 자체를 프론트엔드에 서빙하는 기능이 없다.
rag-baseplate의 이미지 서빙 기능을 유지하려면 별도 처리가 필요.

### 7.2 하이브리드 방안

```
문서 파싱 (RAG-Anything MinerU)
  │
  ├─→ content_list에 img_path 포함
  │     예: {"type": "image", "img_path": "/tmp/mineru_output/images/img_001.png"}
  │
  ├─→ [RAG-Anything 경로]
  │     ImageModalProcessor → VLM 분석 → 설명 텍스트 → LightRAG 삽입
  │
  └─→ [이미지 서빙 경로] (별도 처리)
        img_path에서 이미지 복사 → /assets/extracted_images/doc_{id}/
        → image_assets 테이블에 메타데이터 삽입
        → chunk_images 테이블에 페이지 기반 매핑
```

### 7.3 image_service.py 설계

```python
"""이미지 추출, 저장, 매핑 서비스"""
import shutil
from pathlib import Path
from typing import List, Dict
from app.database import get_cursor

ASSETS_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "extracted_images"

def extract_images_from_content_list(
    content_list: list,
    document_id: int,
) -> List[Dict]:
    """RAG-Anything 파서 출력(content_list)에서 이미지를 추출하여 저장"""
    image_assets = []
    output_dir = ASSETS_ROOT / f"doc_{document_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for item in content_list:
        if item.get("type") != "image":
            continue
        
        img_path = item.get("img_path")
        if not img_path or not Path(img_path).exists():
            continue
        
        # 이미지 파일 복사
        dest_name = Path(img_path).name
        dest_path = output_dir / dest_name
        shutil.copy2(img_path, dest_path)
        
        # 상대 URL 경로
        rel_url = f"/assets/extracted_images/doc_{document_id}/{dest_name}"
        
        image_assets.append({
            "document_id": document_id,
            "page_number": item.get("page_idx", 0) + 1,
            "image_type": "extracted",
            "image_path": rel_url,
            "caption": ", ".join(item.get("img_caption", [])),
            "width": item.get("img_width"),
            "height": item.get("img_height"),
        })
    
    return image_assets

def store_image_assets(image_assets: List[Dict]) -> List[int]:
    """이미지 메타데이터를 PostgreSQL에 저장"""
    image_ids = []
    with get_cursor() as cursor:
        for asset in image_assets:
            cursor.execute("""
                INSERT INTO image_assets 
                (document_id, page_number, image_type, image_path, caption, width, height)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                asset["document_id"], asset["page_number"],
                asset["image_type"], asset["image_path"],
                asset.get("caption"), asset.get("width"), asset.get("height"),
            ))
            image_ids.append(cursor.fetchone()["id"])
    return image_ids

def create_chunk_image_mappings(
    document_id: int,
    image_assets: List[Dict],
    image_ids: List[int],
):
    """페이지 번호 기반으로 청크-이미지 매핑 생성"""
    with get_cursor() as cursor:
        # LightRAG 청크에서 페이지 정보를 추출하기 어려우므로,
        # chunks 테이블의 page_number를 기준으로 매핑
        for asset, img_id in zip(image_assets, image_ids):
            page = asset["page_number"]
            cursor.execute("""
                INSERT INTO chunk_images (chunk_id, image_asset_id)
                SELECT c.id, %s
                FROM chunks c
                WHERE c.document_id = %s AND c.page_number = %s
                ON CONFLICT DO NOTHING
            """, (img_id, document_id, page))
```

## 8. 마이그레이션 전략

### 8.1 단계별 진행 계획

```
Phase 1: 인프라 준비 (1일)
├── vLLM bge-m3 임베딩 서버 설정 및 테스트
├── RAG-Anything 설치 (pip install raganything)
├── LightRAG 동작 확인
└── DB 스키마 마이그레이션 SQL 준비

Phase 2: 백엔드 서비스 개발 (2-3일)
├── raganything_service.py 구현
├── image_service.py 구현
├── config.py 수정
└── 단위 테스트

Phase 3: 라우터 통합 (1-2일)
├── ingest.py 수정 (RAG-Anything 파이프라인 적용)
├── chat.py 수정 (RAG-Anything 쿼리 적용)
├── chunks.py 수정
└── 통합 테스트

Phase 4: 검증 및 최적화 (1-2일)
├── 기존 데모 데이터 재적재
├── 쿼리 품질 비교 (SPLADE vs LightRAG)
├── 성능 벤치마크
└── 프론트엔드 E2E 테스트

Phase 5: 기존 데이터 마이그레이션 (선택)
├── 기존 프로젝트 데이터 재적재
└── 이미지 자산 확인
```

### 8.2 롤백 계획
- rag-baseplate의 기존 코드를 별도 브랜치로 보존
- PostgreSQL 기존 데이터는 건드리지 않음 (RAG-Anything은 별도 저장소 사용)
- 문제 발생 시 기존 브랜치로 복귀 가능

## 9. 리스크 및 대응

| 리스크 | 영향도 | 대응 방안 |
|--------|--------|----------|
| LightRAG 쿼리가 소스 청크 ID를 직접 반환하지 않음 | 높음 | LightRAG 내부 API 분석하여 청크 추출 로직 커스텀 구현 |
| bge-m3의 한국어 검색 품질이 SPLADE-ko보다 낮을 수 있음 | 중간 | A/B 테스트로 검증, 필요시 리랭킹 로직 추가 |
| MinerU 파서가 DOCX/PPTX를 직접 지원하지 않음 | 낮음 | 기존 document_parser.py 유지하여 비-PDF 처리 |
| 프로젝트별 RAGAnything 인스턴스의 메모리 사용 | 중간 | LRU 캐시로 인스턴스 수 제한, lazy 초기화 |
| VLM 서빙 안정성 (Qwen2.5-Omni + vLLM) | 중간 | VLM 실패 시 캡셔닝 스킵, 텍스트만 처리 |
| LightRAG의 엔티티/관계 추출이 한국어에서 품질 저하 | 중간 | 프롬프트 한국어화 (prompt_manager 활용) |

## 10. 검증 계획

### 10.1 기능 테스트
- [ ] 문서 업로드 (PDF, DOCX, PPTX, XLSX)
- [ ] 이미지 추출 및 프론트엔드 표시
- [ ] 텍스트 쿼리 → 관련 소스 및 이미지 반환
- [ ] SSE 스트리밍 응답
- [ ] 프로젝트 생성/삭제
- [ ] 문서 삭제 시 LightRAG 데이터 정리
- [ ] 대화 히스토리 유지

### 10.2 품질 테스트
- [ ] 기존 데모 데이터 6개 프로젝트 재적재
- [ ] 동일 질문에 대한 응답 품질 비교 (SPLADE vs LightRAG)
- [ ] 수치/표 질문 정확도
- [ ] 이미지 관련 질문 응답 품질

### 10.3 성능 테스트
- [ ] 문서 적재 시간 비교
- [ ] 쿼리 응답 시간 (첫 토큰까지)
- [ ] 동시 쿼리 처리

## 11. 의존성 정리

### 11.1 추가 패키지
```
raganything>=1.2.10
lightrag-hku
```

### 11.2 제거 가능 패키지
```
# SPLADE 관련 (더 이상 불필요)
# torch, transformers → VLM/임베딩에 필요할 수 있으므로 유지 검토
```

### 11.3 requirements.txt 업데이트
```
# 기존 유지
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

# 추가
raganything>=1.2.10

# 제거 또는 선택적
# torch (RAG-Anything이 필요시 자동 설치)
# transformers (tokenizer용으로 유지 가능)
```

---

*문서 버전: v1.0*
*작성일: 2026-04-07*
*프로젝트: EICN RAG Prototype*
