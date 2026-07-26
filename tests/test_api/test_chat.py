"""
测试 API 问答端点

v1.0.0: /chat 改为 multipart/form-data，支持文件上传。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ============================================================
# TestClient fixtures
# ============================================================
@pytest.fixture
def client():
    """创建带 mock 依赖的测试客户端（含 JWT 鉴权绕过）。"""
    import app.services.memory_manager as mm_module

    mock_redis_hm = AsyncMock()
    mock_redis_hm.get_history.return_value = []
    mock_redis_hm.add_turn.return_value = None
    mock_redis_hm.clear_history.return_value = True
    mock_redis_hm.close = AsyncMock()
    mock_redis_hm.get_summary = AsyncMock(return_value="")
    mock_redis_hm.set_summary = AsyncMock()

    mock_compiled = MagicMock()
    mock_compiled.invoke.return_value = {
        "answer": "根据病例分析，患者高血压需调整用药方案...",
        "sources": [
            {"drug_name": "硝苯地平", "section": "用法用量",
             "chunk_text": "口服，一次10mg，一日3次。", "score": 0.95, "doc_id": 1,
             "source_type": "drug"},
        ],
        "intent": "clinical",
        "intent_confidence": 0.95,
        "template_used": "case_summary",
        "error": None,
        "case_profile": {},
        "search_breakdown": {"drug": 3},
    }

    with patch("app.api.main.get_graph", return_value=mock_compiled), \
         patch("app.api.main.AsyncRedisHistoryManager", return_value=mock_redis_hm), \
         patch.object(mm_module.MemoryManager, "summarize", return_value=("", [])):
        from app.api.main import app
        from app.api.dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: {
            "user_id": 1,
            "username": "testuser",
        }

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    app.dependency_overrides.clear()


# ============================================================
# Schemas 验证
# ============================================================
class TestChatSchemas:
    """测试 Pydantic 模型验证。"""

    def test_chat_request_valid(self):
        from app.schemas.chat import ChatRequest
        req = ChatRequest(message="患者高血压怎么治疗？")
        assert req.message == "患者高血压怎么治疗？"
        assert req.session_id is None
        assert req.analysis_mode == "comprehensive"

    def test_chat_request_empty_message(self):
        from app.schemas.chat import ChatRequest
        with pytest.raises(Exception):
            ChatRequest(message="")

    def test_chat_request_with_analysis_mode(self):
        from app.schemas.chat import ChatRequest
        req = ChatRequest(message="分析病例", analysis_mode="treatment")
        assert req.analysis_mode == "treatment"

    def test_source_doc_v1(self):
        """SourceDoc v1.0.0 扩展字段。"""
        from app.schemas.chat import SourceDoc
        doc = SourceDoc(
            drug_name="硝苯地平",
            section="用法用量",
            chunk_text="口服，一次10mg",
            score=0.95,
            doc_id=1,
            source_type="drug",
            evidence_level="IA",
        )
        assert doc.drug_name == "硝苯地平"
        assert doc.source_type == "drug"
        assert doc.evidence_level == "IA"

    def test_chat_response(self):
        from app.schemas.chat import ChatResponse
        resp = ChatResponse(
            answer="测试回答",
            sources=[],
            session_id="sess_test",
            intent="clinical",
            elapsed_ms=1500.5,
        )
        assert resp.answer == "测试回答"

    def test_chat_history_item(self):
        from datetime import datetime, timezone
        from app.schemas.chat import ChatHistoryItem
        item = ChatHistoryItem(
            role="user",
            content="患者高血压怎么治疗？",
            timestamp=datetime.now(timezone.utc),
        )
        assert item.role == "user"


# ============================================================
# POST /api/v1/chat
# ============================================================
class TestChatEndpoint:
    """测试单轮问答端点（multipart/form-data）。"""

    def test_chat_success(self, client):
        """成功问答返回 200。"""
        response = client.post("/api/v1/chat", data={
            "message": "患者高血压怎么治疗？",
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["intent"] == "clinical"

    def test_chat_with_session_id(self, client):
        """带 session_id 的请求。"""
        response = client.post("/api/v1/chat", data={
            "message": "追问",
            "session_id": "sess_abc123",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess_abc123"

    def test_chat_with_analysis_mode(self, client):
        """带分析模式的请求。"""
        response = client.post("/api/v1/chat", data={
            "message": "分析病例",
            "analysis_mode": "treatment",
        })
        assert response.status_code == 200

    def test_chat_empty_input(self, client):
        """空输入返回提示。"""
        response = client.post("/api/v1/chat", data={
            "message": "",
        })
        assert response.status_code == 200
        data = response.json()
        assert "请提供病例信息" in data["answer"]


# ============================================================
# GET/DELETE /api/v1/chat/history/{session_id}
# ============================================================
class TestHistoryEndpoints:
    """测试对话历史端点。"""

    @pytest.fixture
    def client_with_history(self):
        mock_redis_hm = AsyncMock()
        mock_redis_hm.get_history.return_value = [
            {"role": "user", "content": "患者高血压怎么治疗？",
             "timestamp": "2026-06-15T10:00:00"},
            {"role": "assistant", "content": "根据病例分析...",
             "timestamp": "2026-06-15T10:00:05",
             "sources": [{"drug_name": "硝苯地平", "section": "用法用量",
                         "chunk_text": "...", "score": 0.95}]},
        ]
        mock_redis_hm.add_turn.return_value = None
        mock_redis_hm.clear_history.return_value = True
        mock_redis_hm.close = AsyncMock()
        mock_redis_hm.get_summary = AsyncMock(return_value="")
        mock_redis_hm.set_summary = AsyncMock()

        with patch("app.api.main.get_graph"), \
             patch("app.api.main.AsyncRedisHistoryManager", return_value=mock_redis_hm):
            from app.api.main import app
            from app.api.dependencies import get_current_user

            app.dependency_overrides[get_current_user] = lambda: {
                "user_id": 1, "username": "testuser",
            }

            with TestClient(app, raise_server_exceptions=False) as c:
                yield c, mock_redis_hm

        app.dependency_overrides.clear()

    def test_get_history(self, client_with_history):
        client, _ = client_with_history
        response = client.get("/api/v1/chat/history/sess_test")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess_test"
        assert len(data["history"]) == 2
        assert data["turn_count"] == 1

    def test_get_empty_history(self, client_with_history):
        client, mock_hm = client_with_history
        mock_hm.get_history.return_value = []

        response = client.get("/api/v1/chat/history/new_session")
        assert response.status_code == 200
        data = response.json()
        assert data["history"] == []
        assert data["turn_count"] == 0

    def test_clear_history(self, client_with_history):
        client, mock_hm = client_with_history
        mock_hm.clear_history.return_value = True

        response = client.delete("/api/v1/chat/history/sess_test")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess_test"
        assert data["cleared"] is True


# ============================================================
# POST /api/v1/chat/stream
# ============================================================
class TestChatStreamEndpoint:
    """测试流式问答端点。"""

    @pytest.fixture
    def stream_client(self):
        mock_redis_hm = AsyncMock()
        mock_redis_hm.get_history.return_value = []
        mock_redis_hm.add_turn.return_value = None
        mock_redis_hm.clear_history.return_value = True
        mock_redis_hm.close = AsyncMock()
        mock_redis_hm.get_summary = AsyncMock(return_value="")
        mock_redis_hm.set_summary = AsyncMock()

        with patch("app.api.main.get_graph"), \
             patch("app.api.main.AsyncRedisHistoryManager", return_value=mock_redis_hm):
            from app.api.main import app
            from app.api.dependencies import get_current_user

            app.dependency_overrides[get_current_user] = lambda: {
                "user_id": 1, "username": "testuser",
            }

            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

        app.dependency_overrides.clear()

    def test_stream_returns_200(self, stream_client):
        """流式请求返回 200。"""
        from app.online.retriever import SearchResult
        from app.online.ranker import RankedDocument

        with patch("app.api.routers.chat.is_greeting", return_value=False), \
             patch("app.api.routers.chat.Gatekeeper") as mock_gk_cls, \
             patch("app.api.routers.chat.Retriever") as mock_ret_cls, \
             patch("app.api.routers.chat.Ranker") as mock_rank_cls, \
             patch("app.api.routers.chat.Generator") as mock_gen_cls, \
             patch("app.graph.nodes._llm_extract_case") as mock_extract, \
             patch("app.graph.nodes.synthesize_node") as mock_synth:
            mock_gk = MagicMock()
            mock_gk.classify.return_value = MagicMock(clinical_related=True, confidence=0.9)
            mock_gk_cls.return_value = mock_gk

            mock_ret = MagicMock()
            mock_ret.retrieve.return_value = [
                SearchResult(chunk_text="测试", drug_name="测试", section="测试",
                            score=0.9, doc_id=1, chunk_index=0, source="milvus")
            ]
            mock_ret_cls.return_value = mock_ret

            mock_rank = MagicMock()
            mock_rank.rerank.return_value = [
                RankedDocument(chunk_text="测试", drug_name="测试", section="测试",
                              score=0.9, doc_id=1, chunk_index=0,
                              rerank_index=0, original_score=0.9)
            ]
            mock_rank_cls.return_value = mock_rank

            mock_gen = MagicMock()
            mock_gen.generate_stream.return_value = iter(["根据", "资料", "，", "测试"])
            mock_gen_cls.return_value = mock_gen

            mock_extract.return_value = {"chief_complaint": "测试"}
            mock_synth.return_value = {"synthesized_context": {}}

            response = stream_client.post("/api/v1/chat/stream", data={
                "message": "测试流式问答",
            })
            assert response.status_code == 200
            content = response.text
            assert "event:" in content or "data:" in content

    def test_stream_reject_non_clinical(self, stream_client):
        """非临床问题流式拦截。"""
        with patch("app.api.routers.chat.is_greeting", return_value=False), \
             patch("app.api.routers.chat.Gatekeeper") as mock_gk_cls:
            mock_gk = MagicMock()
            mock_gk.classify.return_value = MagicMock(clinical_related=False, confidence=0.98)
            mock_gk_cls.return_value = mock_gk

            response = stream_client.post("/api/v1/chat/stream", data={
                "message": "今天天气怎么样？",
            })
            assert response.status_code == 200
            content = response.text
            assert "临床病例分析助手" in content
