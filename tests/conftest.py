"""
共享 pytest fixtures

在 test_offline/、test_online/、test_api/、test_graph/ 中均可使用。
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================
# 环境变量（在所有测试前设置）
# ============================================================
@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
    """为所有测试设置安全的测试环境变量，防止读取真实的 .env 文件。"""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-dashscope-key-12345678")
    monkeypatch.setenv("MYSQL_PASSWORD", "test-password")
    monkeypatch.setenv("MYSQL_HOST", "localhost")
    monkeypatch.setenv("MYSQL_PORT", "3306")
    monkeypatch.setenv("MILVUS_HOST", "localhost")
    monkeypatch.setenv("MILVUS_PORT", "19530")
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("REDIS_PASSWORD", "")
    # 鉴权测试: 默认不设 API Key → 鉴权跳过，保持向后兼容
    monkeypatch.setenv("APP_API_KEY", "")
    yield


# ============================================================
# 测试数据
# ============================================================
@pytest.fixture
def sample_raw_text():
    """标准药品说明书测试文本（含多个【】章节）。"""
    return (
        "【药品名称】\n"
        "通用名称：阿司匹林肠溶片\n"
        "商品名称：拜阿司匹灵\n"
        "英文名称：Aspirin Enteric-coated Tablets\n\n"
        "【适应症】\n"
        "1. 用于解热镇痛：缓解轻至中度疼痛，如头痛、牙痛、神经痛、肌肉痛、痛经及关节痛等。\n"
        "2. 用于感冒等发热疾病的退热。\n"
        "3. 用于预防心脑血管疾病：降低稳定性和不稳定性心绞痛、心肌梗死、"
        "脑梗死及一过性脑缺血发作的风险。\n\n"
        "【用法用量】\n"
        "解热镇痛：成人一次0.3～0.6g，一日3次，饭后服用。\n"
        "心脑血管疾病预防：一次50～100mg，一日1次。\n"
        "儿童用量需咨询医师或药师。\n\n"
        "【禁忌】\n"
        "1. 对阿司匹林或其他非甾体抗炎药过敏者禁用。\n"
        "2. 活动性消化性溃疡或出血者禁用。\n"
        "3. 血友病或血小板减少症患者禁用。\n"
        "4. 妊娠最后三个月孕妇禁用。\n\n"
        "【不良反应】\n"
        "1. 胃肠道反应：恶心、呕吐、上腹部不适或疼痛等。\n"
        "2. 出血倾向：牙龈出血、鼻出血、皮肤瘀斑等。\n"
        "3. 过敏反应：皮疹、荨麻疹、哮喘等。\n"
        "4. 长期大量用药可能引起肝肾功能损伤。\n\n"
        "【注意事项】\n"
        "1. 本品为对症治疗药，用于解热连续使用不得超过3天，止痛不得超过5天。\n"
        "2. 服药期间不得饮酒或含酒精的饮料。\n"
        "3. 老年患者因肾功能下降应适当减量。\n"
        "4. 孕妇及哺乳期妇女慎用。\n\n"
        "【药物相互作用】\n"
        "1. 与其他非甾体抗炎药合用，增加胃肠道损伤风险。\n"
        "2. 与抗凝药（如华法林）合用，增加出血风险。\n"
        "3. 与糖皮质激素合用，增加消化道溃疡和出血风险。\n\n"
        "【贮藏】\n"
        "密封，在阴凉干燥处保存。\n\n"
        "【有效期】\n"
        "36个月\n\n"
        "【生产企业】\n"
        "拜耳医药保健有限公司"
    )


@pytest.fixture
def sample_drug_query():
    """标准药品查询问题。"""
    return "阿司匹林一天吃几次？每次吃多少？"


@pytest.fixture
def sample_non_drug_query():
    """非药品查询问题。"""
    return "今天天气怎么样？"


@pytest.fixture
def sample_chunks(sample_raw_text):
    """模拟 split_document 返回的 Chunk 列表。"""
    return [
        {"section": "适应症", "chunk_text": "用于解热镇痛：缓解轻至中度疼痛...",
         "drug_name": "阿司匹林肠溶片", "score": 0.95, "doc_id": 1, "chunk_index": 0},
        {"section": "用法用量", "chunk_text": "成人一次0.3～0.6g，一日3次...",
         "drug_name": "阿司匹林肠溶片", "score": 0.92, "doc_id": 1, "chunk_index": 1},
        {"section": "禁忌", "chunk_text": "对阿司匹林或其他非甾体抗炎药过敏者禁用...",
         "drug_name": "阿司匹林肠溶片", "score": 0.88, "doc_id": 1, "chunk_index": 2},
        {"section": "不良反应", "chunk_text": "胃肠道反应：恶心、呕吐...",
         "drug_name": "阿司匹林肠溶片", "score": 0.85, "doc_id": 1, "chunk_index": 3},
    ]


@pytest.fixture
def sample_chat_history():
    """模拟对话历史。"""
    return [
        {"role": "user", "content": "阿司匹林有什么不良反应？", "timestamp": "2026-06-15T10:00:00"},
        {"role": "assistant", "content": "阿司匹林的不良反应包括胃肠道反应...",
         "timestamp": "2026-06-15T10:00:05",
         "sources": [{"drug_name": "阿司匹林肠溶片", "section": "不良反应",
                      "chunk_text": "胃肠道反应...", "score": 0.95}]},
    ]


# ============================================================
# Mock 对象
# ============================================================
@pytest.fixture
def mock_dashscope_response():
    """创建可复用的 DashScope API 响应 mock。"""

    def _make_response(status_code=200, text="", choices_content=None, embeddings=None):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.message = "" if status_code == 200 else "Error"

        mock_output = MagicMock()
        if choices_content is not None:
            mock_choice = MagicMock()
            mock_choice.message.content = choices_content
            mock_output.choices = [mock_choice]
            mock_output.text = None
        elif text:
            mock_output.choices = None
            mock_output.text = text
        elif embeddings is not None:
            mock_output.choices = None
            mock_output.text = None
            mock_output.get.return_value = embeddings
        else:
            mock_output.choices = None
            mock_output.text = None

        mock_resp.output = mock_output
        return mock_resp

    return _make_response


@pytest.fixture
def mock_milvus_client():
    """Mock MilvusClient。"""
    client = MagicMock()
    client.connect.return_value = None
    client.disconnect.return_value = None
    client.collection_exists.return_value = True
    client.search.return_value = [
        {"entity": {"doc_id": 1, "drug_name": "阿司匹林肠溶片", "chunk_text": "用于解热镇痛...",
         "section": "适应症", "chunk_index": 0}, "distance": 0.95},
        {"entity": {"doc_id": 1, "drug_name": "阿司匹林肠溶片", "chunk_text": "成人一次0.3～0.6g...",
         "section": "用法用量", "chunk_index": 1}, "distance": 0.92},
    ]
    client.insert_embeddings.return_value = {"insert_count": 2}
    client.count.return_value = 100
    client.get_collection_info.return_value = {"name": "drug_chunks", "num_entities": 100}
    return client


@pytest.fixture
def mock_mysql_client():
    """Mock MySQLClient。"""
    client = MagicMock()
    client.connect.return_value = None
    client.disconnect.return_value = None
    client.is_ready.return_value = True
    client.insert_raw_doc.return_value = 1
    client.insert_chunks_batch.return_value = None
    client.insert_index_record.return_value = None
    client.update_index_record.return_value = None
    client.drug_exists.return_value = False
    client.delete_drug_by_name.return_value = []
    client.get_index_record.return_value = None
    client.bm25_search.return_value = [
        {"doc_id": 1, "drug_name": "阿司匹林肠溶片", "chunk_text": "用于解热镇痛...",
         "section": "适应症", "score": 8.5, "chunk_index": 0},
    ]
    client.get_all_drug_names.return_value = ["阿司匹林肠溶片"]
    return client


@pytest.fixture
def mock_embedder():
    """Mock Embedder。"""
    from app.offline.embedder import EmbeddingResult

    embedder = MagicMock()
    embedder.embed.return_value = EmbeddingResult(
        embeddings=[[0.1] * 1024, [0.2] * 1024, [0.3] * 1024, [0.4] * 1024],
        failed_indices=[],
        total_attempted=4,
        total_succeeded=4,
    )
    return embedder


@pytest.fixture
def mock_redis():
    """Mock AsyncRedis。"""
    redis = MagicMock()
    redis.get.return_value = None
    redis.set.return_value = True
    redis.delete.return_value = True
    redis.ping.return_value = True
    redis.aclose.return_value = None
    redis.ttl.return_value = 3600
    return redis
