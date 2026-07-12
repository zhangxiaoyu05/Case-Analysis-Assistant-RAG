"""
测试 app.offline.pipeline — 离线流程编排

覆盖: PipelineResult, run_pipeline, run_pipeline_batch, _finalize_batch
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.offline.pipeline import (
    PipelineResult,
    _finalize_batch,
    run_pipeline,
    run_pipeline_batch,
)


# ============================================================
# PipelineResult
# ============================================================
class TestPipelineResult:
    """测试 PipelineResult 数据类。"""

    def test_completed_result(self):
        """成功完成的结果。"""
        result = PipelineResult(
            batch_id="test-batch-001",
            doc_id=1,
            drug_name="阿司匹林肠溶片",
            source_file="/tmp/test.txt",
            total_chunks=4,
            indexed_chunks=4,
            failed_chunks=0,
            status="completed",
            elapsed_seconds=1.5,
        )
        assert result.status == "completed"
        assert result.total_chunks == 4
        assert result.indexed_chunks == 4
        assert result.failed_chunks == 0
        assert result.error_message is None

    def test_failed_result(self):
        """失败的结果。"""
        result = PipelineResult(
            batch_id="test-batch-002",
            doc_id=-1,
            drug_name="unknown",
            source_file="/tmp/missing.pdf",
            total_chunks=0,
            indexed_chunks=0,
            failed_chunks=0,
            status="failed",
            error_message="文档加载失败",
            elapsed_seconds=0.1,
        )
        assert result.status == "failed"
        assert result.error_message == "文档加载失败"

    def test_partial_result(self):
        """部分成功。"""
        result = PipelineResult(
            batch_id="test-batch-003",
            doc_id=1,
            drug_name="测试药",
            source_file="/tmp/test.txt",
            total_chunks=4,
            indexed_chunks=3,
            failed_chunks=1,
            status="partial",
            warnings=["Milvus 插入失败: ConnectionError"],
        )
        assert result.status == "partial"
        assert len(result.warnings) == 1


# ============================================================
# _finalize_batch
# ============================================================
class TestFinalizeBatch:
    """测试批次状态更新辅助函数。"""

    def test_finalize_success(self, mock_mysql_client):
        """正常更新批次状态。"""
        _finalize_batch(
            mock_mysql_client,
            batch_id="batch-1",
            status="completed",
            total_chunks=4,
            indexed_chunks=4,
        )
        mock_mysql_client.update_index_record.assert_called_once()

    def test_finalize_with_error(self, mock_mysql_client):
        """更新失败不抛异常（容错）。"""
        mock_mysql_client.update_index_record.side_effect = RuntimeError("DB error")
        # 不应抛出异常
        _finalize_batch(
            mock_mysql_client,
            batch_id="batch-1",
            status="failed",
            error_message="test error",
        )


# ============================================================
# run_pipeline
# ============================================================
class TestRunPipeline:
    """测试 run_pipeline 主流程。"""

    def test_successful_pipeline(self, mock_mysql_client, mock_milvus_client):
        """完整的成功流程。"""
        from app.offline.embedder import EmbeddingResult

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            # 更长的文本产生更多 chunk
            f.write(
                "【药品名称】阿司匹林肠溶片\n\n"
                "【适应症】用于解热镇痛，缓解轻至中度疼痛。如头痛、牙痛、神经痛、"
                "肌肉痛、痛经及关节痛等。也用于感冒等发热疾病的退热。用于预防心脑血管疾病。\n\n"
                "【用法用量】成人一次0.3～0.6g，一日3次，饭后服用。"
                "心脑血管疾病预防：一次50～100mg，一日1次。儿童用量需咨询医师或药师。\n\n"
                "【禁忌】对阿司匹林或其他非甾体抗炎药过敏者禁用。活动性消化性溃疡或出血者禁用。"
                "血友病或血小板减少症患者禁用。妊娠最后三个月孕妇禁用。\n\n"
                "【不良反应】胃肠道反应：恶心、呕吐、上腹部不适或疼痛等。"
                "出血倾向：牙龈出血、鼻出血、皮肤瘀斑等。过敏反应：皮疹、荨麻疹、哮喘等。\n"
            )
            tmp_path = Path(f.name)

        try:
            # 创建动态 embedder mock（返回恰好 1 个向量匹配 1 个 chunk）
            embedder = MagicMock()
            embedder.embed.return_value = EmbeddingResult(
                embeddings=[[0.1] * 1024],
                failed_indices=[],
                total_attempted=1,
                total_succeeded=1,
            )

            with patch("app.offline.pipeline.MySQLClient", return_value=mock_mysql_client), \
                 patch("app.offline.pipeline.MilvusClient", return_value=mock_milvus_client), \
                 patch("app.offline.pipeline.Embedder", return_value=embedder):
                result = run_pipeline(tmp_path)

            assert isinstance(result, PipelineResult)
            assert result.status == "completed"
            assert result.total_chunks > 0
            assert result.indexed_chunks > 0
            assert result.drug_name is not None
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_file_not_found(self, mock_mysql_client, mock_milvus_client, mock_embedder):
        """文件不存在导致失败。"""
        with patch("app.offline.pipeline.MySQLClient", return_value=mock_mysql_client), \
             patch("app.offline.pipeline.MilvusClient", return_value=mock_milvus_client), \
             patch("app.offline.pipeline.Embedder", return_value=mock_embedder):
            result = run_pipeline(Path("/nonexistent/file.txt"))

        assert result.status == "failed"
        assert result.doc_id == -1

    def test_empty_file(self, mock_mysql_client, mock_milvus_client, mock_embedder):
        """空文件导致失败。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            f.write("")
            tmp_path = Path(f.name)

        try:
            with patch("app.offline.pipeline.MySQLClient", return_value=mock_mysql_client), \
                 patch("app.offline.pipeline.MilvusClient", return_value=mock_milvus_client), \
                 patch("app.offline.pipeline.Embedder", return_value=mock_embedder):
                result = run_pipeline(tmp_path)
            assert result.status == "failed"
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_with_explicit_drug_name(self, mock_mysql_client, mock_milvus_client):
        """显式指定药品名。"""
        from app.offline.embedder import EmbeddingResult

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            f.write("【适应症】测试内容足够长以确保能被切分为 chunk " + "X" * 200)
            tmp_path = Path(f.name)

        try:
            embedder = MagicMock()
            embedder.embed.return_value = EmbeddingResult(
                embeddings=[[0.1] * 1024],
                failed_indices=[],
                total_attempted=1,
                total_succeeded=1,
            )

            with patch("app.offline.pipeline.MySQLClient", return_value=mock_mysql_client), \
                 patch("app.offline.pipeline.MilvusClient", return_value=mock_milvus_client), \
                 patch("app.offline.pipeline.Embedder", return_value=embedder):
                result = run_pipeline(tmp_path, drug_name="自定义药名")
            assert result.drug_name == "自定义药名"
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_with_batch_id_and_metadata(self, mock_mysql_client, mock_milvus_client):
        """带 batch_id 和元数据。"""
        from app.offline.embedder import EmbeddingResult

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            f.write("【适应症】测试内容足够长" + "X" * 200)
            tmp_path = Path(f.name)

        try:
            embedder = MagicMock()
            embedder.embed.return_value = EmbeddingResult(
                embeddings=[[0.1] * 1024],
                failed_indices=[],
                total_attempted=1,
                total_succeeded=1,
            )

            with patch("app.offline.pipeline.MySQLClient", return_value=mock_mysql_client), \
                 patch("app.offline.pipeline.MilvusClient", return_value=mock_milvus_client), \
                 patch("app.offline.pipeline.Embedder", return_value=embedder):
                result = run_pipeline(
                    tmp_path,
                    batch_id="custom-batch-id",
                    drug_manufacturer="测试厂家",
                    drug_category="解热镇痛",
                )
            assert result.batch_id == "custom-batch-id"
        finally:
            tmp_path.unlink(missing_ok=True)


# ============================================================
# run_pipeline_batch
# ============================================================
class TestRunPipelineBatch:
    """测试批量处理。"""

    def test_batch_success(self, mock_mysql_client, mock_milvus_client):
        """批量处理多个文件。"""
        from app.offline.embedder import EmbeddingResult

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "药1.txt").write_text("【适应症】测试1" + "X" * 200, encoding="utf-8")
            (tmpdir / "药2.txt").write_text("【适应症】测试2" + "X" * 200, encoding="utf-8")

            embedder = MagicMock()
            embedder.embed.return_value = EmbeddingResult(
                embeddings=[[0.1] * 1024],
                failed_indices=[],
                total_attempted=1,
                total_succeeded=1,
            )

            with patch("app.offline.pipeline.MySQLClient", return_value=mock_mysql_client), \
                 patch("app.offline.pipeline.MilvusClient", return_value=mock_milvus_client), \
                 patch("app.offline.pipeline.Embedder", return_value=embedder):
                results = run_pipeline_batch([
                    tmpdir / "药1.txt",
                    tmpdir / "药2.txt",
                ])
            assert len(results) == 2
            assert all(isinstance(r, PipelineResult) for r in results)

    def test_batch_empty(self, mock_mysql_client, mock_milvus_client):
        """空文件列表。"""
        from app.offline.embedder import EmbeddingResult

        embedder = MagicMock()
        embedder.embed.return_value = EmbeddingResult()

        with patch("app.offline.pipeline.MySQLClient", return_value=mock_mysql_client), \
             patch("app.offline.pipeline.MilvusClient", return_value=mock_milvus_client), \
             patch("app.offline.pipeline.Embedder", return_value=embedder):
            results = run_pipeline_batch([])
        assert results == []
