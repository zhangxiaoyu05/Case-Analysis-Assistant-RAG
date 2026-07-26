"""
知识库管理路由 (v1.0.0)

POST   /api/v1/knowledge/upload             - 上传文档文件并触发离线入库流程
GET    /api/v1/knowledge/status/{id}        - 查询入库批次状态
GET    /api/v1/knowledge/drugs              - 列出已入库的药品（向后兼容）
GET    /api/v1/knowledge/sources            - 多源知识库概览（v1.0.0 新增）
DELETE /api/v1/knowledge/drug/{id}          - 删除指定药品（向后兼容）
DELETE /api/v1/knowledge/source/{source_type}/{id}  - 通用删除（v1.0.0 新增）
"""

import asyncio
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel, Field

from app.config import PROJECT_ROOT
from app.db.milvus_client import MilvusClient
from app.db.mysql_client import MySQLClient
from app.offline.pipeline import run_pipeline

router = APIRouter()

# 上传文件暂存目录
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 批次文件名映射（batch_id → filename），仅用于状态查询接口补充显示
# 核心状态数据存储在 MySQL index_records 表中
_batch_filenames: dict[str, str] = {}

# v1.0.0: source_type labels
_SOURCE_LABELS = {"drug": "药品", "disease": "疾病", "guideline": "指南", "literature": "文献"}


# ============================================================
# Pydantic 模型
# ============================================================
class UploadResponse(BaseModel):
    """上传响应"""
    batch_id: str = Field(description="入库批次 ID")
    filename: str = Field(description="上传的文件名")
    drug_name: str = Field(description="识别的来源名称")
    total_chunks: int = Field(default=0, description="切分后的文本块总数")
    indexed_chunks: int = Field(default=0, description="成功入库的块数")
    status: str = Field(default="completed", description="批次状态")
    message: str = Field(default="", description="处理消息")
    source_type: str = Field(default="drug", description="来源类型")
    # v1.1.0: 自动分类元数据
    classification_method: Optional[str] = Field(default=None, description="分类方式: llm / rule")
    classification_confidence: Optional[str] = Field(default=None, description="分类置信度: high / medium / low")


class BatchStatus(BaseModel):
    """批次状态响应"""
    batch_id: str
    status: str
    filename: str
    drug_name: str
    total_chunks: int
    indexed_chunks: int
    error: Optional[str] = None


class DrugListItem(BaseModel):
    """已入库药品"""
    id: int
    drug_name: str
    manufacturer: Optional[str] = None
    category: Optional[str] = None
    chunk_count: int
    created_at: str


class DrugListResponse(BaseModel):
    """药品列表响应"""
    drugs: list[DrugListItem]
    total: int


class DeleteResponse(BaseModel):
    """删除响应"""
    drug_id: int
    drug_name: str
    deleted: bool


# v1.0.0: 多源模型
class SourceItem(BaseModel):
    """通用知识库条目"""
    id: int
    title: str
    chunk_count: int = 0
    category: Optional[str] = None
    study_type: Optional[str] = None
    issuing_body: Optional[str] = None
    created_at: Optional[str] = None


class SourcesResponse(BaseModel):
    """多源知识库概览"""
    counts: dict[str, int] = Field(description="各 source_type 的文档数量")
    items: dict[str, list[SourceItem]] = Field(description="各 source_type 的条目列表")


# ============================================================
# 辅助函数
# ============================================================
def _check_source_exists(source_type: str, name: str) -> bool:
    """检查指定 source_type 中是否已存在同名文档。"""
    try:
        mysql = MySQLClient()
        mysql.connect()
        docs = mysql.list_source_docs(source_type, limit=1000)
        mysql.disconnect()
        for doc in docs:
            if doc.get("title", "").strip() == name.strip():
                return True
        return False
    except Exception:
        return False  # MySQL 不可用时跳过预检


def _build_extra_fields(source_type: str, **form_fields) -> dict:
    """根据 source_type 构建 extra_fields 字典。"""
    extra = {}
    if source_type == "disease":
        if form_fields.get("disease_category"):
            extra["disease_category"] = form_fields["disease_category"]
        if form_fields.get("evidence_level"):
            extra["evidence_level"] = form_fields["evidence_level"]
    elif source_type == "guideline":
        if form_fields.get("disease_name"):
            extra["disease_name"] = form_fields["disease_name"]
        if form_fields.get("publish_year"):
            extra["publish_year"] = int(form_fields["publish_year"])
        if form_fields.get("issuing_body"):
            extra["issuing_body"] = form_fields["issuing_body"]
        if form_fields.get("evidence_level"):
            extra["evidence_level"] = form_fields["evidence_level"]
    elif source_type == "literature":
        if form_fields.get("disease_name"):
            extra["disease_name"] = form_fields["disease_name"]
        if form_fields.get("study_type"):
            extra["study_type"] = form_fields["study_type"]
        if form_fields.get("publish_year"):
            extra["publish_year"] = int(form_fields["publish_year"])
        if form_fields.get("evidence_level"):
            extra["evidence_level"] = form_fields["evidence_level"]
        if form_fields.get("doi"):
            extra["doi"] = form_fields["doi"]
        if form_fields.get("authors"):
            extra["authors"] = form_fields["authors"]
        if form_fields.get("journal"):
            extra["journal"] = form_fields["journal"]
    return extra


# ============================================================
# POST /api/v1/knowledge/upload
# ============================================================
@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="上传文档构建知识库",
    description="上传文档文件（支持 PDF/DOCX/TXT），自动执行离线流程。v1.0.0: 支持多源类型（drug/disease/guideline/literature）。",
)
async def upload_document(
    file: UploadFile = File(..., description="文档文件"),
    drug_name: Optional[str] = Form(None, description="来源名称（不填则从文件名推断）"),
    source_type: str = Form("auto", description="来源类型：auto/drug/disease/guideline/literature（auto 为 AI 自动识别）"),
    desensitize: bool = Form(False, description="是否启用 LLM 脱敏"),
    overwrite: bool = Form(False, description="当文档已存在时是否覆盖旧数据"),
    # v1.0.0: source-type-specific fields
    disease_category: Optional[str] = Form(None, description="疾病分类（source_type=disease）"),
    guideline_title: Optional[str] = Form(None, description="指南标题（source_type=guideline）"),
    disease_name: Optional[str] = Form(None, description="相关疾病（source_type=disease/guideline/literature）"),
    publish_year: Optional[str] = Form(None, description="发布年份（source_type=guideline/literature）"),
    issuing_body: Optional[str] = Form(None, description="发布机构（source_type=guideline）"),
    study_type: Optional[str] = Form(None, description="研究类型（source_type=literature）"),
    evidence_level: Optional[str] = Form(None, description="证据级别"),
    doi: Optional[str] = Form(None, description="DOI（source_type=literature）"),
    authors: Optional[str] = Form(None, description="作者（source_type=literature）"),
    journal: Optional[str] = Form(None, description="期刊（source_type=literature）"),
) -> UploadResponse:
    # 1. 校验文件类型
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    allowed = {".pdf", ".docx", ".txt"}
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {ext}，仅支持: {', '.join(allowed)}",
        )

    # 2. 保存上传文件
    batch_id = uuid.uuid4().hex[:12]
    safe_name = f"{batch_id}_{filename}"
    save_path = UPLOAD_DIR / safe_name

    try:
        content = await file.read()
        save_path.write_bytes(content)
        logger.info(f"[{batch_id}] 文件已保存: {save_path} ({len(content)} bytes)")
    except Exception as e:
        logger.error(f"[{batch_id}] 文件保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    # 3. 提前检查是否已存在（快速失败）
    # 注意：auto 模式下跳过预检，因为实际 source_type 由 AI 确定
    if source_type != "auto":
        check_name = drug_name or guideline_title or disease_name
        if not overwrite and check_name:
            if _check_source_exists(source_type, check_name):
                label = _SOURCE_LABELS.get(source_type, "文档")
                raise HTTPException(
                    status_code=409,
                    detail=f"{label} '{check_name}' 已存在于知识库中。如需覆盖，请设置 overwrite=true",
                )

    # 4. 记录文件名（供状态查询使用）
    _batch_filenames[batch_id] = filename

    # 5. 构建 resolve name + extra_fields
    resolved_name = drug_name
    if not resolved_name and source_type == "disease":
        resolved_name = disease_name
    elif not resolved_name and source_type == "guideline":
        resolved_name = guideline_title

    extra_fields = _build_extra_fields(
        source_type,
        disease_category=disease_category,
        guideline_title=guideline_title,
        disease_name=disease_name,
        publish_year=publish_year,
        issuing_body=issuing_body,
        study_type=study_type,
        evidence_level=evidence_level,
        doi=doi,
        authors=authors,
        journal=journal,
    )

    # 6. 运行离线流程
    try:
        result = await asyncio.to_thread(
            run_pipeline,
            file_path=str(save_path),
            drug_name=resolved_name,
            desensitize=desensitize,
            overwrite=overwrite,
            source_type=source_type,
            **extra_fields,
        )

        inferred_name = result.drug_name or resolved_name or "未知"
        # v1.1.0: 使用 pipeline 解析后的 source_type（auto 时由 AI 确定）
        resolved_type = result.resolved_source_type or source_type

        if result.status == "skipped":
            logger.info(f"[{batch_id}] {source_type} 已存在，跳过: {inferred_name}")
            return UploadResponse(
                batch_id=batch_id,
                filename=filename,
                drug_name=inferred_name,
                total_chunks=0,
                indexed_chunks=0,
                status="skipped",
                message=result.error_message or f"{_SOURCE_LABELS.get(resolved_type, '文档')}已存在，未覆盖",
                source_type=resolved_type,
                classification_method=result.classification_method,
                classification_confidence=result.classification_confidence,
            )

        logger.info(
            f"[{batch_id}] 入库完成: source={inferred_name}, type={resolved_type}, "
            f"chunks={result.total_chunks}, indexed={result.indexed_chunks}"
        )

        return UploadResponse(
            batch_id=batch_id,
            filename=filename,
            drug_name=inferred_name,
            total_chunks=result.total_chunks,
            indexed_chunks=result.indexed_chunks,
            status=result.status,
            message="知识库构建完成" if result.status == "completed" else (
                result.error_message or "部分完成"
            ),
            source_type=resolved_type,
            classification_method=result.classification_method,
            classification_confidence=result.classification_confidence,
        )

    except Exception as e:
        logger.error(f"[{batch_id}] 入库失败: {e}")
        raise HTTPException(status_code=500, detail=f"知识库构建失败: {e}")


# ============================================================
# GET /api/v1/knowledge/status/{batch_id}
# ============================================================
@router.get(
    "/status/{batch_id}",
    response_model=BatchStatus,
    summary="查询批次状态",
    description="根据 batch_id 查询离线入库的进度和结果。",
)
async def get_batch_status(batch_id: str) -> BatchStatus:
    # 从 MySQL index_records 表读取批次状态（持久化，重启不丢失）
    try:
        from app.db.mysql_client import MySQLClient
        mysql = MySQLClient()
        mysql.connect()
        record = mysql.get_index_record(batch_id)
        mysql.disconnect()
    except Exception as e:
        logger.error(f"查询批次状态失败: {e}")
        raise HTTPException(status_code=503, detail=f"数据库查询失败: {e}")

    if not record:
        # 回退：可能在 MySQL 写入前就被查询
        filename = _batch_filenames.get(batch_id, "")
        if filename:
            return BatchStatus(
                batch_id=batch_id,
                status="running",
                filename=filename,
                drug_name="",
                total_chunks=0,
                indexed_chunks=0,
                error=None,
            )
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")

    filename = _batch_filenames.get(batch_id, "")

    return BatchStatus(
        batch_id=batch_id,
        status=record.get("index_status", "unknown"),
        filename=filename,
        drug_name=record.get("drug_name", ""),
        total_chunks=record.get("total_chunks", 0) or 0,
        indexed_chunks=record.get("indexed_chunks", 0) or 0,
        error=record.get("error_message"),
    )


# ============================================================
# GET /api/v1/knowledge/drugs
# ============================================================
@router.get(
    "/drugs",
    response_model=DrugListResponse,
    summary="列出已入库药品",
    description="从 MySQL 查询已入库的药品列表（含各药品的文本块数量）。",
)
async def list_drugs() -> DrugListResponse:
    try:
        from app.db.mysql_client import MySQLClient
        mysql = MySQLClient()
        mysql.connect()

        # 查询药品及 chunk 统计（复用 MySQLClient 内部连接）
        sql = """
            SELECT
                r.id,
                r.drug_name,
                r.drug_manufacturer AS manufacturer,
                r.drug_category AS category,
                COUNT(c.id) AS chunk_count,
                r.created_at
            FROM drug_raw_docs r
            LEFT JOIN drug_chunks c ON c.doc_id = r.id
            GROUP BY r.id, r.drug_name, r.drug_manufacturer, r.drug_category, r.created_at
            ORDER BY r.created_at DESC
        """
        with mysql.conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()

        mysql.disconnect()

        drugs = [
            DrugListItem(
                id=row["id"],
                drug_name=row["drug_name"],
                manufacturer=row.get("manufacturer"),
                category=row.get("category"),
                chunk_count=row.get("chunk_count", 0),
                created_at=str(row["created_at"]) if row.get("created_at") else "",
            )
            for row in rows
        ]

        return DrugListResponse(drugs=drugs, total=len(drugs))

    except Exception as e:
        logger.error(f"查询药品列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据库查询失败: {e}")


# ============================================================
# GET /api/v1/knowledge/sources (v1.0.0)
# ============================================================
@router.get(
    "/sources",
    response_model=SourcesResponse,
    summary="多源知识库概览",
    description="返回所有 4 种来源类型的入库数量和条目列表。",
)
async def list_sources() -> SourcesResponse:
    try:
        mysql = MySQLClient()
        mysql.connect()

        counts = mysql.get_source_counts()
        items: dict[str, list[SourceItem]] = {}

        for source_type in ["drug", "disease", "guideline", "literature"]:
            docs = mysql.list_source_docs(source_type, limit=500)
            items[source_type] = [
                SourceItem(
                    id=doc.get("id", 0),
                    title=doc.get("title", ""),
                    chunk_count=doc.get("chunk_count", 0),
                    category=doc.get("category"),
                    study_type=doc.get("study_type"),
                    issuing_body=doc.get("issuing_body"),
                    created_at=str(doc["created_at"]) if doc.get("created_at") else None,
                )
                for doc in docs
            ]

        mysql.disconnect()
        return SourcesResponse(counts=counts, items=items)

    except Exception as e:
        logger.error(f"查询多源知识库失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据库查询失败: {e}")


# ============================================================
# DELETE /api/v1/knowledge/source/{source_type}/{id} (v1.0.0)
# ============================================================
@router.delete(
    "/source/{source_type}/{item_id}",
    response_model=DeleteResponse,
    summary="删除指定来源类型文档",
    description="从 MySQL + Milvus 中删除指定来源类型的文档。",
)
async def delete_source_item(source_type: str, item_id: int) -> DeleteResponse:
    # 校验 source_type
    if source_type not in _SOURCE_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"无效的 source_type: {source_type}。支持: {list(_SOURCE_LABELS.keys())}",
        )

    try:
        mysql = MySQLClient()
        mysql.connect()

        # 查询该条目获取名称
        raw_table = MySQLClient.get_source_raw_table(source_type)
        sql = f"SELECT * FROM {raw_table} WHERE id = %s"
        with mysql.conn.cursor() as cursor:
            cursor.execute(sql, (item_id,))
            raw_doc = cursor.fetchone()

        if not raw_doc:
            mysql.disconnect()
            raise HTTPException(
                status_code=404,
                detail=f"{_SOURCE_LABELS.get(source_type)} ID {item_id} 不存在",
            )

        # 提取名称（不同 source_type 有不同名称字段）
        title_cols = {
            "drug": "drug_name",
            "disease": "disease_name",
            "guideline": "guideline_title",
            "literature": "title",
        }
        title_col = title_cols.get(source_type, "id")
        name = raw_doc.get(title_col, str(item_id))

        # MySQL 删除（级联删除 chunks）
        mysql.delete_by_id_generic(source_type, item_id)
        mysql.disconnect()

        # Milvus 删除
        try:
            milvus = MilvusClient(collection_name=f"{source_type}_chunks")
            milvus.connect()
            milvus.delete_by_source_name(name)
            milvus.disconnect()
        except Exception as e:
            logger.warning(f"Milvus 删除失败（可能 Collection 为空）: {e}")

        logger.info(f"{_SOURCE_LABELS.get(source_type)} '{name}' (ID={item_id}) 已删除")
        return DeleteResponse(drug_id=item_id, drug_name=name, deleted=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")


# ============================================================
# DELETE /api/v1/knowledge/drug/{drug_id} (向后兼容)
# ============================================================
@router.delete(
    "/drug/{drug_id}",
    response_model=DeleteResponse,
    summary="删除指定药品",
    description="从 MySQL + Milvus 中删除指定药品的文档、chunks 和向量。",
)
async def delete_drug(drug_id: int) -> DeleteResponse:
    try:
        from app.db.milvus_client import MilvusClient
        from app.db.mysql_client import MySQLClient

        mysql = MySQLClient()
        mysql.connect()

        # 先查药品名（用于返回 + Milvus 删除）
        raw_doc = mysql.get_raw_doc(drug_id)
        if not raw_doc:
            mysql.disconnect()
            raise HTTPException(status_code=404, detail=f"药品 ID {drug_id} 不存在")

        drug_name = raw_doc["drug_name"]

        # MySQL: 使用封装好的方法删除全部关联数据
        mysql.delete_drug_by_name(drug_name)
        mysql.disconnect()

        # Milvus: 按 drug_name 删除向量
        try:
            milvus = MilvusClient()
            milvus.connect()
            milvus.delete_by_drug_name(drug_name)
            milvus.disconnect()
        except Exception as e:
            logger.warning(f"Milvus 删除失败（可能 Collection 为空）: {e}")

        logger.info(f"药品 '{drug_name}' (ID={drug_id}) 已从知识库中删除")
        return DeleteResponse(drug_id=drug_id, drug_name=drug_name, deleted=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除药品失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")
