"""
知识库管理路由

POST   /api/v1/knowledge/upload        - 上传文档文件并触发离线入库流程
GET    /api/v1/knowledge/status/{id}    - 查询入库批次状态
GET    /api/v1/knowledge/drugs          - 列出已入库的药品
DELETE /api/v1/knowledge/drug/{id}      - 删除指定药品
"""

import asyncio
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel, Field

from app.config import PROJECT_ROOT, config as _cfg
from app.offline.pipeline import run_pipeline

router = APIRouter()

# 上传文件暂存目录
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 批次文件名映射（batch_id → filename），仅用于状态查询接口补充显示
# 核心状态数据存储在 MySQL index_records 表中
_batch_filenames: dict[str, str] = {}


# ============================================================
# Pydantic 模型
# ============================================================
class UploadResponse(BaseModel):
    """上传响应"""
    batch_id: str = Field(description="入库批次 ID")
    filename: str = Field(description="上传的文件名")
    drug_name: str = Field(description="识别的药品名称")
    total_chunks: int = Field(default=0, description="切分后的文本块总数")
    indexed_chunks: int = Field(default=0, description="成功入库的块数")
    status: str = Field(default="completed", description="批次状态")
    message: str = Field(default="", description="处理消息")


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


# ============================================================
# POST /api/v1/knowledge/upload
# ============================================================
@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="上传文档构建知识库",
    description="上传药品说明书文件（支持 PDF/DOCX/TXT），自动执行离线流程：加载→清洗→切分→嵌入→入库（MySQL + Milvus）。",
)
async def upload_document(
    file: UploadFile = File(..., description="药品说明书文件"),
    drug_name: Optional[str] = Form(None, description="药品名称（不填则从文件名推断）"),
    desensitize: bool = Form(False, description="是否启用 LLM 脱敏"),
    overwrite: bool = Form(False, description="当药品已存在时是否覆盖旧数据"),
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

    # 3. 提前检查药品是否已存在（快速失败，避免走完整 pipeline 才发现冲突）
    if not overwrite and drug_name:
        try:
            import pymysql
            cfg_params = _cfg.get_mysql_connection()
            conn = pymysql.connect(**cfg_params)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS cnt FROM drug_raw_docs WHERE drug_name = %s", (drug_name,))
            row = cursor.fetchone()
            conn.close()
            if row and row[0] > 0:
                raise HTTPException(
                    status_code=409,
                    detail=f"药品 '{drug_name}' 已存在于知识库中。如需覆盖，请设置 overwrite=true",
                )
        except HTTPException:
            raise
        except Exception:
            pass  # MySQL 不可用时跳过预检，让 pipeline 自己处理

    # 4. 记录文件名（供状态查询使用）
    _batch_filenames[batch_id] = filename

    # 5. 运行离线流程（同步阻塞 → asyncio.to_thread）
    try:
        result = await asyncio.to_thread(
            run_pipeline,
            file_path=str(save_path),
            drug_name=drug_name,
            desensitize=desensitize,
            overwrite=overwrite,
        )

        # 提取结果（PipelineResult 是 dataclass，非 dict）
        inferred_name = result.drug_name or drug_name or "未知"

        # 处理 skipped 状态（药品已存在且未覆盖）
        if result.status == "skipped":
            logger.info(f"[{batch_id}] 药品已存在，跳过: {inferred_name}")
            return UploadResponse(
                batch_id=batch_id,
                filename=filename,
                drug_name=inferred_name,
                total_chunks=0,
                indexed_chunks=0,
                status="skipped",
                message=result.error_message or "药品已存在，未覆盖",
            )

        logger.info(
            f"[{batch_id}] 入库完成: drug={inferred_name}, "
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
# DELETE /api/v1/knowledge/drug/{drug_id}
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
