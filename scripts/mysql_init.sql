-- ============================================================
-- RAG 药品问答系统 - MySQL 初始化脚本
-- 此脚本在 MySQL 容器首次启动时自动执行
-- 用途: 建库建表 + 全文索引（供 BM25 检索使用）
-- ============================================================

-- 创建数据库（如 docker-compose 中已指定则此行可省略）
CREATE DATABASE IF NOT EXISTS rag_pharma
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE rag_pharma;

-- ============================================================
-- 表 1: 药品原始文档表 (drug_raw_docs)
-- 存储原始药品说明书全文（PDF 解析后的完整文本）
-- ============================================================
CREATE TABLE IF NOT EXISTS drug_raw_docs (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键',
    drug_name VARCHAR(200) NOT NULL COMMENT '药品名称',
    drug_manufacturer VARCHAR(200) COMMENT '生产厂家',
    drug_category VARCHAR(50) COMMENT '药品分类（处方药/OTC）',
    raw_content LONGTEXT NOT NULL COMMENT '原始说明书完整文本',
    source_file VARCHAR(500) COMMENT '来源文件路径',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '入库时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_drug_name (drug_name),
    INDEX idx_category (drug_category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='药品原始文档表';

-- ============================================================
-- 表 2: 药品切分文本块表 (drug_chunks)
-- 存储切分后的文本块（供 BM25 检索 + 来源引用）
-- ============================================================
CREATE TABLE IF NOT EXISTS drug_chunks (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键',
    doc_id INT NOT NULL COMMENT '关联原始文档 ID',
    drug_name VARCHAR(200) NOT NULL COMMENT '药品名称（冗余存储，便于按药品过滤）',
    section VARCHAR(50) COMMENT '所属章节（通用名/适应症/用法用量/禁忌/注意事项/不良反应等）',
    chunk_index INT NOT NULL COMMENT '在该文档中的顺序编号',
    chunk_text TEXT NOT NULL COMMENT '切分后的文本块内容',
    char_count INT COMMENT '字符数（用于质量评估）',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '入库时间',
    FOREIGN KEY (doc_id) REFERENCES drug_raw_docs(id) ON DELETE CASCADE,
    INDEX idx_drug_name (drug_name),
    INDEX idx_doc_id (doc_id),
    INDEX idx_section (section),
    -- BM25 全文索引（MySQL 8.0 内置）
    FULLTEXT INDEX ft_chunk_text (chunk_text) WITH PARSER ngram
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='药品切分文本块表（BM25 检索用）';

-- ============================================================
-- 表 3: 药品元数据表 (drug_metadata)
-- 存储药品的结构化属性（药品名称、规格、厂商等）
-- 便于快速过滤和聚合查询
-- ============================================================
CREATE TABLE IF NOT EXISTS drug_metadata (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键',
    drug_name VARCHAR(200) NOT NULL UNIQUE COMMENT '药品名称（唯一）',
    generic_name VARCHAR(200) COMMENT '通用名',
    brand_name VARCHAR(200) COMMENT '商品名/品牌名',
    manufacturer VARCHAR(200) COMMENT '生产厂家',
    specification VARCHAR(200) COMMENT '规格',
    dosage_form VARCHAR(50) COMMENT '剂型（片剂/胶囊/注射液等）',
    category VARCHAR(50) COMMENT '药品分类',
    approval_number VARCHAR(100) COMMENT '批准文号',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_generic_name (generic_name),
    INDEX idx_manufacturer (manufacturer)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='药品元数据表';

-- ============================================================
-- 表 4: 索引记录表 (index_records)
-- 记录离线流程中每次向量化索引的批次信息
-- 便于排查问题和增量更新
-- ============================================================
CREATE TABLE IF NOT EXISTS index_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    batch_id VARCHAR(100) NOT NULL COMMENT '批次 ID（UUID）',
    doc_id INT COMMENT '关联文档 ID',
    drug_name VARCHAR(200) COMMENT '药品名称',
    total_chunks INT COMMENT '本次处理的文本块总数',
    indexed_chunks INT COMMENT '成功索引的块数',
    failed_chunks INT COMMENT '失败的块数',
    index_status VARCHAR(20) DEFAULT 'pending' COMMENT '状态: pending/running/completed/failed',
    error_message TEXT COMMENT '错误信息',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP NULL COMMENT '完成时间',
    INDEX idx_batch_id (batch_id),
    INDEX idx_status (index_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='索引批次记录表';

-- ============================================================
-- 表 5: 用户表 (users)
-- 简单登录系统，密码 ≥4 字符即可
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '用户 ID',
    username VARCHAR(64) NOT NULL UNIQUE COMMENT '用户名（中文/英文/数字/下划线，2-30 字符）',
    password_hash VARCHAR(255) NOT NULL COMMENT '密码哈希（PBKDF2-HMAC-SHA256）',
    display_name VARCHAR(100) COMMENT '显示名称（可选）',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '注册时间',
    last_login_at TIMESTAMP NULL COMMENT '最后登录时间',
    INDEX idx_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='用户表';

-- ============================================================
-- 表 6: 会话表 (conversations)
-- 每个对话窗口一条记录，绑定用户
-- ============================================================
CREATE TABLE IF NOT EXISTS conversations (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '会话记录 ID',
    user_id BIGINT NOT NULL COMMENT '所属用户 ID',
    session_id VARCHAR(32) NOT NULL UNIQUE COMMENT '会话标识（UUID hex 16 位）',
    title VARCHAR(100) COMMENT 'LLM 自动生成的对话标题（≤15 字）',
    is_active BOOLEAN DEFAULT TRUE COMMENT '是否活跃（删除后标记为 false）',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后更新时间',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_active (user_id, is_active),
    INDEX idx_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='会话表（每个对话窗口一条记录）';

-- ============================================================
-- 表 7: 中期记忆表 (user_memories)
-- 跨会话用户动态记忆（药品关注/担忧顾虑/偏好/计划/事实）
-- ============================================================
CREATE TABLE IF NOT EXISTS user_memories (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    memory_type VARCHAR(32) NOT NULL COMMENT 'drug_interest/concern/preference/plan/fact',
    content VARCHAR(500) NOT NULL,
    keywords VARCHAR(300),
    importance_score FLOAT DEFAULT 1.0,
    source_session_id VARCHAR(32),
    access_count INT DEFAULT 0,
    last_accessed_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_score (user_id, importance_score DESC),
    INDEX idx_user_type (user_id, memory_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- 表 8: 长期记忆表 (user_profiles)
-- 用户画像（不可变人口属性，EAV 模式）
-- ============================================================
CREATE TABLE IF NOT EXISTS user_profiles (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    field_name VARCHAR(64) NOT NULL COMMENT 'name/age/gender/birthday/medical_history/allergies/current_medications/pregnancy_status/occupation',
    field_value VARCHAR(500) NOT NULL,
    confidence FLOAT DEFAULT 1.0 COMMENT '提取置信度（0.0~1.0）',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY uk_user_field (user_id, field_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
