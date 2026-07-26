-- ============================================================
-- RAG 临床病例分析助手 v1.0.0 — 增量迁移脚本
--
-- 用途: 对已有 rag_pharma 数据库进行增量迁移，
--       新增 6 张表 + index_records 扩展。
--
-- 适用场景:
--   已经在运行 v0.5.0 的系统，需要升级到 v1.0.0
--
-- 使用方式:
--   mysql -u root -p rag_pharma < scripts/migration_v3.sql
--
-- 安全说明:
--   所有 CREATE TABLE 使用 IF NOT EXISTS，
--   所有 ALTER TABLE 使用 IF NOT EXISTS (MySQL 8.0+)
--   对已有数据无破坏性影响。
-- ============================================================

USE rag_pharma;

-- ============================================================
-- 1. index_records 扩展 — 新增 source_type 列
-- ============================================================
-- 检查列是否已存在，不存在则添加
SET @sql_ir = IF(
    (SELECT COUNT(*) FROM information_schema.COLUMNS
     WHERE TABLE_SCHEMA = 'rag_pharma'
       AND TABLE_NAME = 'index_records'
       AND COLUMN_NAME = 'source_type') = 0,
    'ALTER TABLE index_records ADD COLUMN source_type VARCHAR(20) DEFAULT ''drug'' COMMENT ''来源类型：drug/disease/guideline/literature''',
    'SELECT ''index_records.source_type 列已存在，跳过'' AS msg'
);
PREPARE stmt_ir FROM @sql_ir;
EXECUTE stmt_ir;
DEALLOCATE PREPARE stmt_ir;

-- ============================================================
-- 2. 疾病知识表
-- ============================================================

-- 2.1 疾病原始文档表
CREATE TABLE IF NOT EXISTS disease_raw_docs (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键',
    disease_name VARCHAR(200) NOT NULL COMMENT '疾病名称',
    disease_category VARCHAR(100) COMMENT '疾病分类（心血管/呼吸/内分泌/神经/消化...）',
    department VARCHAR(100) COMMENT '所属科室',
    raw_content LONGTEXT NOT NULL COMMENT '疾病知识全文',
    source_type VARCHAR(50) DEFAULT 'textbook' COMMENT '来源类型：textbook/uptodate/msd_manual/pubmed',
    source_file VARCHAR(500) COMMENT '来源文件路径',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_disease_name (disease_name),
    INDEX idx_category (disease_category),
    INDEX idx_source_type (source_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='疾病知识原始文档表';

-- 2.2 疾病切分文本块表
CREATE TABLE IF NOT EXISTS disease_chunks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    doc_id INT NOT NULL COMMENT '关联 disease_raw_docs.id',
    disease_name VARCHAR(200) NOT NULL COMMENT '疾病名称（冗余存储，便于过滤）',
    section VARCHAR(100) COMMENT '章节（病因/病理/流行病学/临床表现/诊断/鉴别诊断/治疗/预后/预防）',
    chunk_index INT NOT NULL COMMENT '在该文档中的顺序编号',
    chunk_text TEXT NOT NULL COMMENT '切分后的文本块内容',
    char_count INT COMMENT '字符数',
    source_type VARCHAR(50),
    evidence_level VARCHAR(20) COMMENT '证据级别（IA/IB/IIA/IIB/III/IV — 牛津循证医学中心分级）',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES disease_raw_docs(id) ON DELETE CASCADE,
    INDEX idx_disease_name (disease_name),
    INDEX idx_section (section),
    INDEX idx_evidence (evidence_level),
    FULLTEXT INDEX ft_chunk_text (chunk_text) WITH PARSER ngram
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='疾病知识切分文本块表（BM25 检索用）';

-- ============================================================
-- 3. 临床指南表
-- ============================================================

-- 3.1 临床指南原始文档表
CREATE TABLE IF NOT EXISTS guideline_raw_docs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    guideline_title VARCHAR(500) NOT NULL COMMENT '指南全称',
    issuing_body VARCHAR(300) COMMENT '发布机构（中华医学会/NICE/WHO/ESC/ACC/AHA...）',
    publish_year INT COMMENT '发布年份',
    disease_name VARCHAR(200) COMMENT '相关疾病',
    department VARCHAR(100) COMMENT '科室',
    raw_content LONGTEXT NOT NULL,
    source_file VARCHAR(500),
    url VARCHAR(500) COMMENT '官方链接',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_disease (disease_name),
    INDEX idx_body (issuing_body),
    INDEX idx_year (publish_year)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='临床指南原始文档表';

-- 3.2 临床指南切分文本块表
CREATE TABLE IF NOT EXISTS guideline_chunks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    doc_id INT NOT NULL,
    guideline_title VARCHAR(500) NOT NULL,
    disease_name VARCHAR(200),
    section VARCHAR(100) COMMENT '章节（推荐意见/证据总结/诊疗路径/诊断标准/治疗方案/参考文献）',
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    char_count INT,
    evidence_level VARCHAR(20) COMMENT '推荐级别（A/B/C — GRADE分级 或 Ⅰ/Ⅱa/Ⅱb/Ⅲ — ACC/AHA分级）',
    recommendation_grade VARCHAR(20) COMMENT '推荐等级（强推荐/弱推荐/建议/可考虑）',
    issuing_body VARCHAR(200),
    publish_year INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES guideline_raw_docs(id) ON DELETE CASCADE,
    INDEX idx_disease (disease_name),
    INDEX idx_evidence (evidence_level),
    INDEX idx_grade (recommendation_grade),
    FULLTEXT INDEX ft_chunk_text (chunk_text) WITH PARSER ngram
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='临床指南切分文本块表（BM25 检索用）';

-- ============================================================
-- 4. 学术文献表
-- ============================================================

-- 4.1 学术文献原始文档表
CREATE TABLE IF NOT EXISTS literature_raw_docs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(500) NOT NULL COMMENT '文献标题',
    authors TEXT COMMENT '作者列表（分号分隔）',
    journal VARCHAR(300) COMMENT '期刊名称',
    publish_year INT,
    doi VARCHAR(200) COMMENT 'DOI',
    pmid VARCHAR(50) COMMENT 'PubMed ID',
    abstract_text LONGTEXT COMMENT '摘要',
    full_text LONGTEXT COMMENT '全文（如有）',
    study_type VARCHAR(100) COMMENT '研究类型（RCT/meta-analysis/systematic_review/cohort/case_control/case_report/case_series/expert_opinion）',
    disease_name VARCHAR(200),
    keywords TEXT COMMENT '关键词（分号分隔）',
    source_file VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_disease (disease_name),
    INDEX idx_study_type (study_type),
    INDEX idx_year (publish_year),
    INDEX idx_pmid (pmid),
    INDEX idx_doi (doi)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='学术文献原始文档表';

-- 4.2 学术文献切分文本块表
CREATE TABLE IF NOT EXISTS literature_chunks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    doc_id INT NOT NULL,
    title VARCHAR(500) NOT NULL,
    disease_name VARCHAR(200),
    section VARCHAR(100) COMMENT '章节（背景/introduction/方法/methods/结果/results/讨论/discussion/结论/conclusion）',
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    char_count INT,
    study_type VARCHAR(100),
    evidence_level VARCHAR(20) COMMENT '牛津证据等级（1a/1b/1c/2a/2b/2c/3a/3b/4/5）',
    publish_year INT,
    doi VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES literature_raw_docs(id) ON DELETE CASCADE,
    INDEX idx_disease (disease_name),
    INDEX idx_study_type (study_type),
    INDEX idx_evidence (evidence_level),
    FULLTEXT INDEX ft_chunk_text (chunk_text) WITH PARSER ngram
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='学术文献切分文本块表（BM25 检索用）';

-- ============================================================
-- 5. 迁移结果确认
-- ============================================================
SELECT 'migration_v3.sql 执行完毕' AS status;
SELECT TABLE_NAME, TABLE_ROWS, TABLE_COMMENT
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = 'rag_pharma'
  AND TABLE_NAME IN ('disease_raw_docs', 'disease_chunks',
                     'guideline_raw_docs', 'guideline_chunks',
                     'literature_raw_docs', 'literature_chunks')
ORDER BY TABLE_NAME;
