-- ============================================================
-- RAG 药品问答系统 v0.2.0 — 用户系统 + 多会话管理
-- Phase 0: 用户表 + 会话表
-- 使用方式: docker exec -i rag-mysql mysql -uroot -p${MYSQL_PASSWORD} rag_pharma < migration_v2.sql
-- ============================================================

USE rag_pharma;

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
