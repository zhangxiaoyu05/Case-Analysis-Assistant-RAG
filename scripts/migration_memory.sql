-- ============================================================
-- Phase 2: 中期记忆（跨会话用户动态记忆）
-- 在 phase0 的 users/conversations 基础上追加
-- ============================================================

CREATE TABLE IF NOT EXISTS user_memories (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    memory_type VARCHAR(32) NOT NULL COMMENT 'clinical_interest/concern/preference/plan/fact',
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
-- Phase 3: 长期记忆（用户画像 —— 不可变人口属性）
-- ============================================================

CREATE TABLE IF NOT EXISTS user_profiles (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    field_name VARCHAR(64) NOT NULL COMMENT 'name/title/department/hospital/specialty/license_years/guideline_preference/patient_population/common_diseases',
    field_value VARCHAR(500) NOT NULL,
    confidence FLOAT DEFAULT 1.0 COMMENT '提取置信度（0.0~1.0）',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY uk_user_field (user_id, field_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
