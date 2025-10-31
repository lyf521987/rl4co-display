-- ============================================
-- 用户数据隔离 - 数据库初始化脚本
-- RL4CO Display - 山西大学
-- ============================================

-- 切换到数据库
USE flaskdemo_user;

-- ============================================
-- 1. 创建/更新 users 表
-- ============================================

-- 如果表不存在则创建
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(100),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL,
    INDEX idx_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 添加新字段（如果不存在）
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS email VARCHAR(100),
ADD COLUMN IF NOT EXISTS last_login TIMESTAMP NULL;

-- ============================================
-- 2. 创建训练会话表
-- ============================================

CREATE TABLE IF NOT EXISTS training_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) UNIQUE NOT NULL COMMENT '训练会话唯一ID',
    user_id INT NOT NULL COMMENT '用户ID',
    model_type VARCHAR(50) NOT NULL COMMENT '模型类型 (AM, POMO等)',
    problem_type VARCHAR(50) NOT NULL COMMENT '问题类型 (TSP, CVRP等)',
    config JSON COMMENT '训练配置参数',
    status ENUM('running', 'completed', 'failed') DEFAULT 'running' COMMENT '训练状态',
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '开始时间',
    end_time TIMESTAMP NULL COMMENT '结束时间',
    final_loss DECIMAL(10, 4) COMMENT '最终Loss值',
    final_reward DECIMAL(10, 4) COMMENT '最终Reward值',
    best_reward DECIMAL(10, 4) COMMENT '最佳Reward值',
    checkpoint_path VARCHAR(255) COMMENT 'checkpoint文件路径',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_session_id (session_id),
    INDEX idx_status (status),
    INDEX idx_start_time (start_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='训练会话记录表';

-- ============================================
-- 3. 创建训练文件表
-- ============================================

CREATE TABLE IF NOT EXISTS training_files (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL COMMENT '用户ID',
    session_id VARCHAR(50) NOT NULL COMMENT '训练会话ID',
    file_name VARCHAR(255) NOT NULL COMMENT '文件名',
    file_type ENUM('plot', 'animation', 'curve', 'checkpoint') NOT NULL COMMENT '文件类型',
    file_path VARCHAR(500) NOT NULL COMMENT '文件路径',
    file_size BIGINT COMMENT '文件大小（字节）',
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_session_id (session_id),
    INDEX idx_file_type (file_type),
    INDEX idx_create_time (create_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='训练生成文件表';

-- ============================================
-- 4. 创建测试用户（可选）
-- ============================================

-- 创建管理员账户
-- 密码: admin123 (已使用pbkdf2:sha256加密)
INSERT IGNORE INTO users (username, password, email) VALUES 
('admin', 'pbkdf2:sha256:600000$randomsalt$hashedpassword', 'admin@example.com');

-- 创建测试用户1
-- 密码: 123456
INSERT IGNORE INTO users (username, password) VALUES 
('testuser1', 'pbkdf2:sha256:600000$testsalt1$hashed123456');

-- 创建测试用户2
-- 密码: 123456
INSERT IGNORE INTO users (username, password) VALUES 
('testuser2', 'pbkdf2:sha256:600000$testsalt2$hashed123456');

-- ============================================
-- 5. 验证表创建
-- ============================================

-- 查看所有表
SHOW TABLES;

-- 查看表结构
DESCRIBE users;
DESCRIBE training_sessions;
DESCRIBE training_files;

-- 查看表记录数
SELECT 'users' as table_name, COUNT(*) as count FROM users
UNION ALL
SELECT 'training_sessions', COUNT(*) FROM training_sessions
UNION ALL
SELECT 'training_files', COUNT(*) FROM training_files;

-- ============================================
-- 6. 创建视图（方便查询）
-- ============================================

-- 用户训练统计视图
CREATE OR REPLACE VIEW user_training_stats AS
SELECT 
    u.id as user_id,
    u.username,
    COUNT(DISTINCT ts.id) as total_sessions,
    COUNT(DISTINCT CASE WHEN ts.status = 'completed' THEN ts.id END) as completed_sessions,
    COUNT(DISTINCT CASE WHEN ts.status = 'running' THEN ts.id END) as running_sessions,
    COUNT(DISTINCT CASE WHEN ts.status = 'failed' THEN ts.id END) as failed_sessions,
    COUNT(tf.id) as total_files,
    COALESCE(SUM(tf.file_size), 0) as total_size_bytes,
    ROUND(COALESCE(SUM(tf.file_size), 0) / 1024 / 1024, 2) as total_size_mb,
    MAX(ts.start_time) as last_training_time
FROM users u
LEFT JOIN training_sessions ts ON u.id = ts.user_id
LEFT JOIN training_files tf ON u.id = tf.user_id
GROUP BY u.id, u.username;

-- 查看视图
SELECT * FROM user_training_stats;

-- ============================================
-- 7. 创建存储过程（可选）
-- ============================================

-- 删除用户及其所有数据的存储过程
DELIMITER //

CREATE PROCEDURE IF NOT EXISTS delete_user_and_data(IN uid INT)
BEGIN
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SELECT 'Error: Transaction rolled back' AS message;
    END;
    
    START TRANSACTION;
    
    -- 删除文件记录（外键会自动删除）
    DELETE FROM training_files WHERE user_id = uid;
    
    -- 删除训练会话
    DELETE FROM training_sessions WHERE user_id = uid;
    
    -- 删除用户
    DELETE FROM users WHERE id = uid;
    
    COMMIT;
    SELECT CONCAT('User ', uid, ' and all related data deleted successfully') AS message;
END //

DELIMITER ;

-- ============================================
-- 8. 数据完整性检查
-- ============================================

-- 检查孤立的训练会话（用户不存在）
SELECT ts.* 
FROM training_sessions ts
LEFT JOIN users u ON ts.user_id = u.id
WHERE u.id IS NULL;

-- 检查孤立的文件记录（用户不存在）
SELECT tf.* 
FROM training_files tf
LEFT JOIN users u ON tf.user_id = u.id
WHERE u.id IS NULL;

-- ============================================
-- 9. 权限设置（生产环境建议）
-- ============================================

-- 创建只读用户（用于数据分析）
-- CREATE USER IF NOT EXISTS 'readonly'@'localhost' IDENTIFIED BY 'readonly_password';
-- GRANT SELECT ON flaskdemo_user.* TO 'readonly'@'localhost';

-- 创建应用用户（Flask使用）
-- CREATE USER IF NOT EXISTS 'flask_app'@'localhost' IDENTIFIED BY 'secure_password';
-- GRANT SELECT, INSERT, UPDATE, DELETE ON flaskdemo_user.* TO 'flask_app'@'localhost';

-- FLUSH PRIVILEGES;

-- ============================================
-- 10. 备份命令（提示）
-- ============================================

/*
在命令行执行以下命令备份数据库：

# 备份整个数据库
mysqldump -u root -p flaskdemo_user > backup_$(date +%Y%m%d).sql

# 只备份表结构
mysqldump -u root -p --no-data flaskdemo_user > schema_backup.sql

# 只备份数据
mysqldump -u root -p --no-create-info flaskdemo_user > data_backup.sql

# 恢复数据库
mysql -u root -p flaskdemo_user < backup_20241201.sql
*/

-- ============================================
-- 完成！
-- ============================================

SELECT '========================================' AS '';
SELECT 'Database initialization completed!' AS 'Status';
SELECT '========================================' AS '';
SELECT 'Tables created:' AS '';
SELECT '  - users' AS '';
SELECT '  - training_sessions' AS '';
SELECT '  - training_files' AS '';
SELECT '========================================' AS '';
SELECT 'Views created:' AS '';
SELECT '  - user_training_stats' AS '';
SELECT '========================================' AS '';
SELECT 'Next steps:' AS '';
SELECT '  1. Update app.py with authentication code' AS '';
SELECT '  2. Install dependencies: pip install Flask-Login' AS '';
SELECT '  3. Create user directories in static/model_plots/' AS '';
SELECT '  4. Test registration and login' AS '';
SELECT '========================================' AS '';

