-- ====================================================================
-- LLM Observe - Snowflake Database Schema
-- Comprehensive schema for storing LLM evaluation runs and metrics
-- 
-- NOTE: This schema is optimized for Snowflake's architecture:
-- - Uses clustering keys instead of traditional indexes
-- - Employs views and CTEs for complex analytics
-- - Leverages Snowflake's automatic query optimization
-- ====================================================================

-- Create database and schema (optional - modify as needed)
-- CREATE DATABASE IF NOT EXISTS LLM_OBSERVE;
-- USE DATABASE LLM_OBSERVE;
-- CREATE SCHEMA IF NOT EXISTS EVALUATION;
-- USE SCHEMA EVALUATION;

-- ====================================================================
-- 1. CORE TABLES
-- ====================================================================

-- Table to store LLM evaluation runs
CREATE OR REPLACE TABLE LLM_RUNS (
    RUN_ID VARCHAR(100) PRIMARY KEY,
    TIMESTAMP TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
    MODEL VARCHAR(100) NOT NULL,
    PROMPT TEXT NOT NULL,
    RESPONSE TEXT,
    EXECUTION_TIME_MS FLOAT,
    TOKEN_COUNT INTEGER,
    USER_ID VARCHAR(100) DEFAULT CURRENT_USER(),
    SESSION_ID VARCHAR(100),
    CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Table to store detailed metrics for each run
CREATE OR REPLACE TABLE LLM_METRICS (
    METRIC_ID VARCHAR(100) DEFAULT UUID_STRING(),
    RUN_ID VARCHAR(100) NOT NULL,
    METRIC_NAME VARCHAR(100) NOT NULL,
    METRIC_VALUE FLOAT NOT NULL,
    METRIC_THRESHOLD FLOAT,
    PASSED_THRESHOLD BOOLEAN,
    METRIC_CATEGORY VARCHAR(50),
    METRIC_DESCRIPTION TEXT,
    CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
    
    PRIMARY KEY (METRIC_ID),
    FOREIGN KEY (RUN_ID) REFERENCES LLM_RUNS(RUN_ID)
);

-- Table to store run context and metadata
CREATE OR REPLACE TABLE LLM_RUN_CONTEXT (
    RUN_ID VARCHAR(100) PRIMARY KEY,
    TEMPERATURE FLOAT,
    MAX_TOKENS INTEGER,
    TOP_P FLOAT,
    FREQUENCY_PENALTY FLOAT,
    PRESENCE_PENALTY FLOAT,
    SYSTEM_PROMPT TEXT,
    USER_METADATA VARIANT,
    TAGS ARRAY,
    ENVIRONMENT VARCHAR(50) DEFAULT 'PRODUCTION',
    
    FOREIGN KEY (RUN_ID) REFERENCES LLM_RUNS(RUN_ID)
);

-- ====================================================================
-- 2. CLUSTERING KEYS FOR PERFORMANCE (Snowflake-specific optimization)
-- ====================================================================

-- Add clustering keys for better query performance
-- Note: Clustering keys replace traditional indexes in Snowflake
ALTER TABLE LLM_RUNS CLUSTER BY (TIMESTAMP, MODEL);
ALTER TABLE LLM_METRICS CLUSTER BY (RUN_ID, METRIC_NAME, CREATED_AT);

-- ====================================================================
-- 3. VIEWS FOR EASY QUERYING
-- ====================================================================

-- Comprehensive view joining runs with their metrics
CREATE OR REPLACE VIEW VW_LLM_EVALUATION_SUMMARY AS
SELECT 
    r.RUN_ID,
    r.TIMESTAMP,
    r.MODEL,
    r.PROMPT,
    r.RESPONSE,
    r.EXECUTION_TIME_MS,
    r.TOKEN_COUNT,
    r.USER_ID,
    r.SESSION_ID,
    
    -- Aggregate metrics
    COUNT(m.METRIC_ID) as TOTAL_METRICS,
    SUM(CASE WHEN m.PASSED_THRESHOLD THEN 1 ELSE 0 END) as PASSED_METRICS,
    ROUND(SUM(CASE WHEN m.PASSED_THRESHOLD THEN 1 ELSE 0 END) * 100.0 / COUNT(m.METRIC_ID), 2) as PASS_RATE_PCT,
    
    -- Key metrics (pivot style)
    MAX(CASE WHEN m.METRIC_NAME = 'bertscore_f1' THEN m.METRIC_VALUE END) as BERTSCORE_F1,
    MAX(CASE WHEN m.METRIC_NAME = 'faithfulness' THEN m.METRIC_VALUE END) as FAITHFULNESS,
    MAX(CASE WHEN m.METRIC_NAME = 'hallucination_rate' THEN m.METRIC_VALUE END) as HALLUCINATION_RATE,
    MAX(CASE WHEN m.METRIC_NAME = 'citation_accuracy' THEN m.METRIC_VALUE END) as CITATION_ACCURACY,
    MAX(CASE WHEN m.METRIC_NAME = 'harm_detection' THEN m.METRIC_VALUE END) as HARM_DETECTION,
    MAX(CASE WHEN m.METRIC_NAME = 'bias_score' THEN m.METRIC_VALUE END) as BIAS_SCORE,
    
    r.CREATED_AT
FROM LLM_RUNS r
LEFT JOIN LLM_METRICS m ON r.RUN_ID = m.RUN_ID
GROUP BY r.RUN_ID, r.TIMESTAMP, r.MODEL, r.PROMPT, r.RESPONSE, 
         r.EXECUTION_TIME_MS, r.TOKEN_COUNT, r.USER_ID, r.SESSION_ID, r.CREATED_AT;

-- Recent runs view for dashboard
CREATE OR REPLACE VIEW VW_RECENT_LLM_RUNS AS
SELECT *
FROM VW_LLM_EVALUATION_SUMMARY
ORDER BY TIMESTAMP DESC
LIMIT 100;

-- Metrics trend view
CREATE OR REPLACE VIEW VW_LLM_METRICS_TRENDS AS
SELECT 
    m.METRIC_NAME,
    r.MODEL,
    r.TIMESTAMP,
    m.METRIC_VALUE,
    m.PASSED_THRESHOLD,
    LAG(m.METRIC_VALUE) OVER (PARTITION BY m.METRIC_NAME, r.MODEL ORDER BY r.TIMESTAMP) as PREVIOUS_VALUE,
    m.METRIC_VALUE - LAG(m.METRIC_VALUE) OVER (PARTITION BY m.METRIC_NAME, r.MODEL ORDER BY r.TIMESTAMP) as VALUE_CHANGE,
    ROW_NUMBER() OVER (PARTITION BY m.METRIC_NAME, r.MODEL ORDER BY r.TIMESTAMP DESC) as RECENCY_RANK
FROM LLM_METRICS m
JOIN LLM_RUNS r ON m.RUN_ID = r.RUN_ID
ORDER BY r.TIMESTAMP DESC;

-- ====================================================================
-- 4. STORED PROCEDURES FOR DATA MANAGEMENT
-- ====================================================================

-- Procedure to clean up old data
CREATE OR REPLACE PROCEDURE SP_CLEANUP_OLD_RUNS(RETENTION_DAYS INTEGER)
RETURNS STRING
LANGUAGE SQL
AS
$$
DECLARE
    deleted_runs INTEGER;
    deleted_metrics INTEGER;
BEGIN
    -- Delete old metrics first (foreign key constraint)
    DELETE FROM LLM_METRICS 
    WHERE RUN_ID IN (
        SELECT RUN_ID FROM LLM_RUNS 
        WHERE CREATED_AT < DATEADD('DAY', -RETENTION_DAYS, CURRENT_TIMESTAMP())
    );
    
    deleted_metrics := SQLROWCOUNT;
    
    -- Delete old run context
    DELETE FROM LLM_RUN_CONTEXT
    WHERE RUN_ID IN (
        SELECT RUN_ID FROM LLM_RUNS 
        WHERE CREATED_AT < DATEADD('DAY', -RETENTION_DAYS, CURRENT_TIMESTAMP())
    );
    
    -- Delete old runs
    DELETE FROM LLM_RUNS 
    WHERE CREATED_AT < DATEADD('DAY', -RETENTION_DAYS, CURRENT_TIMESTAMP());
    
    deleted_runs := SQLROWCOUNT;
    
    RETURN 'Cleanup completed: ' || deleted_runs || ' runs and ' || deleted_metrics || ' metrics deleted';
END;
$$;

-- View to get model performance summary (replacing problematic function)
CREATE OR REPLACE VIEW VW_MODEL_PERFORMANCE AS
WITH ranked_metrics AS (
    SELECT 
        r.MODEL,
        m.METRIC_NAME,
        m.METRIC_VALUE,
        r.TIMESTAMP,
        ROW_NUMBER() OVER (PARTITION BY r.MODEL, m.METRIC_NAME ORDER BY r.TIMESTAMP DESC) as rn_latest,
        ROW_NUMBER() OVER (PARTITION BY r.MODEL, m.METRIC_NAME ORDER BY r.TIMESTAMP ASC) as rn_first
    FROM LLM_METRICS m
    JOIN LLM_RUNS r ON m.RUN_ID = r.RUN_ID
),
latest_values AS (
    SELECT MODEL, METRIC_NAME, METRIC_VALUE as LATEST_VALUE
    FROM ranked_metrics 
    WHERE rn_latest = 1
),
first_values AS (
    SELECT MODEL, METRIC_NAME, METRIC_VALUE as FIRST_VALUE
    FROM ranked_metrics 
    WHERE rn_first = 1
)
SELECT 
    r.MODEL,
    m.METRIC_NAME,
    AVG(m.METRIC_VALUE) as AVG_VALUE,
    MIN(m.METRIC_VALUE) as MIN_VALUE,
    MAX(m.METRIC_VALUE) as MAX_VALUE,
    COUNT(*) as SAMPLE_COUNT,
    l.LATEST_VALUE,
    CASE 
        WHEN l.LATEST_VALUE > f.FIRST_VALUE THEN 'IMPROVING'
        WHEN l.LATEST_VALUE < f.FIRST_VALUE THEN 'DECLINING'
        ELSE 'STABLE'
    END as TREND
FROM LLM_METRICS m
JOIN LLM_RUNS r ON m.RUN_ID = r.RUN_ID
JOIN latest_values l ON r.MODEL = l.MODEL AND m.METRIC_NAME = l.METRIC_NAME
JOIN first_values f ON r.MODEL = f.MODEL AND m.METRIC_NAME = f.METRIC_NAME
GROUP BY r.MODEL, m.METRIC_NAME, l.LATEST_VALUE, f.FIRST_VALUE;

-- ====================================================================
-- 5. SAMPLE DATA AND VALIDATION
-- ====================================================================

-- Grant appropriate permissions (modify as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON LLM_RUNS TO ROLE <your_role>;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON LLM_METRICS TO ROLE <your_role>;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON LLM_RUN_CONTEXT TO ROLE <your_role>;
-- GRANT SELECT ON ALL VIEWS IN SCHEMA EVALUATION TO ROLE <your_role>;
-- GRANT USAGE ON ALL PROCEDURES IN SCHEMA EVALUATION TO ROLE <your_role>;

-- Validation query to ensure tables are created correctly
SELECT 
    TABLE_NAME,
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
  AND TABLE_NAME IN ('LLM_RUNS', 'LLM_METRICS', 'LLM_RUN_CONTEXT')
ORDER BY TABLE_NAME, ORDINAL_POSITION;

-- Display created objects
SHOW TABLES;
SHOW VIEWS;
SHOW PROCEDURES;

-- Example usage of model performance view
-- (replaces the previous FN_GET_MODEL_PERFORMANCE function)
-- SELECT * FROM VW_MODEL_PERFORMANCE WHERE MODEL = 'llama3-8b';

-- Success message
SELECT 'LLM Observe Snowflake schema created successfully!' as STATUS; 