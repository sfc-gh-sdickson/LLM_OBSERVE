# Streamlit in Snowflake Deployment Guide

This guide provides step-by-step instructions for deploying the LLM Observe application in Snowflake using Streamlit in Snowflake.

## Prerequisites

### Snowflake Account Requirements
- Snowflake account with Streamlit in Snowflake enabled
- Access to Snowflake Cortex (Foundation Models)
- Sufficient privileges to create databases, schemas, tables, and functions
- Warehouse with adequate compute resources

### Required Privileges
Ensure your role has the following privileges:

```sql
-- Basic Streamlit privileges
GRANT USAGE ON WAREHOUSE <warehouse_name> TO ROLE <role_name>;
GRANT CREATE STREAMLIT ON SCHEMA <schema_name> TO ROLE <role_name>;
GRANT CREATE STAGE ON SCHEMA <schema_name> TO ROLE <role_name>;

-- Database object creation privileges
GRANT CREATE TABLE ON SCHEMA <schema_name> TO ROLE <role_name>;
GRANT CREATE VIEW ON SCHEMA <schema_name> TO ROLE <role_name>;
GRANT CREATE FUNCTION ON SCHEMA <schema_name> TO ROLE <role_name>;

-- Cortex privileges for Foundation Models
GRANT USAGE ON FUNCTION SNOWFLAKE.CORTEX.COMPLETE TO ROLE <role_name>;
GRANT USAGE ON FUNCTION SNOWFLAKE.CORTEX.SENTIMENT TO ROLE <role_name>;
GRANT USAGE ON FUNCTION SNOWFLAKE.CORTEX.TRANSLATE TO ROLE <role_name>;
```

## Step 1: Environment Setup

### 1.1 Create Database and Schema

```sql
-- Create dedicated database for LLM Observe
CREATE DATABASE IF NOT EXISTS LLM_OBSERVE
COMMENT = 'Database for LLM monitoring and evaluation application';

-- Create schema for the application
CREATE SCHEMA IF NOT EXISTS LLM_OBSERVE.MONITORING
COMMENT = 'Schema for LLM monitoring tables, views, and functions';

-- Use the schema
USE SCHEMA LLM_OBSERVE.MONITORING;
```

### 1.2 Create Internal Stage for App Files

```sql
-- Create stage for storing application files
CREATE OR REPLACE STAGE LLM_OBSERVE_APP_STAGE
COMMENT = 'Stage for LLM Observe Streamlit application files';

-- List contents (should be empty initially)
LIST @LLM_OBSERVE_APP_STAGE;
```

## Step 2: Deploy Application Files

### 2.1 Upload Files via Snowflake Web Interface

1. **Access Snowflake Web Interface**
   - Log into your Snowflake account
   - Navigate to "Data" > "Databases"
   - Select `LLM_OBSERVE` > `MONITORING`

2. **Create Streamlit App**
   - Click on "Streamlit" in the left navigation
   - Click "Create Streamlit App"
   - Fill in the details:
     - **App Name**: `LLM_OBSERVE`
     - **Warehouse**: Select your compute warehouse
     - **App Location**: `LLM_OBSERVE.MONITORING`

3. **Upload Main Application File**
   - Copy the entire contents of `streamlit_app.py`
   - Paste into the Streamlit editor
   - Save the file

### 2.2 Upload Supporting Files

Since Streamlit in Snowflake requires all files to be in the same editor or uploaded to stages, you have two options:

#### Option A: Inline Code (Recommended for this app)
The current application structure has been designed to minimize dependencies. The main `streamlit_app.py` file contains most functionality, but you'll need to create the supporting modules.

1. **Create metrics/sis_evaluator.py content inline**
2. **Create database/sis_storage.py content inline**

#### Option B: Use Snowflake Stages
```sql
-- Upload supporting Python files
PUT file://metrics/sis_evaluator.py @LLM_OBSERVE_APP_STAGE/metrics/;
PUT file://metrics/__init__.py @LLM_OBSERVE_APP_STAGE/metrics/;
PUT file://database/sis_storage.py @LLM_OBSERVE_APP_STAGE/database/;
PUT file://database/__init__.py @LLM_OBSERVE_APP_STAGE/database/;

-- Verify uploads
LIST @LLM_OBSERVE_APP_STAGE;
```

## Step 3: Create Streamlit Application

### 3.1 Using Snowflake Web Interface

```sql
-- Create the Streamlit application
CREATE STREAMLIT LLM_OBSERVE
ROOT_LOCATION = '@LLM_OBSERVE.MONITORING.LLM_OBSERVE_APP_STAGE'
MAIN_FILE = 'streamlit_app.py'
QUERY_WAREHOUSE = '<your_warehouse_name>'
COMMENT = 'LLM Observe - Foundation Models Monitoring Application';
```

### 3.2 Alternative: Direct SQL Creation

```sql
-- Create Streamlit app with embedded code
CREATE OR REPLACE STREAMLIT LLM_OBSERVE (
    -- Paste your streamlit_app.py content here
    -- Note: This approach requires the entire application to be in one file
)
ROOT_LOCATION = '@LLM_OBSERVE.MONITORING.LLM_OBSERVE_APP_STAGE'
MAIN_FILE = 'streamlit_app.py'
QUERY_WAREHOUSE = '<your_warehouse_name>';
```

## Step 4: Initialize Database Schema

### 4.1 Manual Table Creation (Optional)

If you prefer to create tables manually before first run:

```sql
-- Create runs table
CREATE TABLE IF NOT EXISTS LLM_OBSERVE_RUNS (
    RUN_ID STRING PRIMARY KEY,
    TIMESTAMP TIMESTAMP_NTZ,
    MODEL STRING NOT NULL,
    PROMPT STRING NOT NULL,
    RESPONSE STRING NOT NULL,
    GROUND_TRUTH STRING,
    SOURCE_DOCS STRING,
    TASK_TYPE STRING,
    DOMAIN STRING,
    MAX_TOKENS INTEGER,
    ADDITIONAL_CONTEXT STRING,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Create metrics table
CREATE TABLE IF NOT EXISTS LLM_OBSERVE_METRICS (
    METRIC_ID STRING PRIMARY KEY,
    RUN_ID STRING,
    METRIC_NAME STRING NOT NULL,
    METRIC_VALUE FLOAT,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    FOREIGN KEY (RUN_ID) REFERENCES LLM_OBSERVE_RUNS (RUN_ID)
);
```

### 4.2 Create Views

```sql
-- Combined runs and metrics view
CREATE OR REPLACE VIEW LLM_OBSERVE_RUNS_WITH_METRICS AS
SELECT 
    r.*,
    m.METRIC_NAME,
    m.METRIC_VALUE
FROM LLM_OBSERVE_RUNS r
LEFT JOIN LLM_OBSERVE_METRICS m ON r.RUN_ID = m.RUN_ID;

-- Pivoted metrics view for dashboard
CREATE OR REPLACE VIEW LLM_OBSERVE_METRICS_PIVOT AS
SELECT 
    r.RUN_ID,
    r.TIMESTAMP,
    r.MODEL,
    r.TASK_TYPE,
    r.DOMAIN,
    MAX(CASE WHEN m.METRIC_NAME = 'bertscore_f1' THEN m.METRIC_VALUE END) AS BERTSCORE_F1,
    MAX(CASE WHEN m.METRIC_NAME = 'semantic_similarity' THEN m.METRIC_VALUE END) AS SEMANTIC_SIMILARITY,
    MAX(CASE WHEN m.METRIC_NAME = 'rouge_l' THEN m.METRIC_VALUE END) AS ROUGE_L,
    MAX(CASE WHEN m.METRIC_NAME = 'faithfulness' THEN m.METRIC_VALUE END) AS FAITHFULNESS,
    MAX(CASE WHEN m.METRIC_NAME = 'hallucination_rate' THEN m.METRIC_VALUE END) AS HALLUCINATION_RATE,
    MAX(CASE WHEN m.METRIC_NAME = 'citation_accuracy' THEN m.METRIC_VALUE END) AS CITATION_ACCURACY,
    MAX(CASE WHEN m.METRIC_NAME = 'source_attribution' THEN m.METRIC_VALUE END) AS SOURCE_ATTRIBUTION,
    MAX(CASE WHEN m.METRIC_NAME = 'retrieval_precision' THEN m.METRIC_VALUE END) AS RETRIEVAL_PRECISION,
    MAX(CASE WHEN m.METRIC_NAME = 'faithfulness_delta' THEN m.METRIC_VALUE END) AS FAITHFULNESS_DELTA,
    MAX(CASE WHEN m.METRIC_NAME = 'harm_detection_score' THEN m.METRIC_VALUE END) AS HARM_DETECTION_SCORE,
    MAX(CASE WHEN m.METRIC_NAME = 'sensitive_blocking' THEN m.METRIC_VALUE END) AS SENSITIVE_BLOCKING,
    MAX(CASE WHEN m.METRIC_NAME = 'topic_filtering' THEN m.METRIC_VALUE END) AS TOPIC_FILTERING,
    MAX(CASE WHEN m.METRIC_NAME = 'bias_score' THEN m.METRIC_VALUE END) AS BIAS_SCORE,
    MAX(CASE WHEN m.METRIC_NAME = 'adversarial_resistance' THEN m.METRIC_VALUE END) AS ADVERSARIAL_RESISTANCE,
    MAX(CASE WHEN m.METRIC_NAME = 'audit_completeness' THEN m.METRIC_VALUE END) AS AUDIT_COMPLETENESS
FROM LLM_OBSERVE_RUNS r
LEFT JOIN LLM_OBSERVE_METRICS m ON r.RUN_ID = m.RUN_ID
GROUP BY r.RUN_ID, r.TIMESTAMP, r.MODEL, r.TASK_TYPE, r.DOMAIN;
```

## Step 5: Test Deployment

### 5.1 Access Your Application

1. **Get Application URL**
   ```sql
   -- Show Streamlit apps
   SHOW STREAMLITS;
   
   -- Get app URL
   SELECT GET_DDL('STREAMLIT', 'LLM_OBSERVE');
   ```

2. **Open Application**
   - Navigate to the provided URL
   - Or access via Snowflake interface: "Apps" > "Streamlit" > "LLM_OBSERVE"

### 5.2 Initial Setup

1. **Verify Environment Info**
   - Check the expandable "Snowflake Environment Info" section
   - Confirm correct role, warehouse, database, and schema

2. **Initialize Database**
   - Click "Initialize Database" in the sidebar
   - Verify successful table creation

3. **Test Basic Functionality**
   - Select a model (e.g., "llama2-7b-chat")
   - Enter a simple prompt
   - Click "Generate & Evaluate"
   - Verify response generation and metrics calculation

## Step 6: Configuration and Optimization

### 6.1 Warehouse Configuration

```sql
-- Create dedicated warehouse for LLM workloads (recommended)
CREATE WAREHOUSE IF NOT EXISTS LLM_OBSERVE_WH
WITH WAREHOUSE_SIZE = 'MEDIUM'
AUTO_SUSPEND = 300
AUTO_RESUME = TRUE
COMMENT = 'Warehouse for LLM Observe application';

-- Grant usage to your role
GRANT USAGE ON WAREHOUSE LLM_OBSERVE_WH TO ROLE <your_role>;
```

### 6.2 Performance Optimization

```sql
-- Create indexes for better query performance
-- Note: Snowflake automatically optimizes queries, but you can create clustering keys

-- Cluster runs table by timestamp for time-based queries
ALTER TABLE LLM_OBSERVE_RUNS CLUSTER BY (TIMESTAMP);

-- Cluster metrics table by run_id for join performance
ALTER TABLE LLM_OBSERVE_METRICS CLUSTER BY (RUN_ID);
```

### 6.3 Security Setup

```sql
-- Create role-based access
CREATE ROLE IF NOT EXISTS LLM_OBSERVE_USER;
CREATE ROLE IF NOT EXISTS LLM_OBSERVE_ADMIN;

-- Grant permissions to user role
GRANT USAGE ON DATABASE LLM_OBSERVE TO ROLE LLM_OBSERVE_USER;
GRANT USAGE ON SCHEMA LLM_OBSERVE.MONITORING TO ROLE LLM_OBSERVE_USER;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA LLM_OBSERVE.MONITORING TO ROLE LLM_OBSERVE_USER;
GRANT USAGE ON ALL FUNCTIONS IN SCHEMA LLM_OBSERVE.MONITORING TO ROLE LLM_OBSERVE_USER;

-- Grant admin permissions
GRANT ALL ON DATABASE LLM_OBSERVE TO ROLE LLM_OBSERVE_ADMIN;
GRANT ALL ON SCHEMA LLM_OBSERVE.MONITORING TO ROLE LLM_OBSERVE_ADMIN;
```

## Step 7: Monitoring and Maintenance

### 7.1 Application Monitoring

```sql
-- Monitor Streamlit app usage
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.STREAMLIT_EVENTS 
WHERE STREAMLIT_NAME = 'LLM_OBSERVE'
ORDER BY START_TIME DESC;

-- Monitor warehouse usage
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY 
WHERE WAREHOUSE_NAME = '<your_warehouse>'
ORDER BY START_TIME DESC;
```

### 7.2 Data Maintenance

```sql
-- Create procedure for data cleanup
CREATE OR REPLACE PROCEDURE CLEANUP_OLD_LLMOBSERVE_DATA(DAYS_TO_KEEP NUMBER)
RETURNS STRING
LANGUAGE SQL
AS
$$
DECLARE
    cutoff_date TIMESTAMP_NTZ;
    runs_deleted NUMBER;
BEGIN
    cutoff_date := DATEADD(DAY, -DAYS_TO_KEEP, CURRENT_TIMESTAMP());
    
    -- Delete old metrics first
    DELETE FROM LLM_OBSERVE_METRICS 
    WHERE RUN_ID IN (
        SELECT RUN_ID FROM LLM_OBSERVE_RUNS 
        WHERE TIMESTAMP < :cutoff_date
    );
    
    -- Delete old runs
    DELETE FROM LLM_OBSERVE_RUNS 
    WHERE TIMESTAMP < :cutoff_date;
    
    GET DIAGNOSTICS runs_deleted = ROW_COUNT;
    
    RETURN 'Cleaned up ' || runs_deleted || ' runs older than ' || DAYS_TO_KEEP || ' days';
END;
$$;

-- Schedule cleanup task (optional)
CREATE OR REPLACE TASK CLEANUP_OLD_DATA
WAREHOUSE = '<your_warehouse>'
SCHEDULE = 'USING CRON 0 2 * * 0'  -- Weekly on Sunday at 2 AM
AS
CALL CLEANUP_OLD_LLMOBSERVE_DATA(90);
```

## Troubleshooting

### Common Issues

1. **Permission Denied Errors**
   ```sql
   -- Check current role and grants
   SELECT CURRENT_ROLE();
   SHOW GRANTS TO ROLE <your_role>;
   ```

2. **UDF Creation Failures**
   ```sql
   -- Check if Python UDFs are allowed
   SHOW PARAMETERS LIKE 'PYTHON_CONNECTOR_PYFORMAT_ENABLED';
   ```

3. **Cortex Access Issues**
   ```sql
   -- Verify Cortex function access
   SELECT SNOWFLAKE.CORTEX.COMPLETE('llama2-7b-chat', 'Hello');
   ```

4. **Performance Issues**
   - Increase warehouse size
   - Check for data skew
   - Optimize UDF logic

### Getting Help

- **Snowflake Documentation**: [Streamlit in Snowflake](https://docs.snowflake.com/en/developer-guide/streamlit/about-streamlit)
- **Cortex Documentation**: [Snowflake Cortex](https://docs.snowflake.com/en/user-guide/snowflake-cortex)
- **Support**: Create a support ticket through Snowflake Support Portal

## Next Steps

1. **Customize Metrics**: Add domain-specific evaluation metrics
2. **Scale Deployment**: Configure auto-scaling warehouses
3. **Integrate Data**: Connect with existing data pipelines
4. **Monitor Usage**: Set up alerting and monitoring dashboards
5. **Share Results**: Use Snowflake's data sharing capabilities

---

Your LLM Observe application is now deployed and ready to monitor Foundation Model performance in Snowflake! 🎉 