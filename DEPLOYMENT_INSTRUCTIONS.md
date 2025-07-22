# 🚀 LLM Observe - Deployment Instructions

## Overview
This guide walks you through deploying the LLM Observe application in **Streamlit in Snowflake** with full persistent data storage using Snowflake tables.

## 📋 Prerequisites

### Required Access
- Snowflake account with **Streamlit in Snowflake** enabled
- Database and schema creation privileges
- **Snowflake Cortex** access for Foundation Models
- Role with necessary permissions for:
  - Creating tables, views, and functions
  - Reading from `INFORMATION_SCHEMA`
  - Executing UDFs and stored procedures

### Recommended Setup
- **Compute Warehouse**: Medium or larger for consistent performance
- **Python Version**: 3.10 or 3.12 (recommended), 3.11 supported with fallbacks
- **Snowflake Role**: `ACCOUNTADMIN` or custom role with required privileges

## 🗄️ Step 1: Database Setup

> **Note**: This schema has been optimized for Snowflake's architecture. Unlike traditional databases, Snowflake uses clustering keys instead of indexes and views with CTEs for complex analytics rather than functions.

### 1.1 Create Database and Schema (Optional)
```sql
-- Create dedicated database for LLM monitoring
CREATE DATABASE IF NOT EXISTS LLM_OBSERVE;
USE DATABASE LLM_OBSERVE;

-- Create schema for evaluation data
CREATE SCHEMA IF NOT EXISTS EVALUATION;
USE SCHEMA EVALUATION;
```

### 1.2 Run the Schema Script
Execute the complete `snowflake_schema.sql` script in your Snowflake worksheet:

```sql
-- Copy and paste the entire contents of snowflake_schema.sql
-- This creates:
-- - LLM_RUNS table
-- - LLM_METRICS table  
-- - LLM_RUN_CONTEXT table
-- - Clustering keys for performance optimization
-- - Convenience views for analytics
-- - Stored procedures for data management
```

### 1.3 Verify Schema Creation
```sql
-- Check that all objects were created
SHOW TABLES;
SHOW VIEWS;
SHOW PROCEDURES;

-- Validate table structure
SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE 
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
  AND TABLE_NAME IN ('LLM_RUNS', 'LLM_METRICS', 'LLM_RUN_CONTEXT')
ORDER BY TABLE_NAME, ORDINAL_POSITION;
```

## 📱 Step 2: Streamlit App Deployment

### 2.1 Access Streamlit in Snowflake
1. Navigate to **Data > Streamlit** in your Snowflake interface
2. Click **+ Streamlit App**
3. Choose your database and schema (`LLM_OBSERVE.EVALUATION`)
4. Name your app: `LLM_Observe`

### 2.2 Upload Application Code
1. **Main File**: Copy the entire contents of `streamlit_app.py`
2. **Packages**: The app will automatically handle dependencies
3. **Environment**: The app auto-detects Snowflake environment

### 2.3 Configure Permissions
Ensure your Streamlit app has access to:
```sql
-- Grant necessary permissions to the app's role
GRANT USAGE ON DATABASE LLM_OBSERVE TO ROLE <STREAMLIT_ROLE>;
GRANT USAGE ON SCHEMA LLM_OBSERVE.EVALUATION TO ROLE <STREAMLIT_ROLE>;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA LLM_OBSERVE.EVALUATION TO ROLE <STREAMLIT_ROLE>;
GRANT SELECT ON ALL VIEWS IN SCHEMA LLM_OBSERVE.EVALUATION TO ROLE <STREAMLIT_ROLE>;
GRANT USAGE ON ALL PROCEDURES IN SCHEMA LLM_OBSERVE.EVALUATION TO ROLE <STREAMLIT_ROLE>;

-- Grant Cortex access for LLM calls
GRANT USAGE ON INTEGRATION SNOWFLAKE.CORTEX TO ROLE <STREAMLIT_ROLE>;
```

## ✅ Step 3: Verification and Testing

### 3.1 Launch the Application
1. Click **Run** in the Streamlit interface
2. Check the **System Status** section in the sidebar:
   - ✅ Snowflake Connected
   - ✅ Advanced Modules Loaded  
   - ✅ Persistent Snowflake Storage
   - ✅ Interactive Charts

### 3.2 Test Core Functionality
1. **Query Interface**: Submit a test prompt
2. **Verify Storage**: Check that data appears in tables:
   ```sql
   SELECT COUNT(*) FROM LLM_RUNS;
   SELECT COUNT(*) FROM LLM_METRICS;
   ```
3. **Live Metrics**: Confirm runs appear in the dashboard
4. **Model Comparisons**: Test with multiple models to see comparison charts
5. **Historical Search**: Test search and filtering

### 3.3 Validate Data Persistence
```sql
-- View recent runs
SELECT * FROM VW_RECENT_LLM_RUNS LIMIT 5;

-- Check metrics by category
SELECT METRIC_CATEGORY, COUNT(*) as METRIC_COUNT
FROM LLM_METRICS 
GROUP BY METRIC_CATEGORY;

-- Verify pass/fail rates
SELECT 
    METRIC_NAME,
    AVG(METRIC_VALUE) as AVG_VALUE,
    SUM(CASE WHEN PASSED_THRESHOLD THEN 1 ELSE 0 END) as PASSED,
    COUNT(*) as TOTAL,
    ROUND(100.0 * SUM(CASE WHEN PASSED_THRESHOLD THEN 1 ELSE 0 END) / COUNT(*), 2) as PASS_RATE_PCT
FROM LLM_METRICS 
GROUP BY METRIC_NAME
ORDER BY METRIC_NAME;

-- View model performance trends (using the new analytics view)
SELECT * FROM VW_MODEL_PERFORMANCE 
WHERE MODEL = 'llama3-8b' 
ORDER BY METRIC_NAME;
```

## 🛠️ Step 4: Configuration and Customization

### 4.1 Model Configuration
The app supports these Snowflake Cortex foundation models for evaluation:

**Core Models:**
- `snowflake-arctic`
- `llama2-7b-chat`, `llama2-13b-chat`, `llama2-70b-chat`
- `llama3-8b`, `llama3-70b`
- `llama4-maverick`, `llama4-scout`
- `mistral-7b`, `mistral-8x7b`, `mixtral-8x7b`
- `reka-core`, `reka-flash`

**Advanced Models:**
- `openai-gpt-4.1`
- `claude-4-sonnet`, `claude-3-7-sonnet`, `claude-4-opus`
- `deepseek-r1`
- `pixtral-large`

> **All models are available natively through Snowflake Cortex** - use any model directly with `SNOWFLAKE.CORTEX.COMPLETE()`.

### 4.2 Metric Thresholds
Customize metric thresholds in the `_get_metric_category()` method:
```python
thresholds = {
    'bertscore_f1': 0.85,        # Adjust as needed
    'faithfulness': 0.95,
    'hallucination_rate': 0.05,
    # ... other thresholds
}
```

### 4.3 Data Retention
Use the cleanup procedure for data management:
```sql
-- Clean up data older than 90 days
CALL SP_CLEANUP_OLD_RUNS(90);

-- Schedule regular cleanup (optional)
CREATE TASK CLEANUP_TASK
    WAREHOUSE = 'YOUR_WAREHOUSE'
    SCHEDULE = 'USING CRON 0 2 * * 0'  -- Weekly at 2 AM Sunday
AS
    CALL SP_CLEANUP_OLD_RUNS(90);

-- Start the task
ALTER TASK CLEANUP_TASK RESUME;
```

## 🔧 Troubleshooting

### Common Issues

**1. "SQL compilation error: Cannot create a Python function"**
- **Cause**: Python version compatibility
- **Solution**: The app includes fallbacks for all Python versions
- **Action**: Check System Status for current mode

**2. "ModuleNotFoundError: No module named 'metrics'"**
- **Cause**: Module import in non-Snowflake environment
- **Solution**: Use single-file `streamlit_app.py` deployment
- **Action**: Ensure all code is in the main file

**3. "Tables not found" errors**
- **Cause**: Schema not created or wrong context
- **Solution**: Run `snowflake_schema.sql` first
- **Action**: Verify with `SHOW TABLES;` and `SHOW VIEWS;`

**5. "Cannot create index" or similar syntax errors**
- **Cause**: Using non-Snowflake SQL syntax
- **Solution**: Use the provided Snowflake-optimized schema
- **Action**: Snowflake uses clustering keys, not traditional indexes

**6. Model not found errors**
- **Cause**: Model name not available in your Snowflake account
- **Solution**: Check available models in your Snowflake Cortex setup
- **Action**: Contact your Snowflake admin if specific models are not accessible

**4. Empty dashboard charts**
- **Cause**: No data or filtering issues
- **Solution**: Generate test data first
- **Action**: Run evaluations in Query Interface tab

### Performance Optimization

**For Large Datasets (>10K runs):**
```sql
-- Optimize clustering keys for your query patterns
-- Default clustering is already set on main tables
-- Add additional clustering if needed for specific queries
ALTER TABLE LLM_METRICS CLUSTER BY (METRIC_CATEGORY, CREATED_AT);

-- Enable automatic clustering for very large datasets
ALTER TABLE LLM_RUNS SET ENABLE_SCHEMA_EVOLUTION = TRUE;
ALTER TABLE LLM_METRICS SET ENABLE_SCHEMA_EVOLUTION = TRUE;
```

**For High-Frequency Usage:**
- Use larger compute warehouses
- Consider auto-suspend/resume settings
- Monitor credit usage

## 🔐 Security Considerations

### Data Privacy
- All evaluation data is stored in your Snowflake account
- No data leaves your environment
- Standard Snowflake encryption applies

### Access Control
```sql
-- Create read-only role for viewers
CREATE ROLE LLM_OBSERVE_READER;
GRANT USAGE ON DATABASE LLM_OBSERVE TO ROLE LLM_OBSERVE_READER;
GRANT USAGE ON SCHEMA LLM_OBSERVE.EVALUATION TO ROLE LLM_OBSERVE_READER;
GRANT SELECT ON ALL TABLES IN SCHEMA LLM_OBSERVE.EVALUATION TO ROLE LLM_OBSERVE_READER;
GRANT SELECT ON ALL VIEWS IN SCHEMA LLM_OBSERVE.EVALUATION TO ROLE LLM_OBSERVE_READER;
```

### Audit Logging
The app automatically logs:
- All user interactions (100% completeness)
- Model inputs and outputs
- Evaluation metrics and thresholds
- Timestamps and user attribution

## 🎯 Next Steps

1. **Customize Metrics**: Adapt evaluation criteria to your use case
2. **Set up Monitoring**: Use Snowflake's monitoring tools
3. **Create Alerts**: Set up notifications for metric thresholds
4. **Scale as Needed**: Adjust compute resources based on usage

## 📞 Support

For issues specific to:
- **Snowflake Setup**: Consult Snowflake documentation
- **Streamlit in Snowflake**: Check Streamlit docs
- **Application Logic**: Review the code comments and error messages

---

## ✅ Success Checklist

- [ ] Database and schema created
- [ ] All tables, views, and procedures deployed
- [ ] Streamlit app running without errors
- [ ] Test evaluation completed successfully
- [ ] Data visible in Snowflake tables
- [ ] Dashboard charts displaying data
- [ ] History search working
- [ ] System status shows all green

**🎉 Your LLM Observe application is now fully deployed with persistent Snowflake storage!** 