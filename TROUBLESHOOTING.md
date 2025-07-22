# Troubleshooting Guide - LLM Observe in Snowflake

This guide helps resolve common issues when deploying and running the LLM Observe application in Streamlit in Snowflake.

## 🚨 Common Error: Python Runtime Package Conflicts

### Error Message:
```
Error running Streamlit: [391546] SQL compilation error: Cannot create a Python function with the specified packages. Please check your packages specification and try again. 'One or more package conflicts were detected.'. Hint: These packages are available in other python runtimes:["numpy==2.0.2"->[3.9, 3.10, 3.12], "snowflake-snowpark-python"->[3.8, 3.9, 3.10, 3.12], "pandas==2.2.3"->[3.9, 3.10, 3.12], ...]
```

### Cause:
You're running the application in a Python environment where the required packages aren't available for the specific Python runtime version. Most commonly, this happens when:
- Using Python 3.11 (packages not available)
- UDFs are trying to use an unsupported Python runtime version

### ✅ Solutions:

#### Option 1: Use the Updated Code (Recommended)
The latest version of the application has been updated to:
- Use Python 3.10 runtime for UDFs (supported)
- Gracefully fall back to SQL-based methods if UDFs fail
- Continue working even without UDFs

**Status Check**: Look for the "System Status" section in the sidebar:
- ✅ "UDFs Available" = Full functionality
- ⚠️ "UDFs Not Available" = Using fallback methods (still works!)

#### Option 2: Check Your Python Runtime
```sql
-- Check your current Python runtime version
SELECT SYSTEM$GET_PYTHON_RUNTIME_INFO();
```

#### Option 3: Disable UDF Creation (Workaround)
If you continue having issues, you can modify the `metrics/sis_evaluator.py` file to skip UDF creation entirely:

```python
def _register_udfs(self):
    """Register User Defined Functions for metric calculations"""
    # Skip UDF creation - use fallback methods only
    self.logger.info("Skipping UDF creation - using fallback methods")
    return
```

## 🔐 Permission Errors

### Error: Access Denied to Cortex Functions
```
Error generating response with Cortex: SQL access control error: Insufficient privileges to operate on function 'COMPLETE'
```

### Solutions:
```sql
-- Grant Cortex function access
GRANT USAGE ON FUNCTION SNOWFLAKE.CORTEX.COMPLETE TO ROLE <your_role>;
GRANT USAGE ON FUNCTION SNOWFLAKE.CORTEX.SENTIMENT TO ROLE <your_role>;

-- Grant Streamlit creation privileges
GRANT CREATE STREAMLIT ON SCHEMA <your_schema> TO ROLE <your_role>;
GRANT CREATE TABLE ON SCHEMA <your_schema> TO ROLE <your_role>;
GRANT CREATE VIEW ON SCHEMA <your_schema> TO ROLE <your_role>;
GRANT CREATE FUNCTION ON SCHEMA <your_schema> TO ROLE <your_role>;
```

### Check Your Current Privileges:
```sql
-- Check current role
SELECT CURRENT_ROLE();

-- Check grants for your role
SHOW GRANTS TO ROLE <your_role>;
```

## 🗄️ Database Initialization Errors

### Error: Table Creation Failed
```
Error initializing database: SQL access control error: Insufficient privileges to operate on schema
```

### Solutions:
1. **Check Schema Permissions**:
   ```sql
   GRANT USAGE ON DATABASE <database_name> TO ROLE <your_role>;
   GRANT USAGE ON SCHEMA <schema_name> TO ROLE <your_role>;
   GRANT CREATE TABLE ON SCHEMA <schema_name> TO ROLE <your_role>;
   ```

2. **Use Existing Tables**: If you can't create tables, ask your admin to create them:
   ```sql
   -- Run these as admin
   CREATE TABLE LLM_OBSERVE_RUNS (
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
   
   CREATE TABLE LLM_OBSERVE_METRICS (
       METRIC_ID STRING PRIMARY KEY,
       RUN_ID STRING,
       METRIC_NAME STRING NOT NULL,
       METRIC_VALUE FLOAT,
       CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
   );
   ```

## 🤖 Model Access Issues

### Error: Model Not Found
```
Error generating response with Cortex: Object 'COMPLETE' does not exist or not authorized
```

### Solutions:
1. **Check Available Models**:
   ```sql
   -- Test Cortex access
   SELECT SNOWFLAKE.CORTEX.COMPLETE('llama2-7b-chat', 'Hello world');
   ```

2. **Verify Model Names**: Use exact model names as supported by Snowflake Cortex:
   - `snowflake-arctic`
   - `llama2-7b-chat`
   - `llama2-13b-chat`
   - `llama2-70b-chat`
   - `llama3-8b`
   - `llama3-70b`
   - `mistral-7b`
   - `mixtral-8x7b`

3. **Contact Admin**: Your Snowflake admin may need to enable Cortex for your account.

## 🔧 Application Won't Start

### Error: Failed to Get Snowflake Session
```
Failed to get Snowflake session: 'get_active_session' is not available
```

### Cause:
The application is not running in a Streamlit in Snowflake environment.

### Solutions:
1. **Deploy Properly**: Ensure you're deploying in Snowflake, not running locally
2. **Use Correct Method**: Deploy via Snowflake Web UI or SnowSQL
3. **Check Environment**: This app only works in Streamlit in Snowflake

## 📊 No Data Showing in Dashboard

### Possible Causes:
1. No runs have been executed yet
2. Database tables are empty
3. Permission issues reading tables

### Solutions:
1. **Execute Test Run**: Use the Query Interface to generate some test data
2. **Check Table Contents**:
   ```sql
   SELECT COUNT(*) FROM LLM_OBSERVE_RUNS;
   SELECT COUNT(*) FROM LLM_OBSERVE_METRICS;
   ```
3. **Verify Data**:
   ```sql
   SELECT * FROM LLM_OBSERVE_RUNS LIMIT 5;
   ```

## 🐍 Python Runtime Information

### Check Your Environment:
```sql
-- Get detailed Python runtime info
SELECT SYSTEM$GET_PYTHON_RUNTIME_INFO();
```

### Supported Python Versions for UDFs:
- ✅ Python 3.8
- ✅ Python 3.9  
- ✅ Python 3.10
- ❌ Python 3.11 (limited package support)
- ✅ Python 3.12

## 🔄 Performance Issues

### Slow Query Performance:
1. **Scale Up Warehouse**:
   ```sql
   ALTER WAREHOUSE <warehouse_name> SET WAREHOUSE_SIZE = 'LARGE';
   ```

2. **Check Warehouse Status**:
   ```sql
   SHOW WAREHOUSES LIKE '<warehouse_name>';
   ```

3. **Monitor Query History**:
   ```sql
   SELECT query_text, execution_status, total_elapsed_time
   FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY 
   WHERE query_text ILIKE '%LLM_OBSERVE%'
   ORDER BY start_time DESC
   LIMIT 10;
   ```

## 🆘 Getting Help

### Debug Information to Collect:
1. **Python Runtime Version**: From System Status in sidebar
2. **Current Role and Privileges**: `SELECT CURRENT_ROLE(); SHOW GRANTS TO ROLE <role>;`
3. **Error Messages**: Full error text from Streamlit
4. **Warehouse Info**: `SHOW WAREHOUSES;`
5. **Cortex Access**: Test with `SELECT SNOWFLAKE.CORTEX.COMPLETE('llama2-7b-chat', 'test');`

### Contact Points:
- **Snowflake Support**: For platform-specific issues
- **Admin Team**: For permissions and access issues
- **Documentation**: [Streamlit in Snowflake Docs](https://docs.snowflake.com/en/developer-guide/streamlit/about-streamlit)

## 🔧 Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| Python 3.11 Runtime | Use updated code with Python 3.10 UDFs |
| UDF Creation Fails | App automatically uses fallback methods |
| Permission Denied | Grant Cortex and schema privileges |
| No Data in Dashboard | Run test queries first |
| Model Access Error | Verify Cortex access and model names |
| Database Init Error | Check table creation privileges |

## ✅ Validation Checklist

Before reporting issues, verify:
- [ ] Running in Streamlit in Snowflake (not locally)
- [ ] Have Cortex function access
- [ ] Have schema creation privileges  
- [ ] Using supported model names
- [ ] Check System Status shows UDF availability
- [ ] Test with simple Cortex query first

---

**Remember**: The application is designed to work even when UDFs aren't available - it will automatically use SQL-based fallback methods! 🚀 