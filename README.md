# LLM Observe - Streamlit in Snowflake Foundation Models Monitor

A comprehensive **Streamlit in Snowflake** application for monitoring and evaluating LLM performance with Snowflake Foundation Models. This native Snowflake app provides real-time evaluation of 15+ critical metrics including BERTScore, faithfulness, hallucination detection, safety measures, and bias testing - all running directly within your Snowflake environment.

## 🚀 Features

### Core Functionality
- **Native Snowflake Integration**: Runs directly in Snowflake using Streamlit in Snowflake
- **Snowflake Cortex Models**: Direct integration with Snowflake's Foundation Models (Arctic, LLaMA, Mistral, etc.)
- **Real-time Evaluation**: Comprehensive metrics calculation using Snowflake compute
- **Interactive Dashboard**: Live monitoring and historical analysis within Snowflake
- **Advanced Metrics**: 15+ evaluation metrics leveraging Snowflake UDFs and SQL

### Supported Snowflake Cortex Models
- **Snowflake Arctic**: Enterprise-optimized foundation model
- **LLaMA 2 (7B, 13B, 70B)**: Meta's language models
- **LLaMA 3 (8B, 70B)**: Latest Meta models
- **Mistral (7B)**: Efficient foundation model
- **Mixtral (8x7B)**: Mixture of experts model
- **Reka (Core, Flash)**: Multimodal models

### Evaluation Metrics

#### Content Quality (Using Snowflake UDFs)
- **BERTScore F1**: Token overlap with semantic weighting
- **Semantic Similarity**: Jaccard similarity and custom functions
- **ROUGE-L**: Longest common subsequence overlap
- **Faithfulness**: Source document alignment

#### Retrieval & Attribution (Snowflake SQL)
- **Citation Accuracy**: Pattern matching and source verification
- **Source Attribution**: Content grounding analysis
- **Retrieval Precision@K**: Relevance scoring
- **Answer Faithfulness Delta**: Cross-context consistency

#### Safety & Security (Snowflake Cortex)
- **Harm Detection**: Using Snowflake Cortex Sentiment
- **Sensitive Blocking**: SQL pattern matching
- **Topic Filtering**: Content classification
- **Bias Testing**: Statistical bias indicators
- **Adversarial Resistance**: Prompt injection detection

#### Operational
- **Hallucination Rate**: Source verification scoring
- **Audit Logging**: Complete Snowflake-native traceability

## 📋 Requirements

- Snowflake Account with Streamlit in Snowflake enabled
- Access to Snowflake Cortex (Foundation Models)
- Appropriate Snowflake roles and permissions
- Warehouse with sufficient compute resources

## 🛠️ Installation & Deployment

### 1. Prepare Your Snowflake Environment

#### Required Privileges
```sql
-- Grant necessary privileges for Streamlit in Snowflake
GRANT USAGE ON WAREHOUSE <your_warehouse> TO ROLE <your_role>;
GRANT CREATE STREAMLIT ON SCHEMA <your_schema> TO ROLE <your_role>;
GRANT CREATE STAGE ON SCHEMA <your_schema> TO ROLE <your_role>;
GRANT CREATE TABLE ON SCHEMA <your_schema> TO ROLE <your_role>;
GRANT CREATE VIEW ON SCHEMA <your_schema> TO ROLE <your_role>;
GRANT CREATE FUNCTION ON SCHEMA <your_schema> TO ROLE <your_role>;

-- Grant Cortex privileges for Foundation Models
GRANT USAGE ON FUNCTION SNOWFLAKE.CORTEX.COMPLETE TO ROLE <your_role>;
GRANT USAGE ON FUNCTION SNOWFLAKE.CORTEX.SENTIMENT TO ROLE <your_role>;
```

#### Create Database and Schema
```sql
CREATE DATABASE IF NOT EXISTS LLM_OBSERVE;
CREATE SCHEMA IF NOT EXISTS LLM_OBSERVE.MONITORING;
USE SCHEMA LLM_OBSERVE.MONITORING;
```

### 2. Deploy the Streamlit App

#### Option A: Using Snowflake Web Interface
1. Log into Snowflake
2. Navigate to "Streamlit" in the left sidebar
3. Click "Create Streamlit App"
4. Name your app (e.g., "LLM_OBSERVE")
5. Select your warehouse, database, and schema
6. Copy the contents of `streamlit_app.py` into the editor
7. Click "Deploy"

#### Option B: Using SnowSQL
```sql
-- Create the Streamlit app
CREATE STREAMLIT LLM_OBSERVE
ROOT_LOCATION = '@LLM_OBSERVE.MONITORING.app_stage'
MAIN_FILE = 'streamlit_app.py'
QUERY_WAREHOUSE = '<your_warehouse>';

-- Upload app files to stage
PUT file://streamlit_app.py @LLM_OBSERVE.MONITORING.app_stage;
PUT file://metrics/sis_evaluator.py @LLM_OBSERVE.MONITORING.app_stage/metrics/;
PUT file://database/sis_storage.py @LLM_OBSERVE.MONITORING.app_stage/database/;
PUT file://metrics/__init__.py @LLM_OBSERVE.MONITORING.app_stage/metrics/;
PUT file://database/__init__.py @LLM_OBSERVE.MONITORING.app_stage/database/;
```

### 3. Initialize the Database
1. Open your deployed Streamlit app
2. Click "Initialize Database" in the sidebar
3. Verify tables are created successfully

## ⚙️ Configuration

### Snowflake Context
The app automatically uses your current Snowflake session context:
- **Role**: Your current role
- **Warehouse**: Current warehouse (configurable)
- **Database**: Current database
- **Schema**: Current schema

### Model Parameters
- **Max Tokens**: Control response length
- **Model Selection**: Choose from available Cortex models
- **Evaluation Settings**: Toggle different metric categories

## 🚀 Usage

### 1. Access Your App
Navigate to your Streamlit app URL (provided when deployed) or find it in the Snowflake interface under "Streamlit"

### 2. Initialize Database
First time setup:
- Click "Initialize Database" in the sidebar
- Verify successful table creation

### 3. Run Evaluations
1. **Select Model**: Choose from available Snowflake Cortex models
2. **Enter Prompt**: Type your question or instruction
3. **Add Context** (optional):
   - Ground truth for comparison
   - Source documents for faithfulness evaluation
4. **Configure Settings**:
   - Task type (extractive, abstractive, open-domain, creative)
   - Domain (enterprise, creative, critical, general)
5. **Generate & Evaluate**: Click the button to run

### 4. View Results
- **Generated Response**: From Snowflake Cortex
- **Comprehensive Metrics**: Color-coded performance indicators
- **Detailed Analysis**: Expandable metrics breakdown

## 📊 Dashboard Features

### Live Metrics Tab
- Real-time performance overview
- Recent runs summary
- Key metric averages

### Dashboard Tab
- Historical trend analysis
- Model performance comparison
- Time-range filtering
- Safety metrics correlation

### History Tab
- Search and filter past runs
- Detailed run examination
- Export capabilities to Snowflake stages

## 🔧 Snowflake-Specific Features

### Data Storage
- **Native Tables**: All data stored in Snowflake tables
- **Snowpark DataFrames**: Efficient data manipulation
- **Time Travel**: Leverage Snowflake's data recovery features

### Compute Optimization
- **UDF Execution**: Metrics calculated using Snowflake compute
- **Vectorized Operations**: Efficient SQL-based calculations
- **Warehouse Scaling**: Automatic compute scaling

### Security & Governance
- **Role-Based Access**: Inherit Snowflake security model
- **Audit Trail**: Complete data lineage in Snowflake
- **Data Sharing**: Share results across Snowflake accounts

## 📈 Metrics Implementation

### SQL-Based Metrics
```sql
-- Example: Token overlap calculation
SELECT TOKEN_OVERLAP_F1(response, ground_truth) as bertscore_f1
FROM llm_runs;

-- Example: Bias detection
SELECT DETECT_BIAS_INDICATORS(response) as bias_score
FROM llm_runs;
```

### Snowflake Cortex Integration
```sql
-- Sentiment analysis for safety
SELECT SNOWFLAKE.CORTEX.SENTIMENT(response) as sentiment_score;

-- Foundation model completion
SELECT SNOWFLAKE.CORTEX.COMPLETE('llama2-7b-chat', prompt) as response;
```

## 🛡️ Security & Compliance

### Data Protection
- **Encryption**: Snowflake's native encryption at rest and in transit
- **Access Control**: Role-based access control (RBAC)
- **Network Security**: Private connectivity options

### Audit & Compliance
- **Query History**: Full audit trail in Snowflake
- **Data Lineage**: Track data flow and transformations
- **Retention Policies**: Configurable data retention

## 🗄️ Database Schema

### Tables Created
```sql
-- Main runs table
LLM_OBSERVE_RUNS (
    RUN_ID STRING PRIMARY KEY,
    TIMESTAMP TIMESTAMP_NTZ,
    MODEL STRING,
    PROMPT STRING,
    RESPONSE STRING,
    -- ... additional columns
);

-- Metrics table
LLM_OBSERVE_METRICS (
    METRIC_ID STRING PRIMARY KEY,
    RUN_ID STRING,
    METRIC_NAME STRING,
    METRIC_VALUE FLOAT,
    -- ... additional columns
);
```

### Views Created
- `LLM_OBSERVE_RUNS_WITH_METRICS`: Joined view of runs and metrics
- `LLM_OBSERVE_METRICS_PIVOT`: Pivoted metrics for dashboard

## 🧪 Development

### Local Development
For development purposes, you can work locally with Snowpark:

```python
from snowflake.snowpark import Session

# Create session for testing
session = Session.builder.configs(connection_parameters).create()
```

### Custom UDFs
Add new metric calculations by creating UDFs:

```sql
CREATE OR REPLACE FUNCTION YOUR_METRIC(text STRING)
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.8'
HANDLER = 'your_function'
AS $$
def your_function(text):
    # Your calculation logic
    return result
$$;
```

## 🤝 Best Practices

### Performance Optimization
- **Warehouse Sizing**: Use appropriate warehouse size for your workload
- **Query Optimization**: Leverage Snowflake's query optimizer
- **Caching**: Utilize result caching for repeated queries

### Cost Management
- **Auto-Suspend**: Configure warehouse auto-suspend
- **Resource Monitoring**: Monitor credit usage
- **Query Optimization**: Write efficient SQL for UDFs

## 🆘 Troubleshooting

### Common Issues

1. **Permission Errors**
   ```sql
   -- Grant missing privileges
   GRANT USAGE ON FUNCTION SNOWFLAKE.CORTEX.COMPLETE TO ROLE <your_role>;
   ```

2. **UDF Creation Failures**
   - Check Python runtime permissions
   - Verify function syntax
   - Ensure proper error handling

3. **Performance Issues**
   - Scale up warehouse if needed
   - Optimize SQL queries
   - Check for data skew

### Getting Help
- Check Snowflake documentation for Streamlit in Snowflake
- Review query history for error details
- Monitor warehouse utilization

## 🔄 Updates & Maintenance

### Updating the App
1. Modify code locally
2. Upload to Snowflake stage
3. Refresh Streamlit app

### Data Maintenance
```sql
-- Clean old data
CALL CLEANUP_OLD_DATA(90); -- Keep 90 days

-- Export data
COPY INTO @my_stage/export.csv FROM llm_observe_runs;
```

## 📞 Support

For issues specific to:
- **Snowflake Features**: Consult Snowflake documentation
- **Cortex Models**: Check Snowflake Cortex documentation
- **Performance**: Review warehouse and query optimization guides

---

**Built specifically for Snowflake's Streamlit in Snowflake environment** 🔷

Experience the power of native Snowflake LLM monitoring with enterprise-grade security, scalability, and governance. 