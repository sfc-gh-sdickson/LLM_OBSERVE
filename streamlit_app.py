# LLM Observe - Streamlit in Snowflake Application
# A comprehensive LLM monitoring and evaluation app for Snowflake Foundation Models

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional
import uuid

# Snowflake imports for Streamlit in Snowflake
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col, lit, when, regexp_extract, length, split
from snowflake.snowpark.types import StringType, FloatType, TimestampType, VariantType
from snowflake.cortex import Complete, Sentiment, Translate, Summarize
import snowflake.snowpark as snowpark

# Import our custom modules
from metrics.sis_evaluator import SiSMetricsEvaluator
from database.sis_storage import SiSDatabase

# Page configuration
st.set_page_config(
    page_title="🔍 LLM Observe - Snowflake Foundation Models Monitor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Snowflake session
@st.cache_resource
def init_snowflake_session():
    """Initialize Snowflake session for Streamlit in Snowflake"""
    try:
        session = get_active_session()
        return session
    except Exception as e:
        st.error(f"Failed to get Snowflake session: {e}")
        return None

# Initialize application components
@st.cache_resource
def init_components():
    """Initialize application components for Streamlit in Snowflake"""
    session = init_snowflake_session()
    if session is None:
        return None, None
    
    try:
        # Initialize database and metrics components
        db = SiSDatabase(session)
        metrics_evaluator = SiSMetricsEvaluator(session)
        
        return db, metrics_evaluator
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return None, None

def get_available_models(session):
    """Get available Snowflake Foundation Models"""
    # These are the currently available Snowflake Cortex LLM functions
    return [
        "snowflake-arctic",
        "llama2-7b-chat", 
        "llama2-13b-chat",
        "llama2-70b-chat",
        "llama3-8b",
        "llama3-70b",
        "mistral-7b",
        "mistral-8x7b",
        "mixtral-8x7b",
        "reka-core",
        "reka-flash"
    ]

def main():
    """Main application entry point"""
    st.title("🔍 LLM Observe - Snowflake Foundation Models Monitor")
    st.markdown("**Streamlit in Snowflake Application** - Monitor and evaluate LLM performance with comprehensive metrics tracking")
    
    # Initialize components
    session = init_snowflake_session()
    if session is None:
        st.error("❌ Unable to connect to Snowflake session. Please ensure this app is running in Snowflake.")
        st.info("💡 This application is designed to run in **Streamlit in Snowflake** environment.")
        return
    
    db, metrics_evaluator = init_components()
    if db is None or metrics_evaluator is None:
        st.error("❌ Failed to initialize application components.")
        st.info("🔧 Try refreshing the page or check the deployment guide for troubleshooting steps.")
        return
    
    # Display current Snowflake context
    with st.expander("ℹ️ Snowflake Environment Info"):
        try:
            current_role = session.sql("SELECT CURRENT_ROLE()").collect()[0][0]
            current_warehouse = session.sql("SELECT CURRENT_WAREHOUSE()").collect()[0][0]
            current_database = session.sql("SELECT CURRENT_DATABASE()").collect()[0][0]
            current_schema = session.sql("SELECT CURRENT_SCHEMA()").collect()[0][0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Role", current_role)
            with col2:
                st.metric("Warehouse", current_warehouse)
            with col3:
                st.metric("Database", current_database)
            with col4:
                st.metric("Schema", current_schema)
                
            # Check Python runtime info
            st.markdown("**Python Runtime Information:**")
            python_info = session.sql("SELECT SYSTEM$GET_PYTHON_RUNTIME_INFO()").collect()[0][0]
            st.code(python_info, language="json")
            
        except Exception as e:
            st.warning(f"Could not retrieve Snowflake context: {e}")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("🤖 Model Selection")
        available_models = get_available_models(session)
        selected_model = st.selectbox(
            "Choose Snowflake Foundation Model:",
            available_models,
            help="Select from available Snowflake Cortex models"
        )
        
        # Model parameters for Snowflake Cortex
        st.subheader("🎛️ Model Parameters")
        # Note: Snowflake Cortex has limited parameter control
        max_tokens = st.slider("Max Tokens", 50, 4000, 1000, 50)
        
        # Evaluation settings
        st.subheader("📊 Evaluation Settings")
        enable_ground_truth = st.checkbox("Enable Ground Truth Comparison", True)
        enable_source_docs = st.checkbox("Enable Source Document Analysis", True)
        enable_safety_checks = st.checkbox("Enable Safety & Bias Checks", True)
        
        # Data management
        st.subheader("🗄️ Data Management")
        if st.button("🔄 Initialize Database"):
            with st.spinner("Initializing database tables..."):
                try:
                    db.initialize_tables()
                    st.success("✅ Database tables initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Error initializing database: {e}")
                    st.info("💡 This might be due to insufficient privileges. Check the deployment guide.")
        
        # UDF status
        st.subheader("🔧 System Status")
        if hasattr(metrics_evaluator, 'udfs_available'):
            if metrics_evaluator.udfs_available:
                st.success("✅ UDFs Available")
            else:
                st.warning("⚠️ UDFs Not Available (using fallback methods)")
                st.info("💡 This is normal if Python runtime 3.11 is being used. The app will use SQL-based fallback methods.")
        
        # Runtime information
        try:
            runtime_info = session.sql("SELECT SYSTEM$GET_PYTHON_RUNTIME_INFO()").collect()[0][0]
            runtime_data = json.loads(runtime_info)
            python_version = runtime_data.get('python_version', 'Unknown')
            st.info(f"🐍 Python Runtime: {python_version}")
        except:
            st.info("🐍 Python Runtime: Unknown")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Query Interface", 
        "📊 Live Metrics", 
        "📈 Dashboard", 
        "🗄️ History"
    ])
    
    with tab1:
        query_interface(session, db, metrics_evaluator, selected_model, 
                       max_tokens, enable_ground_truth, enable_source_docs, enable_safety_checks)
    
    with tab2:
        live_metrics_view(session, db)
    
    with tab3:
        dashboard_view(session, db)
    
    with tab4:
        history_view(session, db)

def query_interface(session, db, metrics_evaluator, selected_model: str, max_tokens: int,
                   enable_ground_truth: bool, enable_source_docs: bool, enable_safety_checks: bool):
    """Query interface for LLM interaction and evaluation"""
    
    st.header("💬 Query Interface")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Input")
        prompt = st.text_area(
            "Enter your prompt:",
            height=150,
            placeholder="Ask a question or provide a prompt for the LLM...",
            key="main_prompt"
        )
        
        # Optional inputs for evaluation
        ground_truth = None
        source_docs = None
        
        if enable_ground_truth:
            ground_truth = st.text_area(
                "Ground Truth (optional):",
                height=100,
                placeholder="Expected or reference answer for comparison...",
                key="ground_truth_input"
            )
            
        if enable_source_docs:
            source_docs = st.text_area(
                "Source Documents (optional):",
                height=100,
                placeholder="Source documents or context for faithfulness evaluation...",
                key="source_docs_input"
            )
    
    with col2:
        st.subheader("🎯 Evaluation Context")
        
        # Task type selection
        task_type = st.selectbox(
            "Task Type:",
            ["extractive", "abstractive", "open-domain", "creative"],
            help="Select task type for appropriate metric thresholds",
            key="task_type_select"
        )
        
        # Domain selection
        domain = st.selectbox(
            "Domain:",
            ["enterprise", "creative", "critical", "general"],
            help="Select domain for appropriate safety thresholds",
            key="domain_select"
        )
        
        # Additional context
        additional_context = st.text_input(
            "Additional Context:",
            placeholder="Any additional context for evaluation...",
            key="additional_context_input"
        )
    
    # Submit button
    if st.button("🚀 Generate & Evaluate", type="primary", use_container_width=True):
        if not prompt.strip():
            st.error("❌ Please enter a prompt!")
            return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Generate response using Snowflake Cortex
            status_text.text("🔄 Generating response with Snowflake Cortex...")
            progress_bar.progress(20)
            
            # Use Snowflake Cortex Complete function
            response = generate_cortex_response(session, selected_model, prompt, max_tokens)
            
            progress_bar.progress(40)
            
            # Step 2: Evaluate metrics
            status_text.text("📊 Evaluating metrics...")
            
            evaluation_context = {
                'prompt': prompt,
                'response': response,
                'ground_truth': ground_truth,
                'source_docs': source_docs,
                'task_type': task_type,
                'domain': domain,
                'additional_context': additional_context,
                'model': selected_model,
                'max_tokens': max_tokens
            }
            
            metrics_results = metrics_evaluator.evaluate_all_metrics(evaluation_context)
            progress_bar.progress(70)
            
            # Step 3: Store results in Snowflake
            status_text.text("💾 Storing results in Snowflake...")
            
            run_id = str(uuid.uuid4())
            run_data = {
                'run_id': run_id,
                'timestamp': datetime.now(),
                'model': selected_model,
                'prompt': prompt,
                'response': response,
                'ground_truth': ground_truth,
                'source_docs': source_docs,
                'task_type': task_type,
                'domain': domain,
                'max_tokens': max_tokens,
                'additional_context': additional_context,
                'metrics': metrics_results
            }
            
            db.store_run(run_data)
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            # Display results
            display_results(response, metrics_results, run_id)
            
        except Exception as e:
            st.error(f"❌ Error during generation/evaluation: {str(e)}")
            st.exception(e)
            
            # Show helpful troubleshooting info
            with st.expander("🔧 Troubleshooting Information"):
                st.markdown("""
                **Common Issues:**
                1. **UDF Creation Errors**: If using Python 3.11, UDFs may not be available. The app will use fallback methods.
                2. **Permission Errors**: Ensure your role has access to Snowflake Cortex functions.
                3. **Model Access**: Verify that the selected model is available in your Snowflake account.
                
                **Solutions:**
                - Try a different model
                - Contact your Snowflake administrator for Cortex access
                - Check the deployment guide for required privileges
                """)
                
        finally:
            progress_bar.empty()
            status_text.empty()

def generate_cortex_response(session, model: str, prompt: str, max_tokens: int) -> str:
    """Generate response using Snowflake Cortex"""
    try:
        # Clean the prompt for SQL safety
        clean_prompt = prompt.replace("'", "''").replace("\\", "\\\\")
        
        # Use Snowflake Cortex Complete function
        cortex_query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            '{clean_prompt}',
            {{
                'max_tokens': {max_tokens}
            }}
        ) as response
        """
        
        result = session.sql(cortex_query).collect()
        
        if result and len(result) > 0:
            return result[0]['RESPONSE']
        else:
            return "Error: No response generated"
            
    except Exception as e:
        st.error(f"Error generating response with Cortex: {e}")
        
        # Check for common issues
        if "permission" in str(e).lower():
            st.warning("⚠️ Permission issue detected. Please ensure your role has access to Snowflake Cortex functions.")
            st.code("GRANT USAGE ON FUNCTION SNOWFLAKE.CORTEX.COMPLETE TO ROLE <your_role>;")
        
        # Fallback response for demonstration
        return f"""This is a simulated response from {model} for your prompt. 

In a properly configured Snowflake environment with Cortex access, this would be the actual response from the foundation model.

Error details: {str(e)}

To resolve this issue:
1. Ensure you have access to Snowflake Cortex
2. Verify the model name is correct
3. Check your role permissions
"""

def display_results(response: str, metrics_results: Dict, run_id: str):
    """Display the generated response and evaluation metrics"""
    
    st.success("✅ Generation and evaluation complete!")
    
    # Response section
    st.subheader("🤖 Generated Response")
    st.markdown(f"**Run ID:** `{run_id}`")
    st.markdown("---")
    st.write(response)
    
    # Metrics section
    st.subheader("📊 Evaluation Metrics")
    
    # Create metrics display in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📝 Content Quality**")
        display_metric_card("BERTScore F1", metrics_results.get('bertscore_f1', 0), 0.85, 0.95)
        display_metric_card("Semantic Similarity", metrics_results.get('semantic_similarity', 0), 0.80, 0.90)
        display_metric_card("ROUGE-L Overlap", metrics_results.get('rouge_l', 0), 0.70, 0.85)
        display_metric_card("Faithfulness", metrics_results.get('faithfulness', 0), 0.90, 0.95)
    
    with col2:
        st.markdown("**🔍 Retrieval & Attribution**")
        display_metric_card("Source Attribution", metrics_results.get('source_attribution', 0), 0.90, 0.95)
        display_metric_card("Citation Accuracy", metrics_results.get('citation_accuracy', 0), 0.85, 0.95)
        display_metric_card("Hallucination Rate", metrics_results.get('hallucination_rate', 0), 0.05, 0.10, inverse=True)
        display_metric_card("Faithfulness Delta", metrics_results.get('faithfulness_delta', 0), 0.05, 0.10, inverse=True)
    
    with col3:
        st.markdown("**🛡️ Safety & Security**")
        display_metric_card("Harm Detection", metrics_results.get('harm_detection_score', 1), 0.95, 0.99)
        display_metric_card("Bias Score", metrics_results.get('bias_score', 1), 0.90, 0.95)
        display_metric_card("Adversarial Resistance", metrics_results.get('adversarial_resistance', 1), 0.95, 0.99)
        display_metric_card("Audit Completeness", metrics_results.get('audit_completeness', 1), 0.95, 1.0)
    
    # Detailed metrics in expandable section
    with st.expander("📋 Detailed Metrics"):
        metrics_df = pd.DataFrame([
            {"Metric": k, "Value": v, "Type": "Score" if isinstance(v, (int, float)) else "Text"}
            for k, v in metrics_results.items()
        ])
        st.dataframe(metrics_df, use_container_width=True)

def display_metric_card(name: str, value: float, warning_threshold: float, 
                       good_threshold: float, inverse: bool = False):
    """Display a metric card with color coding"""
    
    if inverse:
        # For metrics where lower is better (like error rates)
        if value <= warning_threshold:
            color = "🟢"
        elif value <= good_threshold:
            color = "🟡"
        else:
            color = "🔴"
    else:
        # For metrics where higher is better
        if value >= good_threshold:
            color = "🟢"
        elif value >= warning_threshold:
            color = "🟡"
        else:
            color = "🔴"
    
    if isinstance(value, float):
        display_value = f"{value:.3f}"
    else:
        display_value = str(value)
    
    st.metric(
        label=f"{color} {name}",
        value=display_value
    )

def live_metrics_view(session, db):
    """Display live metrics and real-time monitoring"""
    
    st.header("📊 Live Metrics")
    
    try:
        # Get recent runs from Snowflake
        recent_runs_df = db.get_recent_runs(limit=50)
        
        if recent_runs_df.is_empty():
            st.info("📭 No recent runs found. Start by making some queries!")
            return
        
        # Convert to pandas for easier manipulation
        recent_runs_pd = recent_runs_df.to_pandas()
        
        # Real-time metrics summary
        st.subheader("🔴 Real-time Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = recent_runs_pd['BERTSCORE_F1'].mean() if 'BERTSCORE_F1' in recent_runs_pd.columns else 0
            st.metric("Avg BERTScore", f"{avg_score:.3f}")
        
        with col2:
            avg_faithfulness = recent_runs_pd['FAITHFULNESS'].mean() if 'FAITHFULNESS' in recent_runs_pd.columns else 0
            st.metric("Avg Faithfulness", f"{avg_faithfulness:.3f}")
        
        with col3:
            avg_safety = recent_runs_pd['HARM_DETECTION_SCORE'].mean() if 'HARM_DETECTION_SCORE' in recent_runs_pd.columns else 1
            st.metric("Avg Safety Score", f"{avg_safety:.3f}")
        
        with col4:
            total_runs = len(recent_runs_pd)
            st.metric("Total Runs", total_runs)
        
        # Recent runs table
        st.subheader("📋 Recent Runs")
        
        # Display recent runs with key columns
        display_columns = ['RUN_ID', 'MODEL', 'TIMESTAMP', 'TASK_TYPE', 'DOMAIN']
        available_columns = [col for col in display_columns if col in recent_runs_pd.columns]
        
        if available_columns:
            st.dataframe(
                recent_runs_pd[available_columns].head(10),
                use_container_width=True
            )
        else:
            st.warning("⚠️ No data columns available for display")
            
    except Exception as e:
        st.error(f"❌ Error loading live metrics: {e}")

def dashboard_view(session, db):
    """Display comprehensive dashboard with historical metrics"""
    
    st.header("📈 Metrics Dashboard")
    
    # Time range selector
    col1, col2 = st.columns([1, 3])
    with col1:
        time_range = st.selectbox(
            "📅 Time Range:",
            ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"]
        )
    
    try:
        # Get historical data from Snowflake
        historical_df = db.get_historical_metrics(time_range)
        
        if historical_df.is_empty():
            st.info("📭 No historical data available for the selected time range.")
            return
        
        # Convert to pandas for plotting
        historical_pd = historical_df.to_pandas()
        
        # Metrics over time charts
        st.subheader("📈 Metrics Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # BERTScore trend
            if 'BERTSCORE_F1' in historical_pd.columns and 'TIMESTAMP' in historical_pd.columns:
                fig_bert = px.line(
                    historical_pd, 
                    x='TIMESTAMP', 
                    y='BERTSCORE_F1',
                    title='BERTScore F1 Over Time',
                    labels={'BERTSCORE_F1': 'BERTScore F1', 'TIMESTAMP': 'Time'}
                )
                fig_bert.add_hline(y=0.85, line_dash="dash", line_color="orange", annotation_text="Warning")
                fig_bert.add_hline(y=0.95, line_dash="dash", line_color="green", annotation_text="Good")
                st.plotly_chart(fig_bert, use_container_width=True)
        
        with col2:
            # Faithfulness trend
            if 'FAITHFULNESS' in historical_pd.columns and 'TIMESTAMP' in historical_pd.columns:
                fig_faith = px.line(
                    historical_pd,
                    x='TIMESTAMP',
                    y='FAITHFULNESS',
                    title='Faithfulness Over Time',
                    labels={'FAITHFULNESS': 'Faithfulness Score', 'TIMESTAMP': 'Time'}
                )
                fig_faith.add_hline(y=0.90, line_dash="dash", line_color="orange", annotation_text="Warning")
                fig_faith.add_hline(y=0.95, line_dash="dash", line_color="green", annotation_text="Good")
                st.plotly_chart(fig_faith, use_container_width=True)
        
        # Model comparison
        st.subheader("🤖 Model Performance Comparison")
        
        if 'MODEL' in historical_pd.columns:
            model_stats = historical_pd.groupby('MODEL').agg({
                col: 'mean' for col in historical_pd.columns 
                if col not in ['RUN_ID', 'MODEL', 'TIMESTAMP', 'PROMPT', 'RESPONSE']
                and historical_pd[col].dtype in ['float64', 'int64']
            }).round(3)
            
            if not model_stats.empty:
                st.dataframe(model_stats, use_container_width=True)
            else:
                st.info("📊 No numeric metrics available for model comparison")
        
    except Exception as e:
        st.error(f"❌ Error loading dashboard data: {e}")

def history_view(session, db):
    """Display detailed history and search functionality"""
    
    st.header("🗄️ Run History")
    
    # Search and filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_query = st.text_input("🔍 Search prompts/responses:", key="history_search")
    
    with col2:
        model_filter = st.selectbox(
            "Filter by model:", 
            ["All"] + get_available_models(session),
            key="history_model_filter"
        )
    
    with col3:
        date_filter = st.date_input("📅 Filter by date:", key="history_date_filter")
    
    try:
        # Get filtered history from Snowflake
        history_df = db.search_history(
            search_query=search_query if search_query else None,
            model_filter=model_filter if model_filter != "All" else None,
            date_filter=date_filter
        )
        
        if history_df.is_empty():
            st.info("📭 No runs found matching the criteria.")
            return
        
        # Convert to pandas for display
        history_pd = history_df.to_pandas()
        
        # Display results with expandable details
        for idx, run in history_pd.iterrows():
            run_id = run.get('RUN_ID', 'Unknown')
            model = run.get('MODEL', 'Unknown')
            timestamp = run.get('TIMESTAMP', 'Unknown')
            
            with st.expander(f"🔍 Run {str(run_id)[:8]} - {model} - {timestamp}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📝 Prompt:**")
                    prompt = run.get('PROMPT', 'N/A')
                    st.text(prompt[:200] + "..." if len(str(prompt)) > 200 else prompt)
                    
                    st.markdown("**🤖 Response:**")
                    response = run.get('RESPONSE', 'N/A')
                    st.text(response[:200] + "..." if len(str(response)) > 200 else response)
                
                with col2:
                    st.markdown("**📊 Key Metrics:**")
                    metrics_data = {}
                    
                    # Extract metric columns
                    for col_name in run.index:
                        if col_name not in ['RUN_ID', 'TIMESTAMP', 'MODEL', 'PROMPT', 'RESPONSE', 'GROUND_TRUTH', 'SOURCE_DOCS']:
                            if pd.notna(run[col_name]):
                                metrics_data[col_name] = run[col_name]
                    
                    if metrics_data:
                        st.json(metrics_data)
                    else:
                        st.info("No metrics data available")
                        
    except Exception as e:
        st.error(f"❌ Error loading history: {e}")

if __name__ == "__main__":
    main() 