#!/bin/bash
# start_airflow.sh - Automates Airflow 3.x configuration for 8082 port and Jupyter compatibility.

# 1. Set Airflow Home to Project Directory
export AIRFLOW_HOME="/workspace/pro-cv-finalproject-cv-01/training"

# 2. Configure Ports (Avoid 8080 collision with Jupyter)
export AIRFLOW_WEBSERVER_PORT=8082
export AIRFLOW__WEBSERVER__WEB_SERVER_PORT=8082

# 3. Fix Airflow 3.x Execution API for custom port
# In Airflow 3.x, workers connect to {base_url}/execution/ to report state.
export AIRFLOW__API__BASE_URL="http://localhost:8082"
export AIRFLOW__CORE__EXECUTION_API_SERVER_URL="http://localhost:8082/execution/"

# 4. Use LocalExecutor (standard for Airflow 3.x multiprocess)
export AIRFLOW__CORE__EXECUTOR="LocalExecutor"

echo "🚀 Starting Airflow 3.x on port 8082..."
echo "🔗 Base URL: $AIRFLOW__API__BASE_URL"
echo "📡 Execution API: $AIRFLOW__CORE__EXECUTION_API_SERVER_URL"

# Enable the project's venv
source /workspace/pro-cv-finalproject-cv-01/training/.venv/bin/activate

# Launch
airflow standalone
