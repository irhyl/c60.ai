# Deployment Guide

## Local Deployment
1. Create and activate virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python -m interface.cli`

## Cloud Deployment
1. Set up cloud credentials
2. Configure `config/cloud_config.json`
3. Run deployment script: `python -m deploy.c60toolkit.deploy_cloud`

## Environment Variables
- `C60_ENV`: Set to 'development' or 'production'
- `C60_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `C60_CACHE_DIR`: Cache directory path
