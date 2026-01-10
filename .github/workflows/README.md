# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automated ML pipeline execution.

## Available Workflows

### `default.yml` - Weekly ML Retraining Pipeline

Runs data quality gates on feedback data from the database.

**Triggers:**
- Scheduled: Every Monday at 2 AM UTC
- Manual: `workflow_dispatch` (can be triggered manually)
- Push: When changes are made to `trainer/**`, `backend/**`, or `retrainer/**` directories

## Required GitHub Secrets

Before running the workflow, you must configure the following secrets in your GitHub repository:

**Go to:** `Settings` → `Secrets and variables` → `Actions` → `New repository secret`

### Required Secrets:

1. **`DB_PASSWORD`** (Required)
   - Your PostgreSQL database password
   - Example: `your_secure_password_here`

### Optional Secrets (with defaults):

2. **`DB_NAME`** (Optional, default: `ai_text_feedback`)
   - PostgreSQL database name

3. **`DB_USER`** (Optional, default: `postgres`)
   - PostgreSQL username

4. **`DB_HOST`** (Optional, default: `localhost`)
   - Database host address
   - For cloud databases (AWS RDS, Azure, etc.), use the full hostname
   - Example: `your-db.xxxxx.us-east-1.rds.amazonaws.com`

5. **`DB_PORT`** (Optional, default: `5432`)
   - PostgreSQL port number

6. **`MLFLOW_TRACKING_URI`** (Optional, default: `sqlite:///./mlflow.db`)
   - MLflow tracking URI
   - For remote MLflow: `http://your-mlflow-server:5000`
   - For cloud: Check your MLflow service documentation

## Network Configuration

### Connecting to External Databases

If your database is hosted externally (AWS RDS, Azure Database, etc.):

1. **Ensure your database allows connections from GitHub Actions IPs:**
   - GitHub Actions uses dynamic IP addresses
   - Consider using a VPN or allowing all IPs temporarily (less secure)
   - For AWS RDS: Modify security group to allow GitHub Actions IP ranges

2. **Use SSL connections** (recommended for production):
   - Add SSL parameters to connection string if needed

### Testing Locally

To test the workflow locally without GitHub Actions:

```bash
# Set environment variables
export DB_NAME=ai_text_feedback
export DB_USER=postgres
export DB_PASSWORD=your_password
export DB_HOST=localhost
export DB_PORT=5432

# Run the script
python retrainer/db_fetch_qualgates.py
```

## Workflow Outputs

The workflow generates:
- **Artifact**: `clean-feedback-dataset` (CSV file with cleaned feedback data)
- **Summary**: Dataset statistics and quality metrics

## Troubleshooting

### Database Connection Failed

**Error:** `Database connection test failed`

**Solutions:**
1. Verify all secrets are set correctly in GitHub Settings
2. Check that your database allows connections from GitHub Actions IPs
3. Verify database credentials are correct
4. Ensure database is running and accessible

### No Output File Generated

**Error:** `No output file generated`

**Possible causes:**
1. No data in the database yet
2. All data was filtered out by quality gates
3. Database query failed

**Solution:** Check database logs and ensure feedback data exists in `feedback_events` table

### Insufficient Data Warning

**Warning:** `Dataset has fewer than 10 rows`

**Meaning:** The cleaned dataset is too small for effective model training.

**Solution:** Collect more feedback data before running training

## Security Best Practices

1. **Never commit secrets** to the repository
2. **Use GitHub Secrets** for all sensitive data
3. **Rotate passwords** regularly
4. **Use least-privilege database users** for GitHub Actions
5. **Enable database SSL** for production connections
6. **Monitor workflow logs** for any exposed credentials

## Next Steps

After the data quality gates complete successfully:
1. Review the cleaned dataset artifact
2. Use the dataset for model retraining (manual or automated)
3. Update model version in production

