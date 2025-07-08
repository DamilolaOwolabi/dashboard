# GCP Deployment Guide: Wind Energy & Battery Management System

This guide walks you through deploying your codebase to Google Cloud Platform (GCP) using BigQuery for data storage/processing and Looker Studio for visualization.

---

## 1. Prepare Your Codebase

- **Remove** any local virtual environments (`venv/`) and the Streamlit dashboard (`dashboard/`) from your deployment folder.
- **Keep** only the necessary folders/files: `src/`, `data/`, `models/`, `outputs/`, `requirements.txt`.
- **Create a deployment folder** (e.g., `gcp_deploy/`) and copy the above into it.

```bash
mkdir gcp_deploy
cp -r src data models outputs requirements.txt gcp_deploy/
```

---

## 2. Upload Data to Google Cloud Storage (GCS) and BigQuery

1. **Create a GCS bucket** (if needed):
   - Go to [Cloud Storage](https://console.cloud.google.com/storage/browser) → Create bucket.
2. **Upload your CSVs** (e.g., `ercot_combined_data.csv`) to the bucket.
3. **In BigQuery Console:**
   - Create a dataset (e.g., `ercot_data`).
   - Create tables by importing your CSVs from GCS or your local machine.
   - Use the same table names as referenced in your code (e.g., `ercot_combined_data`).

---

## 3. Set Up Your GCP Environment

- **Open Cloud Shell** or SSH into a Compute Engine VM.
- **Upload your `gcp_deploy/` folder** to your Cloud Shell or VM:
  - Use the Cloud Shell upload button, or
  - Use `gcloud compute scp` or `scp` if using a VM.

---

## 4. Install Dependencies

```bash
cd ~/gcp_deploy
pip install --user -r requirements.txt
pip install --user google-cloud-bigquery db-dtypes pandas-gbq
```

---

## 5. Authenticate with Google Cloud

```bash
gcloud auth application-default login
```
- This will open a browser for you to log in and set up credentials.

---

## 6. Update Your Code for BigQuery

- In `gcp_deploy/src/main.py`, set your project, dataset, and table names:

```python
BQ_PROJECT = 'your_project_id'      # e.g., 'ds-7346'
BQ_DATASET = 'your_dataset_name'    # e.g., 'ercot_data'
BQ_TABLE = 'your_table_name'        # e.g., 'ercot_combined_data'
```
- Make sure your code uses the correct table names as seen in the BigQuery Console.

---

## 7. Run Your Script

From your `gcp_deploy` directory:

```bash
python -m src.main
```

- This will read from BigQuery, process your data, and write results (e.g., `evaluation_metrics`) back to BigQuery.

---

## 8. Check Your Results in BigQuery

- Go to the [BigQuery Console](https://console.cloud.google.com/bigquery).
- Navigate to your project and dataset.
- Click on the output table (e.g., `evaluation_metrics`) and click **Preview** to see your results.

---

## 9. Visualize in Looker Studio

1. Go to [Looker Studio](https://lookerstudio.google.com/).
2. Create a new report.
3. Add BigQuery as a data source:
   - Click **Add Data** → **BigQuery**
   - Select your project, dataset, and table(s)
4. Build your dashboard using Looker Studio's drag-and-drop interface.

---

## 10. Troubleshooting Tips

- **ModuleNotFoundError**: Install missing packages with `pip install --user package_name`.
- **BigQuery Not Found Errors**: Double-check your project, dataset, and table names in the BigQuery Console.
- **Authentication Issues**: Run `gcloud auth application-default login` again.
- **Permissions**: Make sure your GCP user has BigQuery Data Editor and Viewer roles.

---

## 11. Example: Minimal BigQuery Read/Write in Python

```python
from google.cloud import bigquery
client = bigquery.Client(project='ds-7346')
df = client.query('SELECT * FROM `ds-7346.ercot_data.ercot_combined_data`').to_dataframe()

# ... process df ...

from google.cloud import bigquery
job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
client.load_table_from_dataframe(df, 'ds-7346.ercot_data.evaluation_metrics', job_config=job_config).result()
```

---

## 12. Additional Resources
- [Google Cloud BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [Looker Studio Documentation](https://support.google.com/looker-studio/)
- [Google Cloud Python Client](https://cloud.google.com/python/docs/reference/bigquery/latest)

---

**If you have any issues, check the error messages and refer to the troubleshooting section above.** 