#!/usr/bin/env python3
"""
Setup BigQuery Permissions
Helps configure the service account with proper BigQuery permissions.
"""

import json
from pathlib import Path

def check_service_account_permissions(credentials_path: str):
    """Check and display service account information for permission setup."""
    print("🔐 Service Account Permission Setup Guide")
    print("=" * 50)
    
    try:
        # Read credentials
        with open(credentials_path, 'r') as f:
            creds = json.load(f)
        
        project_id = creds.get('project_id')
        client_email = creds.get('client_email')
        
        print(f"📊 Project ID: {project_id}")
        print(f"📧 Service Account Email: {client_email}")
        print()
        
        print("🔧 Required BigQuery Permissions:")
        print("=" * 40)
        print("The service account needs the following IAM roles:")
        print()
        print("1. BigQuery Admin (recommended)")
        print("   - bigquery.datasets.create")
        print("   - bigquery.datasets.get")
        print("   - bigquery.tables.create")
        print("   - bigquery.tables.get")
        print("   - bigquery.tables.update")
        print("   - bigquery.jobs.create")
        print("   - bigquery.jobs.get")
        print()
        print("2. Or BigQuery Data Editor + BigQuery Job User")
        print("   - BigQuery Data Editor role")
        print("   - BigQuery Job User role")
        print()
        
        print("🚀 Setup Instructions:")
        print("=" * 30)
        print("1. Go to Google Cloud Console:")
        print(f"   https://console.cloud.google.com/iam-admin/iam?project={project_id}")
        print()
        print("2. Find the service account:")
        print(f"   {client_email}")
        print()
        print("3. Click 'Edit' (pencil icon)")
        print()
        print("4. Add the 'BigQuery Admin' role:")
        print("   - Click 'ADD ANOTHER ROLE'")
        print("   - Select 'BigQuery Admin'")
        print("   - Click 'SAVE'")
        print()
        print("5. Wait 2-3 minutes for permissions to propagate")
        print()
        print("6. Re-run the data upload script:")
        print("   python scripts/send_data_to_gcp_bigquery.py")
        print()
        
        print("🔍 Alternative: Using gcloud CLI")
        print("=" * 35)
        print("If you have gcloud CLI installed, you can run:")
        print()
        print(f"gcloud projects add-iam-policy-binding {project_id} \\")
        print(f"    --member='serviceAccount:{client_email}' \\")
        print("    --role='roles/bigquery.admin'")
        print()
        
        print("📋 Verification Commands:")
        print("=" * 25)
        print("After setting up permissions, verify with:")
        print()
        print(f"gcloud projects get-iam-policy {project_id} \\")
        print(f"    --flatten='bindings[].members' \\")
        print(f"    --filter='bindings.members:{client_email}'")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading credentials: {e}")
        return False

def main():
    credentials_path = ".secrets/cement-ops-key.json"
    
    if not Path(credentials_path).exists():
        print(f"❌ Credentials file not found: {credentials_path}")
        return 1
    
    success = check_service_account_permissions(credentials_path)
    
    if success:
        print("✅ Permission setup guide completed!")
        print("📝 Follow the instructions above to grant BigQuery permissions")
        print("🔄 Then re-run: python scripts/send_data_to_gcp_bigquery.py")
    else:
        print("❌ Failed to generate permission setup guide")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
