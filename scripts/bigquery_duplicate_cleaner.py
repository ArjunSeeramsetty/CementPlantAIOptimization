"""Check BigQuery for duplicates and clean them up."""

import os
import sys
import logging
from typing import List, Dict
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BigQueryDuplicateCleaner:
    """
    Check BigQuery for duplicate tables and clean them up.
    """
    
    def __init__(self):
        """Initialize the BigQuery cleaner."""
        self.project_id = "cement-ai-opt-38517"
        self.dataset_id = "cement_analytics"
        self.duplicates_found = []
        self.tables_deleted = []
        
    def check_bigquery_tables(self) -> Dict:
        """Check all tables in the BigQuery dataset."""
        logger.info("=== Checking BigQuery Tables ===")
        
        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
            
            # Load credentials
            credentials_path = '.secrets/cement-ops-key.json'
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            client = bigquery.Client(credentials=credentials, project=self.project_id)
            
            # Get dataset reference
            dataset_ref = client.dataset(self.dataset_id)
            
            # List all tables
            tables = list(client.list_tables(dataset_ref))
            
            logger.info(f"Found {len(tables)} tables in dataset {self.dataset_id}:")
            
            table_info = {}
            table_names = []
            
            for table in tables:
                table_ref = client.get_table(table.reference)
                table_name = table.table_id
                table_names.append(table_name)
                
                table_info[table_name] = {
                    'num_rows': table_ref.num_rows,
                    'created': table_ref.created,
                    'modified': table_ref.modified,
                    'size_bytes': table_ref.num_bytes,
                    'schema_fields': len(table_ref.schema)
                }
                
                logger.info(f"  - {table_name}: {table_ref.num_rows} rows, {len(table_ref.schema)} fields")
            
            # Check for duplicates
            duplicates = self._find_duplicates(table_names)
            
            if duplicates:
                logger.warning(f"⚠️ Found {len(duplicates)} duplicate table groups:")
                for group in duplicates:
                    logger.warning(f"  Duplicate group: {group}")
                self.duplicates_found = duplicates
            else:
                logger.info("✅ No duplicate tables found")
            
            return {
                'total_tables': len(tables),
                'table_info': table_info,
                'duplicates': duplicates,
                'table_names': table_names
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to check BigQuery tables: {e}")
            return {}
    
    def _find_duplicates(self, table_names: List[str]) -> List[List[str]]:
        """Find duplicate table names."""
        # Check for exact duplicates
        name_counts = {}
        for name in table_names:
            name_counts[name] = name_counts.get(name, 0) + 1
        
        duplicates = []
        for name, count in name_counts.items():
            if count > 1:
                duplicates.append([name] * count)
        
        # Check for similar names that might be duplicates
        similar_duplicates = self._find_similar_names(table_names)
        duplicates.extend(similar_duplicates)
        
        return duplicates
    
    def _find_similar_names(self, table_names: List[str]) -> List[List[str]]:
        """Find tables with similar names that might be duplicates."""
        similar_groups = []
        processed = set()
        
        for i, name1 in enumerate(table_names):
            if name1 in processed:
                continue
                
            similar_group = [name1]
            
            for j, name2 in enumerate(table_names[i+1:], i+1):
                if name2 in processed:
                    continue
                    
                # Check for similar patterns
                if self._are_similar_names(name1, name2):
                    similar_group.append(name2)
                    processed.add(name2)
            
            if len(similar_group) > 1:
                similar_groups.append(similar_group)
                processed.update(similar_group)
        
        return similar_groups
    
    def _are_similar_names(self, name1: str, name2: str) -> bool:
        """Check if two table names are similar enough to be considered duplicates."""
        # Remove common suffixes/prefixes
        base1 = name1.replace('_v1', '').replace('_v2', '').replace('_backup', '').replace('_old', '')
        base2 = name2.replace('_v1', '').replace('_v2', '').replace('_backup', '').replace('_old', '')
        
        # Check if base names are the same
        if base1 == base2:
            return True
        
        # Check for timestamp suffixes
        import re
        pattern = r'_\d{8}_\d{6}$'  # _YYYYMMDD_HHMMSS
        clean1 = re.sub(pattern, '', name1)
        clean2 = re.sub(pattern, '', name2)
        
        if clean1 == clean2:
            return True
        
        return False
    
    def delete_duplicate_tables(self, duplicates: List[List[str]]) -> bool:
        """Delete duplicate tables, keeping the most recent one."""
        logger.info("=== Deleting Duplicate Tables ===")
        
        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
            
            # Load credentials
            credentials_path = '.secrets/cement-ops-key.json'
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            client = bigquery.Client(credentials=credentials, project=self.project_id)
            
            deleted_count = 0
            
            for duplicate_group in duplicates:
                if len(duplicate_group) <= 1:
                    continue
                
                logger.info(f"Processing duplicate group: {duplicate_group}")
                
                # Get table metadata to determine which to keep
                table_metadata = {}
                for table_name in duplicate_group:
                    table_ref = client.get_table(f"{self.project_id}.{self.dataset_id}.{table_name}")
                    table_metadata[table_name] = {
                        'created': table_ref.created,
                        'modified': table_ref.modified,
                        'num_rows': table_ref.num_rows
                    }
                
                # Keep the table with the most recent modification time
                keep_table = max(table_metadata.keys(), 
                               key=lambda x: table_metadata[x]['modified'])
                
                logger.info(f"Keeping table: {keep_table} (modified: {table_metadata[keep_table]['modified']})")
                
                # Delete the others
                for table_name in duplicate_group:
                    if table_name == keep_table:
                        continue
                    
                    table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
                    
                    try:
                        client.delete_table(table_id)
                        logger.info(f"✅ Deleted duplicate table: {table_name}")
                        self.tables_deleted.append({
                            'table_name': table_name,
                            'reason': 'duplicate',
                            'kept_table': keep_table,
                            'deleted_at': datetime.now()
                        })
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"❌ Failed to delete table {table_name}: {e}")
            
            logger.info(f"✅ Deleted {deleted_count} duplicate tables")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete duplicate tables: {e}")
            return False
    
    def clean_empty_tables(self) -> bool:
        """Delete tables with 0 rows."""
        logger.info("=== Cleaning Empty Tables ===")
        
        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
            
            # Load credentials
            credentials_path = '.secrets/cement-ops-key.json'
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            client = bigquery.Client(credentials=credentials, project=self.project_id)
            
            # Get dataset reference
            dataset_ref = client.dataset(self.dataset_id)
            tables = list(client.list_tables(dataset_ref))
            
            deleted_count = 0
            
            for table in tables:
                table_ref = client.get_table(table.reference)
                
                if table_ref.num_rows == 0:
                    table_id = f"{self.project_id}.{self.dataset_id}.{table.table_id}"
                    
                    try:
                        client.delete_table(table_id)
                        logger.info(f"✅ Deleted empty table: {table.table_id}")
                        self.tables_deleted.append({
                            'table_name': table.table_id,
                            'reason': 'empty',
                            'deleted_at': datetime.now()
                        })
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"❌ Failed to delete empty table {table.table_id}: {e}")
            
            if deleted_count == 0:
                logger.info("✅ No empty tables found")
            else:
                logger.info(f"✅ Deleted {deleted_count} empty tables")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clean empty tables: {e}")
            return False
    
    def verify_cleanup(self) -> Dict:
        """Verify the cleanup was successful."""
        logger.info("=== Verifying Cleanup ===")
        
        # Re-check tables after cleanup
        final_state = self.check_bigquery_tables()
        
        logger.info("Final BigQuery state:")
        logger.info(f"  Total tables: {final_state.get('total_tables', 0)}")
        logger.info(f"  Duplicates found: {len(final_state.get('duplicates', []))}")
        logger.info(f"  Tables deleted: {len(self.tables_deleted)}")
        
        return final_state
    
    def save_cleanup_report(self):
        """Save cleanup report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'artifacts/bigquery_cleanup_report_{timestamp}.json'
        
        report = {
            'cleanup_timestamp': datetime.now().isoformat(),
            'project_id': self.project_id,
            'dataset_id': self.dataset_id,
            'duplicates_found': self.duplicates_found,
            'tables_deleted': self.tables_deleted,
            'final_state': self.verify_cleanup()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Cleanup report saved to: {report_file}")
    
    def run_complete_cleanup(self) -> bool:
        """Run the complete cleanup process."""
        logger.info("🧹 Starting BigQuery Duplicate Cleanup")
        logger.info("=" * 50)
        
        # Step 1: Check current state
        initial_state = self.check_bigquery_tables()
        
        if not initial_state:
            logger.error("❌ Failed to check BigQuery state")
            return False
        
        # Step 2: Delete duplicates
        duplicates = initial_state.get('duplicates', [])
        if duplicates:
            logger.info(f"Found {len(duplicates)} duplicate groups, cleaning up...")
            if not self.delete_duplicate_tables(duplicates):
                logger.error("❌ Failed to delete duplicates")
                return False
        else:
            logger.info("✅ No duplicates found")
        
        # Step 3: Clean empty tables
        if not self.clean_empty_tables():
            logger.error("❌ Failed to clean empty tables")
            return False
        
        # Step 4: Verify cleanup
        final_state = self.verify_cleanup()
        
        # Step 5: Save report
        self.save_cleanup_report()
        
        logger.info("\n" + "=" * 50)
        logger.info("🎉 BigQuery Cleanup Complete!")
        logger.info(f"✅ Tables deleted: {len(self.tables_deleted)}")
        logger.info(f"✅ Final tables: {final_state.get('total_tables', 0)}")
        logger.info(f"✅ Duplicates remaining: {len(final_state.get('duplicates', []))}")
        
        return True


if __name__ == "__main__":
    # Run the complete cleanup
    cleaner = BigQueryDuplicateCleaner()
    success = cleaner.run_complete_cleanup()
    
    if success:
        logger.info("🎉 BigQuery cleanup completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ BigQuery cleanup failed")
        sys.exit(1)
