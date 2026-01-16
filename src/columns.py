#!/usr/bin/env python3
"""
columns.py - Database Schema Inspector

This script inspects all tables and columns in your database to understand
the complete data structure and help debug data flow issues.
"""

import sys
import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class DatabaseInspector:
    """
    Comprehensive database schema and data inspector
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Try common database paths
            possible_paths = [
                os.path.join(project_root, "crypto_bot.db"),
                os.path.join(project_root, "data", "crypto_history.db"),
                os.path.join(project_root, "data", "crypto_bot.db"),
                "crypto_bot.db",
                "crypto_history.db"
            ]
            
            self.db_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.db_path = path
                    break
            
            if self.db_path is None:
                print("❌ No database file found! Tried:")
                for path in possible_paths:
                    print(f"   - {path}")
                sys.exit(1)
        else:
            self.db_path = db_path
            
        self.schema_info = {}
        self.data_samples = {}
        
    def inspect_all(self):
        """Run complete database inspection"""
        print("=" * 80)
        print("🔍 DATABASE SCHEMA & DATA INSPECTOR")
        print("=" * 80)
        print(f"📂 Database: {self.db_path}")
        print()
        
        # Get all tables
        self.get_all_tables()
        
        # Inspect each table
        for table_name in self.schema_info.keys():
            self.inspect_table(table_name)
        
        # Generate summary report
        self.generate_summary_report()
        
        # Save detailed results
        self.save_results()
    
    def get_all_tables(self):
        """Get list of all tables in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("""
                SELECT name, type, sql 
                FROM sqlite_master 
                WHERE type='table' 
                ORDER BY name
            """)
            
            tables = cursor.fetchall()
            
            print(f"📊 Found {len(tables)} tables:")
            for table in tables:
                table_name = table['name']
                self.schema_info[table_name] = {
                    'type': table['type'],
                    'creation_sql': table['sql'],
                    'columns': [],
                    'indexes': [],
                    'row_count': 0,
                    'sample_data': []
                }
                print(f"   - {table_name}")
            
            conn.close()
            print()
            
        except Exception as e:
            print(f"❌ Error getting tables: {e}")
            sys.exit(1)
    
    def inspect_table(self, table_name: str):
        """Inspect detailed structure and data for a specific table"""
        print(f"🔎 Inspecting table: {table_name}")
        print("-" * 60)
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            print(f"📋 Columns ({len(columns)}):")
            for col in columns:
                col_info = {
                    'name': col['name'],
                    'type': col['type'],
                    'not_null': bool(col['notnull']),
                    'default_value': col['dflt_value'],
                    'primary_key': bool(col['pk'])
                }
                self.schema_info[table_name]['columns'].append(col_info)
                
                # Format column info
                pk_marker = " 🔑" if col_info['primary_key'] else ""
                not_null_marker = " ⚠️" if col_info['not_null'] else ""
                default = f" (default: {col_info['default_value']})" if col_info['default_value'] else ""
                
                print(f"   {col_info['name']:<20} {col_info['type']:<15}{pk_marker}{not_null_marker}{default}")
            
            # Get indexes
            cursor.execute(f"PRAGMA index_list({table_name})")
            indexes = cursor.fetchall()
            
            if indexes:
                print(f"\n📚 Indexes ({len(indexes)}):")
                for idx in indexes:
                    cursor.execute(f"PRAGMA index_info({idx['name']})")
                    idx_columns = cursor.fetchall()
                    col_names = [col['name'] for col in idx_columns]
                    
                    idx_info = {
                        'name': idx['name'],
                        'unique': bool(idx['unique']),
                        'columns': col_names
                    }
                    self.schema_info[table_name]['indexes'].append(idx_info)
                    
                    unique_marker = " 🔒" if idx_info['unique'] else ""
                    print(f"   {idx_info['name']:<30} on ({', '.join(col_names)}){unique_marker}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
            row_count = cursor.fetchone()['count']
            self.schema_info[table_name]['row_count'] = row_count
            
            print(f"\n📊 Row count: {row_count:,}")
            
            # Get sample data if table has data
            if row_count > 0:
                # Get first few rows
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample_rows = cursor.fetchall()
                
                # Get most recent rows if there's a timestamp column
                timestamp_columns = ['timestamp', 'created_at', 'updated_at', 'time']
                timestamp_col = None
                for col in timestamp_columns:
                    if any(c['name'] == col for c in self.schema_info[table_name]['columns']):
                        timestamp_col = col
                        break
                
                recent_rows = []
                if timestamp_col:
                    try:
                        cursor.execute(f"SELECT * FROM {table_name} ORDER BY {timestamp_col} DESC LIMIT 3")
                        recent_rows = cursor.fetchall()
                    except:
                        pass
                
                # Display sample data
                print(f"\n📄 Sample data (first 3 rows):")
                for i, row in enumerate(sample_rows, 1):
                    row_dict = dict(row)
                    print(f"   Row {i}:")
                    for key, value in row_dict.items():
                        # Truncate long values
                        if isinstance(value, str) and len(value) > 50:
                            value = value[:47] + "..."
                        print(f"     {key}: {value}")
                    print()
                
                if recent_rows and timestamp_col:
                    print(f"📄 Most recent data (by {timestamp_col}):")
                    for i, row in enumerate(recent_rows, 1):
                        row_dict = dict(row)
                        print(f"   Recent Row {i}:")
                        for key, value in row_dict.items():
                            if isinstance(value, str) and len(value) > 50:
                                value = value[:47] + "..."
                            print(f"     {key}: {value}")
                        print()
                
                # Store sample data
                self.schema_info[table_name]['sample_data'] = [dict(row) for row in sample_rows]
                
                # Get data freshness info
                if timestamp_col:
                    cursor.execute(f"""
                        SELECT 
                            MIN({timestamp_col}) as oldest,
                            MAX({timestamp_col}) as newest,
                            COUNT(*) as total
                        FROM {table_name}
                    """)
                    freshness = cursor.fetchone()
                    
                    print(f"📅 Data freshness:")
                    print(f"   Oldest: {freshness['oldest']}")
                    print(f"   Newest: {freshness['newest']}")
                    print(f"   Total: {freshness['total']:,} records")
                    
                    # Check for recent data (last 24 hours)
                    cursor.execute(f"""
                        SELECT COUNT(*) as recent_count 
                        FROM {table_name} 
                        WHERE {timestamp_col} >= datetime('now', '-24 hours')
                    """)
                    recent_count = cursor.fetchone()['recent_count']
                    print(f"   Recent (24h): {recent_count:,} records")
                    
                    self.schema_info[table_name]['freshness'] = {
                        'oldest': freshness['oldest'],
                        'newest': freshness['newest'],
                        'total': freshness['total'],
                        'recent_24h': recent_count
                    }
            
            conn.close()
            print("\n" + "=" * 60 + "\n")
            
        except Exception as e:
            print(f"❌ Error inspecting table {table_name}: {e}")
            print("\n" + "=" * 60 + "\n")
    
    def generate_summary_report(self):
        """Generate summary report of database structure"""
        print("📋 DATABASE SUMMARY REPORT")
        print("=" * 80)
        
        total_tables = len(self.schema_info)
        total_rows = sum(table['row_count'] for table in self.schema_info.values())
        
        print(f"📊 Overview:")
        print(f"   Total tables: {total_tables}")
        print(f"   Total rows: {total_rows:,}")
        print()
        
        # Table sizes
        print(f"📈 Table sizes (by row count):")
        sorted_tables = sorted(
            self.schema_info.items(), 
            key=lambda x: x[1]['row_count'], 
            reverse=True
        )
        
        for table_name, info in sorted_tables:
            print(f"   {table_name:<25} {info['row_count']:>10,} rows")
        print()
        
        # Tables with recent data
        print(f"🕐 Recent data (last 24 hours):")
        recent_data_tables = []
        for table_name, info in self.schema_info.items():
            if 'freshness' in info and info['freshness']['recent_24h'] > 0:
                recent_data_tables.append((table_name, info['freshness']['recent_24h']))
        
        if recent_data_tables:
            recent_data_tables.sort(key=lambda x: x[1], reverse=True)
            for table_name, recent_count in recent_data_tables:
                print(f"   {table_name:<25} {recent_count:>10,} recent rows")
        else:
            print("   ⚠️  No tables have data from the last 24 hours!")
        print()
        
        # Critical tables for volatility method
        print(f"🎯 Critical tables for volatility analysis:")
        critical_tables = ['price_history', 'market_data', 'technical_indicators']
        
        for table_name in critical_tables:
            if table_name in self.schema_info:
                info = self.schema_info[table_name]
                status = "✅" if info['row_count'] > 0 else "❌"
                recent = ""
                if 'freshness' in info:
                    recent_count = info['freshness']['recent_24h']
                    recent = f" ({recent_count:,} recent)"
                print(f"   {status} {table_name:<20} {info['row_count']:>10,} rows{recent}")
            else:
                print(f"   ❌ {table_name:<20} {'NOT FOUND':>15}")
        print()
        
        # Column analysis for critical tables
        if 'price_history' in self.schema_info:
            print(f"🔍 price_history table analysis:")
            ph_info = self.schema_info['price_history']
            required_columns = ['token', 'timestamp', 'price', 'volume', 'market_cap']
            
            for col_name in required_columns:
                has_col = any(col['name'] == col_name for col in ph_info['columns'])
                status = "✅" if has_col else "❌"
                print(f"   {status} {col_name}")
            print()
        
        if 'market_data' in self.schema_info:
            print(f"🔍 market_data table analysis:")
            md_info = self.schema_info['market_data']
            required_columns = ['chain', 'timestamp', 'price', 'volume']
            
            for col_name in required_columns:
                has_col = any(col['name'] == col_name for col in md_info['columns'])
                status = "✅" if has_col else "❌"
                print(f"   {status} {col_name}")
            print()
        
        # Data quality issues
        print(f"⚠️  Potential data quality issues:")
        issues = []
        
        # Check for empty critical tables
        for table_name in critical_tables:
            if table_name in self.schema_info and self.schema_info[table_name]['row_count'] == 0:
                issues.append(f"Table '{table_name}' is empty")
        
        # Check for stale data
        for table_name, info in self.schema_info.items():
            if 'freshness' in info and info['freshness']['recent_24h'] == 0 and info['row_count'] > 0:
                issues.append(f"Table '{table_name}' has no recent data (older than 24 hours)")
        
        # Check for missing columns
        if 'price_history' in self.schema_info:
            ph_cols = [col['name'] for col in self.schema_info['price_history']['columns']]
            missing_cols = [col for col in ['token', 'timestamp', 'price'] if col not in ph_cols]
            if missing_cols:
                issues.append(f"price_history missing critical columns: {', '.join(missing_cols)}")
        
        if issues:
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        else:
            print("   ✅ No obvious data quality issues detected")
        print()
    
    def save_results(self):
        """Save detailed inspection results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"database_inspection_{timestamp}.json"
            
            # Prepare data for JSON serialization
            json_data = {
                'database_path': self.db_path,
                'inspection_time': datetime.now().isoformat(),
                'summary': {
                    'total_tables': len(self.schema_info),
                    'total_rows': sum(table['row_count'] for table in self.schema_info.values())
                },
                'tables': self.schema_info
            }
            
            with open(filename, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            print(f"💾 Detailed inspection results saved to: {filename}")
            
        except Exception as e:
            print(f"⚠️  Could not save inspection results: {e}")

def main():
    """Main execution function"""
    print("🔍 Database Schema & Data Inspector")
    print("This script will show you all tables, columns, and data in your database")
    print()
    
    # Check for custom database path
    db_path = None
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
        if not os.path.exists(db_path):
            print(f"❌ Database file not found: {db_path}")
            return
    
    # Initialize inspector
    try:
        inspector = DatabaseInspector(db_path)
        inspector.inspect_all()
    except KeyboardInterrupt:
        print("\n\n⏹️  Inspection interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Inspection failed with error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()