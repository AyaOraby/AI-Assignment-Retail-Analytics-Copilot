import sqlite3
import pandas as pd
from typing import List, Dict, Any

class SQLiteTool:
    def __init__(self, db_path="data/northwind.sqlite"):
        self.db_path = db_path
    
    def get_schema(self) -> str:
        """Get database schema information"""
        schema_info = []
        tables = [
            'orders', 'order_items', 'products', 'customers', 
            'Categories', 'Suppliers', 'Employees'
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            for table in tables:
                try:
                    # Get column info
                    cursor = conn.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    
                    if columns:
                        col_info = [f"{col[1]} ({col[2]})" for col in columns]
                        schema_info.append(f"Table: {table}\nColumns: {', '.join(col_info)}")
                        
                        # Get sample data
                        try:
                            sample = conn.execute(f"SELECT * FROM {table} LIMIT 1").fetchone()
                            if sample:
                                schema_info.append(f"Sample row: {dict(zip([col[1] for col in columns], sample))}")
                        except:
                            pass
                            
                        schema_info.append("")  # Empty line between tables
                except:
                    continue
        
        return "\n".join(schema_info)
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Validate it's a SELECT query for safety
                if not query.strip().upper().startswith('SELECT'):
                    return {"error": "Only SELECT queries are allowed"}
                
                df = pd.read_sql_query(query, conn)
                return {
                    "success": True,
                    "columns": df.columns.tolist(),
                    "rows": df.to_dict('records'),
                    "row_count": len(df)
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "columns": [],
                "rows": [],
                "row_count": 0
            }

# Global SQL tool instance
sql_tool = SQLiteTool()