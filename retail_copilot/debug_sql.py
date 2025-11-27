import sqlite3

def debug_top_category_query():
    conn = sqlite3.connect("data/northwind.sqlite")
    
    # The exact SQL being generated
    query = """
    SELECT c.CategoryName, SUM(od.Quantity) as total_quantity
    FROM orders o
    JOIN order_items od ON o.OrderID = od.OrderID
    JOIN products p ON od.ProductID = p.ProductID
    JOIN Categories c ON p.CategoryID = c.CategoryID
    WHERE strftime('%Y', o.OrderDate) = '2022'
    AND strftime('%m', o.OrderDate) IN ('06', '07', '08')
    GROUP BY c.CategoryName
    ORDER BY total_quantity DESC
    LIMIT 1
    """
    
    print("🔍 Executing SQL query:")
    print(query)
    print()
    
    cursor = conn.execute(query)
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    
    print(f"📊 Query returned {len(rows)} rows")
    print(f"📋 Columns: {columns}")
    
    if rows:
        print("📦 Row data:")
        for i, row in enumerate(rows):
            print(f"  Row {i}: {row}")
            print(f"  - CategoryName: {row[0]} (type: {type(row[0])})")
            print(f"  - total_quantity: {row[1]} (type: {type(row[1])})")
    else:
        print("❌ No rows returned - no data for summer 2022")
    
    # Check if we have any orders in summer 2022
    print("\n🔍 Checking if we have orders in summer 2022:")
    check_query = """
    SELECT COUNT(*) as order_count 
    FROM orders 
    WHERE strftime('%Y', OrderDate) = '2022'
    AND strftime('%m', OrderDate) IN ('06', '07', '08')
    """
    cursor = conn.execute(check_query)
    order_count = cursor.fetchone()[0]
    print(f"   Orders in summer 2022: {order_count}")
    
    conn.close()

if __name__ == "__main__":
    debug_top_category_query()