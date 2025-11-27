import sqlite3

def check_database():
    conn = sqlite3.connect("data/northwind.sqlite")
    
    # Check what years we have orders for
    print("Checking order years:")
    cursor = conn.execute("""
    SELECT DISTINCT strftime('%Y', OrderDate) as year 
    FROM orders 
    WHERE OrderDate IS NOT NULL 
    ORDER BY year
    """)
    years = cursor.fetchall()
    print(f"Years with orders: {years}")
    
    # Check date range
    print("\nChecking date range:")
    cursor = conn.execute("SELECT MIN(OrderDate), MAX(OrderDate) FROM orders")
    date_range = cursor.fetchone()
    print(f"Date range: {date_range[0]} to {date_range[1]}")
    
    # Check sample orders
    print("\nSample orders:")
    cursor = conn.execute("SELECT OrderID, OrderDate FROM orders ORDER BY OrderDate LIMIT 5")
    sample_orders = cursor.fetchall()
    for order in sample_orders:
        print(f"  Order {order[0]}: {order[1]}")
    
    conn.close()

if __name__ == "__main__":
    check_database()