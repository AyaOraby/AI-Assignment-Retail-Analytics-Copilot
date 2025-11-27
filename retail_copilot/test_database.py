import sqlite3

def test_database():
    conn = sqlite3.connect("data/northwind.sqlite")
    
    # Test if we have orders in Summer 1997
    print("Testing Summer 1997 orders:")
    cursor = conn.execute("SELECT COUNT(*) FROM orders WHERE OrderDate BETWEEN '1997-06-01' AND '1997-06-30'")
    summer_orders = cursor.fetchone()[0]
    print(f"Summer orders: {summer_orders}")
    
    # Test if we have orders in Winter 1997
    print("\nTesting Winter 1997 orders:")
    cursor = conn.execute("SELECT COUNT(*) FROM orders WHERE OrderDate BETWEEN '1997-12-01' AND '1997-12-31'")
    winter_orders = cursor.fetchone()[0]
    print(f"Winter orders: {winter_orders}")
    
    # Test if we have orders in 1997
    print("\nTesting 1997 orders:")
    cursor = conn.execute("SELECT COUNT(*) FROM orders WHERE OrderDate BETWEEN '1997-01-01' AND '1997-12-31'")
    all_1997_orders = cursor.fetchone()[0]
    print(f"All 1997 orders: {all_1997_orders}")
    
    # Test categories
    print("\nTesting categories:")
    cursor = conn.execute("SELECT CategoryName FROM Categories")
    categories = cursor.fetchall()
    print(f"Categories: {categories}")
    
    conn.close()

if __name__ == "__main__":
    test_database()