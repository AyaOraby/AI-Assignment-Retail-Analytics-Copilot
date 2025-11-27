import dspy
from agent.dspy_signatures import SQLGenerator

# Training examples for SQL generation
train_examples = [
    {
        "question": "What are the top 3 products by revenue?",
        "schema_info": "Table: products\nColumns: ProductID (INT), ProductName (TEXT)...",
        "constraints": "",
        "sql_query": "SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as revenue FROM order_items od JOIN products p ON od.ProductID = p.ProductID GROUP BY p.ProductID, p.ProductName ORDER BY revenue DESC LIMIT 3"
    }
]

def optimize_sql_generator():
    sql_generator = SQLGenerator()
    
    # Simple few-shot optimization
    optimizer = dspy.BootstrapFewShot(metric=sql_accuracy_metric)
    optimized_sql_generator = optimizer.compile(
        sql_generator, 
        trainset=train_examples
    )
    
    return optimized_sql_generator

def sql_accuracy_metric(example, prediction, trace=None):
    # Simple metric: check if SQL executes without error
    from agent.tools.sqlite_tool import sql_tool
    result = sql_tool.execute_query(prediction.sql_query)
    return 1.0 if result['success'] else 0.0