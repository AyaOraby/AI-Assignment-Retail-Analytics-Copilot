
# Retail Analytics Hybrid Agent

##  Run Output

```
PS C:\Users\Aya Oraby\retail_copilot> python run_agent_hybrid_final.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
🚀 Initializing Retail Analytics Agent...

==================================================
📝 Processing rag_policy_beverages_return_days
❓ According to the product policy, what is the return window (days) for unopened Beverages? Return an integer.      
🔍 Processing: rag_policy_beverages_return_days
   🚦 Routing question...
   Route: rag
   📄 Retrieving documents...
   Found 2 relevant chunks
   🎯 Synthesizing answer...
   ✅ Final answer: 14
   📊 Confidence: 0.7

==================================================
📝 Processing hybrid_top_category_qty_summer_1997
❓ During 'Summer Beverages 1997' as defined in the marketing calendar, which product category had the highest total quantity sold? Return {category:str, quantity:int}.
🔍 Processing: hybrid_top_category_qty_summer_1997
   🚦 Routing question...
   Route: hybrid
   📄 Retrieving documents...
   Found 3 relevant chunks
   🗄️  Generating SQL...
   📅 Using year: 2022 (equivalent to requested 1997)
   🔍 Conditions - category: True, quantity: True, summer: True, highest: True
   🎯 Generating SQL for top category by quantity (summer)
   ⚡ Executing SQL...
   ✅ SQL returned 1 rows
   🎯 Synthesizing answer...
   📦 SQL returned rows: [('Confections', 60941)]
   🔍 Extracted - category: 'Confections', quantity: 60941
   ✅ Final answer: {'category': 'Confections', 'quantity': 60941}
   📊 Confidence: 1.0

==================================================
📝 Processing hybrid_aov_winter_1997
❓ Using the AOV definition from the KPI docs, what was the Average Order Value during 'Winter Classics 1997'? Return a float rounded to 2 decimals.
🔍 Processing: hybrid_aov_winter_1997
   🚦 Routing question...
   Route: hybrid
   📄 Retrieving documents...
   Found 3 relevant chunks
   🗄️  Generating SQL...
   📅 Using year: 2022 (equivalent to requested 1997)
   🔍 Conditions - category: False, quantity: False, summer: False, highest: False
   🎯 Generating SQL for AOV (winter)
   ⚡ Executing SQL...
   ✅ SQL returned 1 rows
   🎯 Synthesizing answer...
   📦 SQL returned rows: [(29144.21,)]
   ✅ Final answer: 29144.21
   📊 Confidence: 1.0

==================================================
📝 Processing sql_top3_products_by_revenue_alltime
❓ Top 3 products by total revenue all-time. Revenue uses Order Details: SUM(UnitPrice*Quantity*(1-Discount)). Return list[{product:str, revenue:float}].
🔍 Processing: sql_top3_products_by_revenue_alltime
   🚦 Routing question...
   Route: sql
   🗄️  Generating SQL...
   📅 Using year: 2022 (equivalent to requested 1997)
   🔍 Conditions - category: False, quantity: True, summer: False, highest: False
   🎯 Generating SQL for top 3 products by revenue
   ⚡ Executing SQL...
   ✅ SQL returned 3 rows
   🎯 Synthesizing answer...
   📦 SQL returned rows: [('Côte de Blaye', 53265895.23), ('Thüringer Rostbratwurst', 24623469.23), ('Mishi Kobe Niku', 19423037.5)]
   ✅ Final answer: [{'product': 'Côte de Blaye', 'revenue': 53265895.23}, {'product': 'Thüringer Rostbratwurst', 'revenue': 24623469.23}, {'product': 'Mishi Kobe Niku', 'revenue': 19423037.5}]
   📊 Confidence: 0.9

==================================================
📝 Processing hybrid_revenue_beverages_summer_1997
❓ Total revenue from the 'Beverages' category during 'Summer Beverages 1997' dates. Return a float rounded to 2 decimals.
🔍 Processing: hybrid_revenue_beverages_summer_1997
   🚦 Routing question...
   Route: hybrid
   📄 Retrieving documents...
   Found 3 relevant chunks
   🗄️  Generating SQL...
   📅 Using year: 2022 (equivalent to requested 1997)
   🔍 Conditions - category: True, quantity: False, summer: True, highest: False
   🎯 Generating SQL for beverages revenue (summer)
   ⚡ Executing SQL...
   ✅ SQL returned 1 rows
   🎯 Synthesizing answer...
   📦 SQL returned rows: [(2171086.0,)]
   ✅ Final answer: 2171086.0
   📊 Confidence: 1.0

==================================================
📝 Processing hybrid_best_customer_margin_1997
❓ Per the KPI definition of gross margin, who was the top customer by gross margin in 1997? Assume CostOfGoods is approximated by 70% of UnitPrice if not available. Return {customer:str, margin:float}.
🔍 Processing: hybrid_best_customer_margin_1997
   🚦 Routing question...
   Route: hybrid
   📄 Retrieving documents...
   Found 2 relevant chunks
   🗄️  Generating SQL...
   📅 Using year: 2022 (equivalent to requested 1997)
   🔍 Conditions - category: False, quantity: False, summer: False, highest: False
   🎯 Generating SQL for gross margin by customer
   ⚡ Executing SQL...
   ✅ SQL returned 1 rows
   🎯 Synthesizing answer...
   📦 SQL returned rows: [('Consolidated Holdings', 241915.53)]
   ✅ Final answer: {'customer': 'Consolidated Holdings', 'margin': 241915.53}
   📊 Confidence: 1.0

🎉 Successfully processed 6 questions
💾 Results written to outputs_hybrid.jsonl

📈 Summary:
  ✅ rag_policy_beverages_return_days: confidence 0.7
  ✅ hybrid_top_category_qty_summer_1997: confidence 1.0
  ✅ hybrid_aov_winter_1997: confidence 1.0
  ✅ sql_top3_products_by_revenue_alltime: confidence 0.9
  ✅ hybrid_revenue_beverages_summer_1997: confidence 1.0
  ✅ hybrid_best_customer_margin_1997: confidence 1.0
PS C:\Users\Aya Oraby\retail_copilot>
```

