# Retail Analytics Hybrid Agent

A hybrid **RAG + SQL generation** system for answering analytical
queries over retail data.

## Project Overview

This project implements a **Retail Analytics Agent** that can answer
business questions using:

-   **RAG (Retrieval-Augmented Generation)** for policy and
    document-based answers\
-   **SQL generation + execution** for data-based analytical queries\
-   **Hybrid mode**, combining both RAG + SQL logic depending on the
    question\
-   **Automatic SQL error repair attempts**\
-   **Confidence scoring** for every final answer

The agent processes batches of questions and outputs results in a
structured JSONL file.

## Features

### Intelligent Routing

Each question is routed to one of the pipelines:

  -----------------------------------------------------------------------
  Route                           Purpose
  ------------------------------- ---------------------------------------
  **rag**                         Retrieve text chunks and synthesize
                                  answers (policies, definitions, rules).

  **sql**                         Generate and run SQL on the SQLite
                                  database.

  **hybrid**                      Combines SQL + RAG when both numeric
                                  and textual reasoning are required.
  -----------------------------------------------------------------------

### SQL Execution Engine

-   Generates SQL from natural language\
-   Executes SQL on SQLite\
-   Detects SQL errors\
-   Attempts auto-repair (e.g., wrong tables, missing columns)\
-   Falls back safely if unrecoverable

### Year Normalization Logic

The dataset contains **2022 data only**, so: - Requests for older years
(e.g., 1997) are automatically mapped to **2022**. - Prevents SQL
failures such as `no such table: order_details`.

### Debugging Mode

The agent includes a debugging utility (`debug_sql.py`) that: -
Generates SQL for a query\
- Runs SQL directly\
- Shows row counts and full results\
- Helps diagnose failed SQL generations

### Output Format

Each processed question is returned with:

-   Answer\
-   Confidence score\
-   Route used\
-   Intermediate reasoning (for debugging)\
-   SQL (if used)\
-   SQL output or error trace

## How to Run the Project

### 1. Install Dependencies

``` bash
pip install -r requirements.txt
```

### 2. Prepare the Database

Ensure `retail.db` exists in the project root.

If needed, verify connectivity:

``` bash
python debug_sql.py --test
```

### 3. Run the Hybrid Agent

Process a JSONL batch of questions:

``` bash
python run_agent_hybrid_final.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
```

### 4. View Output

Results will be saved in:

    outputs_hybrid.jsonl

Each entry includes: - Final answer\
- Confidence\
- Route (rag / sql / hybrid)\
- Reasoning\
- SQL + execution logs (if applicable)

### 5. Debug SQL Queries Manually

If SQL fails or you want to inspect generated queries:

``` bash
python debug_sql.py --query "total revenue for beverages in summer"
```

This will: - Print the generated SQL\
- Execute it\
- Show row counts + result table

## Common Errors Handled

-   Wrong table name (`order_details` vs `order_items`)
-   Missing years (1997 → mapped to 2022)
-   Syntax errors in SQL
-   Empty query results
-   JOIN inconsistencies

## Files

-   `run_agent_hybrid_final.py` --- Main agent runner\
-   `debug_sql.py` --- SQL debugging helper\
-   `outputs_hybrid.jsonl` --- Batch run output\
-   `retail.db` --- SQLite database

## Example Use Cases

✔ "What is the return window for unopened beverages?" → RAG\
✔ "Top category by quantity in summer 2022?" → SQL\
✔ "Revenue + policy conditions for alcoholic beverages?" → Hybrid

## Future Enhancements

-   Add real multi-year datasets\
-   Automatic table schema inference\
-   Advanced SQL repair using AST rewriting

## Final Results Summary

| Question ID | Confidence | Status | Type | 
|-------------|------------|--------|------|
| rag_policy_beverages_return_days | 0.7  | RAG | Pure document retrieval |
| hybrid_top_category_qty_summer_1997 | 0.77  | Hybrid | SQL + document constraints |
| hybrid_aov_winter_1997 | 1.0  | Hybrid | SQL + document constraints |
| sql_top3_products_by_revenue_alltime | 0.9  | SQL | Pure database query |
| hybrid_revenue_beverages_summer_1997 | 1.0  | Hybrid | SQL + document constraints |
| hybrid_best_customer_margin_1997 | 1.0  | Hybrid | SQL + document constraints |

**Success Rate: 6/6 (100%)**  
**Average Confidence: 0.895**

## License

MIT License
