# Retail Analytics Copilot

A local AI agent for retail analytics using hybrid RAG + SQL approach. This system answers complex business questions by combining document retrieval with database queries.

## 🎯 Project Overview

This project implements a Retail Analytics Copilot that answers complex business questions by combining:

- **RAG (Retrieval-Augmented Generation)** over local policy documents
- **SQL queries** over the Northwind sample database  
- **Hybrid reasoning** that uses both documents and database data
- **Local execution** with no external API dependencies

## 📊 Final Results Summary

| Question ID | Confidence | Status | Type | Notes |
|-------------|------------|--------|------|-------|
| rag_policy_beverages_return_days | 0.7 | ✅ | RAG | Pure document retrieval |
| hybrid_top_category_qty_summer_1997 | 0.77 | ✅ | Hybrid | SQL + document constraints |
| hybrid_aov_winter_1997 | 1.0 | ✅ | Hybrid | SQL + document constraints |
| sql_top3_products_by_revenue_alltime | 0.9 | ✅ | SQL | Pure database query |
| hybrid_revenue_beverages_summer_1997 | 1.0 | ✅ | Hybrid | SQL + document constraints |
| hybrid_best_customer_margin_1997 | 1.0 | ✅ | Hybrid | SQL + document constraints |

**Success Rate: 6/6 (100%)**  
**Average Confidence: 0.895**

## 🏗️ System Architecture

### 6-Node LangGraph Workflow:
1. **Router** - Classifies questions as RAG, SQL, or Hybrid using rule-based logic
2. **Retriever** - TF-IDF search over document chunks with similarity scoring
3. **Planner** - Extracts constraints (dates, KPIs, categories) from documents
4. **SQL Generator** - Creates SQL queries based on question type and constraints
5. **Executor** - Runs SQL queries on SQLite database with error handling
6. **Synthesizer** - Combines results into final answer with proper formatting

### Key Features:
- **🔁 Repair Loop**: Automatic SQL error recovery (2 attempts)
- **📚 Citation Tracking**: Documents sources for both docs and DB tables
- **🎯 Confidence Scoring**: Heuristic based on retrieval + SQL success
- **✅ Format Validation**: Ensures output matches required format hints
- **📅 Date Adaptation**: Maps requested 1997 dates to actual 2022 database data

## 📁 Project Structure


## 📁 Project Structure
retail_copilot/
├── run_agent_hybrid.py # Main CLI entry point
├── run_agent_hybrid_final.py # Final working version
├── sample_questions_hybrid_eval.jsonl # Evaluation questions
├── outputs_hybrid.jsonl # Generated results
├── test_condition.py # Debugging utilities
├── check_database_years.py # Database inspection
├── data/
│ └── northwind.sqlite # SQLite database (2012-2023 data)
├── docs/ # Policy documents
│ ├── marketing_calendar.md
│ ├── kpi_definitions.md
│ ├── product_policy.md
│ └── catalog.md
└── README.md



## 🚀 Usage

```bash
# Run the retail analytics agent
python run_agent_hybrid_final.py \
  --batch sample_questions_hybrid_eval.jsonl \
  --out outputs_hybrid.jsonl