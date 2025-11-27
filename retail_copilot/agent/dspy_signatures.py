import dspy
from typing import Literal

# Configure DSPy with local model
lm = dspy.OllamaLocal(model='phi3:3.8b-mini-instruct-q4_K_M', max_tokens=1000)
dspy.configure(lm=lm)

class RouteQuery(dspy.Signature):
    """Determine if a question requires RAG, SQL, or both."""
    question: str = dspy.InputField()
    route: Literal['rag', 'sql', 'hybrid'] = dspy.OutputField(description="Use 'rag' for policy/docs questions, 'sql' for pure data questions, 'hybrid' for questions needing both")

class GenerateSQL(dspy.Signature):
    """Generate SQL query based on question and database schema."""
    question: str = dspy.InputField()
    schema_info: str = dspy.InputField(description="Database schema information")
    constraints: str = dspy.InputField(description="Constraints from retrieved documents")
    sql_query: str = dspy.OutputField(description="Valid SQLite SELECT query")

class SynthesizeAnswer(dspy.Signature):
    """Synthesize final answer from SQL results and document context."""
    question: str = dspy.InputField()
    sql_results: str = dspy.InputField()
    document_context: str = dspy.InputField()
    format_hint: str = dspy.InputField()
    final_answer: str = dspy.OutputField(description="Final answer matching the format hint exactly")
    explanation: str = dspy.OutputField(description="Brief explanation of how answer was derived")
    citations: list = dspy.OutputField(description="List of cited tables and document chunks")

class Router(dspy.Module):
    def __init__(self):
        super().__init__()
        self.route = dspy.ChainOfThought(RouteQuery)
    
    def forward(self, question):
        return self.route(question=question)

class SQLGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_sql = dspy.ChainOfThought(GenerateSQL)
    
    def forward(self, question, schema_info, constraints):
        return self.generate_sql(
            question=question,
            schema_info=schema_info,
            constraints=constraints
        )

class AnswerSynthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(SynthesizeAnswer)
    
    def forward(self, question, sql_results, document_context, format_hint):
        return self.synthesize(
            question=question,
            sql_results=sql_results,
            document_context=document_context,
            format_hint=format_hint
        )