from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
import json

class AgentState(BaseModel):
    # Inputs
    question: str
    format_hint: str
    question_id: str
    
    # Processing
    route: Optional[Literal['rag', 'sql', 'hybrid']] = None
    retrieved_context: List[Dict] = Field(default_factory=list)
    generated_sql: str = ""
    sql_result: Dict[str, Any] = Field(default_factory=dict)
    final_answer: Any = None
    explanation: str = ""
    citations: List[str] = Field(default_factory=list)
    
    # Control flow
    error_count: int = 0
    max_repairs: int = 2
    current_step: str = ""

class HybridAgent:
    def __init__(self):
        from agent.dspy_signatures import Router, SQLGenerator, AnswerSynthesizer
        from agent.rag.retrieval import retriever
        from agent.tools.sqlite_tool import sql_tool
        
        self.router = Router()
        self.sql_generator = SQLGenerator()
        self.synthesizer = AnswerSynthesizer()
        self.retriever = retriever
        self.sql_tool = sql_tool
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._route_question)
        workflow.add_node("retriever", self._retrieve_docs)
        workflow.add_node("planner", self._plan_constraints)
        workflow.add_node("sql_generator", self._generate_sql)
        workflow.add_node("executor", self._execute_sql)
        workflow.add_node("synthesizer", self._synthesize_answer)
        
        # Define edges
        workflow.set_entry_point("router")
        
        workflow.add_edge("router", "retriever")
        workflow.add_conditional_edges(
            "retriever",
            self._should_plan,
            {
                "plan": "planner",
                "skip_plan": "synthesizer"
            }
        )
        workflow.add_edge("planner", "sql_generator")
        workflow.add_edge("sql_generator", "executor")
        workflow.add_conditional_edges(
            "executor",
            self._check_sql_result,
            {
                "success": "synthesizer",
                "retry": "sql_generator",
                "fail": "synthesizer"
            }
        )
        workflow.add_conditional_edges(
            "synthesizer",
            self._validate_final_answer,
            {
                "success": END,
                "retry": "planner",
                "fail": END
            }
        )
        
        return workflow.compile()
    
    def _route_question(self, state: AgentState):
        state.current_step = "routing"
        result = self.router.forward(state.question)
        state.route = result.route
        return state
    
    def _retrieve_docs(self, state: AgentState):
        state.current_step = "retrieval"
        if state.route in ['rag', 'hybrid']:
            state.retrieved_context = self.retriever.search(state.question, top_k=3)
        return state
    
    def _should_plan(self, state: AgentState):
        if state.route == 'rag' and not state.retrieved_context:
            return "skip_plan"
        return "plan"
    
    def _plan_constraints(self, state: AgentState):
        state.current_step = "planning"
        # Extract key constraints from retrieved context
        constraints = []
        for ctx in state.retrieved_context:
            constraints.append(f"From {ctx['chunk_id']}: {ctx['content']}")
        state.citations = [ctx['chunk_id'] for ctx in state.retrieved_context]
        return state
    
    def _generate_sql(self, state: AgentState):
        state.current_step = "sql_generation"
        if state.route in ['sql', 'hybrid']:
            schema_info = self.sql_tool.get_schema()
            constraints = "\n".join([ctx['content'] for ctx in state.retrieved_context])
            
            result = self.sql_generator.forward(
                question=state.question,
                schema_info=schema_info,
                constraints=constraints
            )
            state.generated_sql = result.sql_query
        return state
    
    def _execute_sql(self, state: AgentState):
        state.current_step = "execution"
        if state.route in ['sql', 'hybrid'] and state.generated_sql:
            result = self.sql_tool.execute_query(state.generated_sql)
            state.sql_result = result
            
            # Extract table citations from SQL
            if "FROM" in state.generated_sql.upper():
                sql_upper = state.generated_sql.upper()
                tables = ['orders', 'order_items', 'products', 'customers', 'categories']
                for table in tables:
                    if table.upper() in sql_upper:
                        state.citations.append(table)
        return state
    
    def _check_sql_result(self, state: AgentState):
        if state.route in ['sql', 'hybrid']:
            if not state.sql_result.get('success', False):
                state.error_count += 1
                if state.error_count <= state.max_repairs:
                    return "retry"
                return "fail"
        return "success"
    
    def _synthesize_answer(self, state: AgentState):
        state.current_step = "synthesis"
        sql_results_str = json.dumps(state.sql_result.get('rows', []), default=str)
        doc_context = "\n".join([ctx['content'] for ctx in state.retrieved_context])
        
        result = self.synthesizer.forward(
            question=state.question,
            sql_results=sql_results_str,
            document_context=doc_context,
            format_hint=state.format_hint
        )
        
        state.final_answer = self._parse_final_answer(result.final_answer, state.format_hint)
        state.explanation = result.explanation
        state.citations = list(set(state.citations + result.citations))
        
        return state
    
    def _parse_final_answer(self, answer: str, format_hint: str):
        try:
            if format_hint == "int":
                return int(answer.strip())
            elif format_hint == "float":
                return round(float(answer.strip()), 2)
            elif format_hint.startswith("{") and format_hint.endswith("}"):
                return json.loads(answer)
            elif format_hint.startswith("list[{"):
                return json.loads(answer)
            else:
                return answer
        except:
            return answer
    
    def _validate_final_answer(self, state: AgentState):
        if state.final_answer is None or state.final_answer == "":
            state.error_count += 1
            if state.error_count <= state.max_repairs:
                return "retry"
        return "success"
    
    def process_question(self, question_id: str, question: str, format_hint: str):
        state = AgentState(
            question_id=question_id,
            question=question,
            format_hint=format_hint
        )
        
        final_state = self.graph.invoke(state)
        
        return {
            "id": question_id,
            "final_answer": final_state.final_answer,
            "sql": final_state.generated_sql,
            "confidence": max(0.0, 1.0 - (final_state.error_count * 0.3)),
            "explanation": final_state.explanation,
            "citations": final_state.citations
        }