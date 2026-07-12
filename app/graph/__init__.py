"""
LangGraph RAG 流程编排包

使用方式:
    from app.graph import get_graph, GraphResult, RagState

    graph = get_graph()                    # 获取编译好的图（单例）
    state: RagState = {"query": "阿司匹林一天吃几次？", "history": []}
    result_state = graph.invoke(state)     # 同步执行
    result = GraphResult.from_state(result_state)
    print(result.answer)
"""

from app.graph.graph import build_graph, get_graph
from app.graph.state import GraphResult, RagState

__all__ = [
    "RagState",
    "GraphResult",
    "build_graph",
    "get_graph",
]
