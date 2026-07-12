r"""
RAG 药品问答系统 - Streamlit 前端

使用方式:
    cd D:\RAG_project
    streamlit run frontend/streamlit_app.py

功能:
    - 📁 知识库管理：上传药品说明书文件（PDF/DOCX/TXT），自动入库
    - 💬 智能问答：基于 RAG 流程的药品知识问答，支持流式输出
"""

import sys
from dataclasses import asdict
from pathlib import Path

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="药品智能问答系统",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 自定义 CSS
# ============================================================
st.markdown("""
<style>
    /* 主色调 */
    :root {
        --primary: #2563eb;
        --primary-light: #eff6ff;
        --sidebar-bg: #1e293b;
    }
    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #f1f5f9 !important;
    }
    /* 聊天消息 */
    .chat-message {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        line-height: 1.7;
    }
    .chat-message.user {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
    }
    .chat-message.assistant {
        background: #f8fafc;
        border-left: 4px solid #10b981;
    }
    /* 来源引用 */
    .source-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.88rem;
    }
    /* 状态卡片 */
    .stat-card {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        text-align: center;
        margin: 0.3rem 0;
    }
    .stat-card .value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #60a5fa;
    }
    .stat-card .label {
        font-size: 0.75rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 会话状态初始化
# ============================================================
def init_session_state():
    """初始化 Streamlit session_state。"""
    defaults = {
        "messages": [],       # 对话消息列表: [{"role": "user/assistant", "content": "...", "sources": [...]}]
        "chat_history_llm": [],  # LLM 用的对话历史（仅 role+content）
        "processing": False,  # 是否正在处理
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ============================================================
# 懒加载资源（缓存）
# ============================================================
@st.cache_resource(show_spinner=False)
def get_clients():
    """获取数据库客户端（缓存，避免重复连接）。"""
    from app.db.mysql_client import MySQLClient
    from app.db.milvus_client import MilvusClient

    mysql = MySQLClient()
    milvus = MilvusClient()
    try:
        mysql.connect()
    except Exception:
        pass
    try:
        milvus.connect()
    except Exception:
        pass
    return mysql, milvus


# ============================================================
# 侧边栏 — 知识库管理
# ============================================================
def render_sidebar():
    """渲染侧边栏：文件上传 + 知识库状态。"""
    with st.sidebar:
        st.title("💊 药品知识库")
        st.markdown("---")

        # ---- 文件上传 ----
        st.subheader("📁 上传说明书")
        uploaded_file = st.file_uploader(
            "拖拽或点击上传药品说明书",
            type=["pdf", "docx", "txt"],
            help="支持 PDF / DOCX / TXT 格式，单文件上传后自动入库",
            label_visibility="collapsed",
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            drug_name_input = st.text_input(
                "药品名称（可选）",
                placeholder="留空则从文件名自动推断",
                label_visibility="collapsed",
            )
        with col2:
            upload_btn = st.button("🚀 入库", type="primary", use_container_width=True)

        if upload_btn and uploaded_file is not None:
            _handle_upload(uploaded_file, drug_name_input)
        elif upload_btn and uploaded_file is None:
            st.warning("请先选择文件")

        st.markdown("---")

        # ---- 知识库状态 ----
        st.subheader("📊 知识库状态")
        _render_kb_stats()

        st.markdown("---")

        # ---- 操作 ----
        st.subheader("⚙️ 操作")
        if st.button("🔄 刷新状态", use_container_width=True):
            st.rerun()
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history_llm = []
            st.rerun()


def _handle_upload(uploaded_file, drug_name_input: str):
    """处理文件上传并调用离线入库流程。"""
    from app.offline.pipeline import run_pipeline

    # 保存上传文件到 data/uploads/ 目录，使用原始文件名（保留药名信息）
    original_name = Path(uploaded_file.name).name
    upload_dir = PROJECT_ROOT / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = original_name.replace(" ", "_")
    tmp_path = upload_dir / safe_name
    tmp_path.write_bytes(uploaded_file.getvalue())

    drug_name = drug_name_input.strip() if drug_name_input.strip() else None

    with st.spinner(f"正在处理「{uploaded_file.name}」..."):
        try:
            result = run_pipeline(
                file_path=tmp_path,
                drug_name=drug_name,
            )

            if result.status == "completed":
                st.success(
                    f"✅ 「{result.drug_name}」入库成功！\n\n"
                    f"{result.total_chunks} 个文本块已索引到向量库"
                )
            elif result.status == "partial":
                st.warning(
                    f"⚠️ 「{result.drug_name}」部分入库成功\n\n"
                    f"已索引 {result.indexed_chunks}/{result.total_chunks} 个文本块\n"
                    f"原因: {result.error_message}"
                )
            else:
                st.error(f"❌ 入库失败: {result.error_message}")

        except Exception as e:
            st.error(f"❌ 入库异常: {e}")


def _get_kb_stats() -> dict:
    """获取知识库统计信息（每次刷新时重新查询）。"""
    try:
        mysql, milvus = get_clients()
        # 确保连接有效，失败则重连
        try:
            if not mysql.is_connected():
                mysql.connect()
        except Exception:
            mysql.connect()
        try:
            milvus.connect()
        except Exception:
            pass

        drugs = mysql.get_all_drug_names() or []
        drug_count = len(drugs)
        chunk_count = milvus.count() or 0
        mysql_ok = mysql.is_connected()
        milvus_ok = True  # 能调用 count() 说明已连接
        redis_ok = False
        try:
            from app.config import config as cfg
            import redis as redis_lib
            r = redis_lib.Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT, socket_connect_timeout=2)
            r.ping()
            redis_ok = True
            r.close()
        except Exception:
            pass

        return {
            "drug_count": drug_count,
            "chunk_count": chunk_count,
            "mysql_ok": mysql_ok,
            "milvus_ok": milvus_ok,
            "redis_ok": redis_ok,
        }
    except Exception:
        return {
            "drug_count": 0,
            "chunk_count": 0,
            "mysql_ok": False,
            "milvus_ok": False,
            "redis_ok": False,
        }


def _render_kb_stats():
    """渲染知识库状态卡片。"""
    stats = _get_kb_stats()

    # 服务状态
    status_icon = lambda ok: "🟢" if ok else "🔴"
    st.markdown(
        f"{status_icon(stats['milvus_ok'])} Milvus  "
        f"{status_icon(stats['mysql_ok'])} MySQL  "
        f"{status_icon(stats['redis_ok'])} Redis"
    )

    # 数据统计
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"<div class='stat-card'><div class='value'>{stats['drug_count']}</div>"
            f"<div class='label'>药品数量</div></div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"<div class='stat-card'><div class='value'>{stats['chunk_count']}</div>"
            f"<div class='label'>文本块数量</div></div>",
            unsafe_allow_html=True,
        )

    if not any([stats["mysql_ok"], stats["milvus_ok"]]):
        st.warning("⚠️ 数据库服务未连接，请先启动 Docker 服务")


# ============================================================
# 主区域 — 问答聊天
# ============================================================
def render_chat():
    """渲染主聊天区域。"""
    st.title("💬 药品智能问答")

    # 欢迎信息
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#64748b;">
            <div style="font-size:4rem; margin-bottom:1rem;">💊</div>
            <h2 style="color:#334155;">药品知识问答助手</h2>
            <p style="font-size:1rem;">
                可以问我药品的适应症、用法用量、禁忌、不良反应、药物相互作用等问题<br>
                例如：「阿司匹林的适应症是什么？」「布洛芬一次吃多少？」
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 渲染历史消息
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # 显示来源引用
            if msg.get("sources"):
                with st.expander(f"📚 参考来源（{len(msg['sources'])} 条）"):
                    for j, src in enumerate(msg["sources"], 1):
                        drug = src.get("drug_name", "未知")
                        section = src.get("section", "")
                        score = src.get("score", 0)
                        text = src.get("chunk_text", "")

                        section_str = f" · {section}" if section else ""
                        st.markdown(
                            f"<div class='source-box'>"
                            f"<strong>[{j}] {drug}{section_str}</strong> "
                            f"<span style='color:#94a3b8;'>(得分: {score:.4f})</span><br>"
                            f"{text[:300]}{'...' if len(text) > 300 else ''}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    # 聊天输入
    if prompt := st.chat_input("请输入药品相关问题...", key="chat_input"):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        st.session_state.chat_history_llm.append({"role": "user", "content": prompt})

        # 立即显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回答
        with st.chat_message("assistant"):
            _handle_query(prompt)


def _handle_query(query: str):
    """处理用户查询: intent → retrieve → rank → generate → 流式展示。"""
    from app.online.intent import IntentClassifier
    from app.online.retriever import Retriever
    from app.online.ranker import Ranker
    from app.online.generator import Generator
    from app.graph.nodes import chitchat_node

    # ---- 阶段 1: 意图识别 ----
    with st.spinner("🤔 分析意图中..."):
        try:
            classifier = IntentClassifier()
            intent_result = classifier.classify(query)
        except Exception as e:
            intent_result = type("R", (), {"intent": "drug_inquiry", "confidence": 0.5})()
            st.warning(f"意图识别失败，默认视为药品问题: {e}")

    # ---- 闲聊: 直接返回 ----
    if intent_result.intent == "chitchat":
        chitchat_state = chitchat_node({"query": query})
        answer = chitchat_state.get("answer", "你好！有什么可以帮您的吗？")
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": []})
        st.session_state.chat_history_llm.append({"role": "assistant", "content": answer})
        return

    # ---- 拒绝: 非药品问题 ----
    if intent_result.intent == "other":
        answer = (
            "抱歉，这个问题超出了药品知识范围，我无法给出专业回答。\n\n"
            "我是药品知识问答助手，擅长回答药品适应症、用法用量、禁忌、不良反应、药物相互作用等问题。\n\n"
            "您可以换个药品相关的问题试试，我很乐意帮助您！"
        )
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": []})
        st.session_state.chat_history_llm.append({"role": "assistant", "content": answer})
        return

    # ---- 阶段 2: 混合检索 ----
    with st.spinner("🔍 检索知识库中..."):
        try:
            retriever = Retriever()
            search_results = retriever.retrieve(query)
            search_dicts = [asdict(r) for r in search_results]
        except Exception as e:
            st.error(f"检索失败: {e}")
            search_dicts = []

    if not search_dicts:
        answer = (
            "抱歉，未能在知识库中检索到与您问题相关的药品信息。\n\n"
            "建议：\n"
            "1. 尝试使用药品的通用名或商品名进行查询\n"
            "2. 简化问题描述，聚焦于单一药品的查询\n"
            "3. 确认您查询的药品说明书已录入系统"
        )
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": []})
        st.session_state.chat_history_llm.append({"role": "assistant", "content": answer})
        return

    # ---- 阶段 3: 重排序 ----
    with st.spinner("📊 重排序中..."):
        try:
            ranker = Ranker()
            ranked = ranker.rerank(query, search_dicts)
            ranked_dicts = [asdict(r) for r in ranked]
        except Exception:
            # 回退到原始 RRF 排序
            ranked_dicts = sorted(search_dicts, key=lambda x: x.get("score", 0), reverse=True)

    # ---- 阶段 4: 流式生成 ----
    answer_placeholder = st.empty()
    full_answer = ""

    try:
        generator = Generator()
        stream = generator.generate_stream(
            query=query,
            context_docs=ranked_dicts,
            history=st.session_state.chat_history_llm,
        )

        # 逐个 token 流式展示
        for token in stream:
            full_answer += token
            answer_placeholder.markdown(full_answer + "▌")

        # 最终展示（去掉光标）
        answer_placeholder.markdown(full_answer)

    except Exception as e:
        full_answer = f"回答生成失败: {e}\n\n以下是为您检索到的参考资料，供参考。"
        st.error(full_answer)
        answer_placeholder.markdown(full_answer)

    # ---- 保存消息 ----
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": ranked_dicts if full_answer and "回答生成失败" not in full_answer else [],
    })
    st.session_state.chat_history_llm.append({"role": "assistant", "content": full_answer})

    # ---- 展示来源 ----
    if ranked_dicts and "回答生成失败" not in full_answer:
        with st.expander(f"📚 参考来源（{len(ranked_dicts)} 条）"):
            for j, src in enumerate(ranked_dicts, 1):
                drug = src.get("drug_name", "未知")
                section = src.get("section", "")
                score = src.get("score", 0)
                text = src.get("chunk_text", "")

                section_str = f" · {section}" if section else ""
                st.markdown(
                    f"<div class='source-box'>"
                    f"<strong>[{j}] {drug}{section_str}</strong> "
                    f"<span style='color:#94a3b8;'>(得分: {score:.4f})</span><br>"
                    f"{text[:300]}{'...' if len(text) > 300 else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ============================================================
# 主入口
# ============================================================
def main():
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
