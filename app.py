import streamlit as st
import time
from google import genai
import pypdf
import json
import re
import sys
import io
import traceback
from datetime import datetime

# ==========================================
# 🌟 向量檢索相關套件
# ==========================================
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False
    st.warning("⚠️ 未安裝 sentence-transformers，將使用簡化版檢索。請執行：pip install sentence-transformers")

# --- 1. 網頁基礎設定 ---
st.set_page_config(
    page_title="AI Agent 論文助手", 
    layout="wide", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 樣式 - 簡約黑灰白風格
st.markdown("""
<style>
    /* 主要容器 */
    .main {
        padding: 0rem 1rem;
    }
    
    /* 聊天訊息容器 */
    .stChatMessage {
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* 標題樣式 */
    h1 {
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    /* 移除所有按鈕的刺眼顏色 */
    .stButton > button {
        background-color: transparent;
        border: 1px solid rgba(250, 250, 250, 0.2);
        color: inherit;
    }
    
    .stButton > button:hover {
        border-color: rgba(250, 250, 250, 0.4);
        background-color: rgba(250, 250, 250, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- 輔助函式：清洗 JSON ---
def clean_json_string(text):
    if not text: return None
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()

# ==========================================
# 🌟 LangChain 風格：向量檢索記憶管理器
# ==========================================
class VectorMemoryManager:
    """
    完全符合圖片流程的向量檢索系統
    流程：Text → Text Splitter → Embedding → VectorStore → Similarity Search
    """
    def __init__(self):
        self.chunks = []  # 儲存文本塊
        self.embeddings = []  # 儲存向量
        self.embedding_model = None
        
        # 初始化 Embedding 模型（步驟 5）
        if VECTOR_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                st.error(f"❌ Embedding 模型載入失敗：{e}")
                self.embedding_model = None
    
    def load_pdf(self, reader):
        """
        步驟 1-4: Local Documents → Unstructured Loader → Text → Text Splitter
        步驟 5-6: Text Chunks → Embedding → VectorStore
        """
        # 步驟 1-2: 載入文件
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text: 
                full_text += text + "\n"
        
        # 步驟 3-4: Text Splitter（分割成 chunks）
        chunk_size = 1000
        chunk_overlap = 200  # 重疊部分，提高檢索連貫性
        
        self.chunks = []
        for i in range(0, len(full_text), chunk_size - chunk_overlap):
            chunk = full_text[i:i + chunk_size]
            if chunk.strip():
                self.chunks.append(chunk)
        
        # 步驟 5-6: Embedding（將文本轉換成向量）
        if self.embedding_model and self.chunks:
            try:
                self.embeddings = self.embedding_model.encode(
                    self.chunks, 
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
            except Exception as e:
                st.error(f"❌ 向量化失敗：{e}")
                self.embeddings = []
        
        return len(self.chunks)
    
    def retrieve(self, query, top_k=3):
        """
        步驟 8-11: Query → Embedding → Query Vector → Vector Similarity → Related Text Chunks
        """
        if not self.chunks:
            return "(目前尚無檢索內容)"
        
        # 如果沒有向量模型，使用簡單字串匹配（降級方案）
        if not self.embedding_model or len(self.embeddings) == 0:
            return self._fallback_retrieve(query, top_k)
        
        try:
            # 步驟 8-9: Query → Embedding（將問題轉換成向量）
            query_vector = self.embedding_model.encode(
                [query], 
                convert_to_numpy=True
            )[0]
            
            # 步驟 10: Vector Similarity（計算餘弦相似度）
            similarities = []
            for i, chunk_vector in enumerate(self.embeddings):
                # 餘弦相似度計算
                similarity = np.dot(query_vector, chunk_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(chunk_vector)
                )
                similarities.append((i, similarity))
            
            # 步驟 11: 排序並取出最相關的 chunks
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in similarities[:top_k]]
            
            # 步驟 12: Related Text Chunks
            related_chunks = [self.chunks[i] for i in top_indices]
            retrieved_text = "\n\n---\n\n".join(related_chunks)
            
            return retrieved_text
        
        except Exception as e:
            st.error(f"❌ 向量檢索失敗：{e}")
            return self._fallback_retrieve(query, top_k)
    
    def _fallback_retrieve(self, query, top_k=3):
        """降級方案：簡單字串匹配"""
        relevant_chunks = []
        for chunk in self.chunks:
            if query.lower() in chunk.lower():
                relevant_chunks.append(chunk)
        
        if not relevant_chunks:
            relevant_chunks = self.chunks[:top_k]
        
        return "\n\n---\n\n".join(relevant_chunks[:top_k])

# ==========================================
# 初始化 Session State
# ==========================================
if 'conversations' not in st.session_state:
    st.session_state.conversations = {
        "對話 1": {
            "messages": [],
            "memory_manager": VectorMemoryManager(),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    }

if 'current_conversation' not in st.session_state:
    st.session_state.current_conversation = "對話 1"

if 'conversation_counter' not in st.session_state:
    st.session_state.conversation_counter = 1

# 獲取當前對話
current_conv = st.session_state.conversations[st.session_state.current_conversation]
memory_manager = current_conv["memory_manager"]
messages = current_conv["messages"]

# ==========================================
# 📂 左側邊欄：對話管理 + 文件上傳
# ==========================================
with st.sidebar:
    st.title("🤖 AI Agent 助手")
    
    # 顯示系統狀態
    if VECTOR_AVAILABLE:
        st.success("✅ 向量檢索系統已啟用")
    else:
        st.warning("⚠️ 使用簡化版檢索")
    
    st.divider()
    
    # 新增對話按鈕
    if st.button("➕ 新增對話", use_container_width=True):
        st.session_state.conversation_counter += 1
        new_conv_name = f"對話 {st.session_state.conversation_counter}"
        st.session_state.conversations[new_conv_name] = {
            "messages": [],
            "memory_manager": VectorMemoryManager(),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        st.session_state.current_conversation = new_conv_name
        st.rerun()
    
    st.divider()
    
    # 對話列表
    st.subheader("💬 對話記錄")
    for conv_name in st.session_state.conversations.keys():
        conv_data = st.session_state.conversations[conv_name]
        msg_count = len(conv_data["messages"])
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            if st.button(
                f"{'📌' if conv_name == st.session_state.current_conversation else '💭'} {conv_name} ({msg_count})",
                key=f"conv_{conv_name}",
                use_container_width=True
            ):
                st.session_state.current_conversation = conv_name
                st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"del_{conv_name}", use_container_width=True):
                if len(st.session_state.conversations) > 1:
                    del st.session_state.conversations[conv_name]
                    st.session_state.current_conversation = list(st.session_state.conversations.keys())[0]
                    st.rerun()
                else:
                    st.warning("至少需要保留一個對話")
    
    st.divider()
    
    # 文件上傳區
    st.subheader("📂 文件管理")
    uploaded_file = st.file_uploader("上傳 PDF 文件", type=["pdf"], key="pdf_uploader")
    
    if uploaded_file:
        if not memory_manager.chunks:
            try:
                with st.spinner("🔄 正在處理 PDF（步驟 1-6）..."):
                    reader = pypdf.PdfReader(uploaded_file)
                    chunks_count = memory_manager.load_pdf(reader)
                    
                    if VECTOR_AVAILABLE and len(memory_manager.embeddings) > 0:
                        st.success(f"✅ 已載入 {chunks_count} 個區塊並完成向量化")
                    else:
                        st.success(f"✅ 已載入 {chunks_count} 個區塊（使用簡化版檢索）")
            except Exception as e:
                st.error(f"❌ 讀取失敗：{e}")
        else:
            st.info(f"📄 已載入：{uploaded_file.name}")
    
    # 記憶體狀態
    with st.expander("🧠 VectorStore 狀態"):
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("文本區塊", len(memory_manager.chunks))
        col_m2.metric("向量數量", len(memory_manager.embeddings) if hasattr(memory_manager, 'embeddings') else 0)
    
    st.divider()
    
    # 系統設定
    st.subheader("⚙️ 系統設定")
    
    # ✅ API Key 預設填入
    gemini_api_key = st.text_input(
        "Google API Key", 
        value="",
        type="password"
    )
    
    # ✅ 預設使用 gemma-3-27b-it
    model_options = {
        "gemma-3-27b-it": "🎯 Gemma 3-27B (主要測試)",
        # "gemini-2.5-flash": "⚡ Gemini 2.5 Flash (備用)",
        # "gemini-3-flash": "✨ Gemini 3 Flash (備用)"
    }
    
    model_name = st.selectbox(
        "選擇模型",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0
    )
    
    st.caption("💡 其他可用模型：gemini-2.5-flash, gemini-3-flash")
    
    # 檢索參數設定
    top_k = st.slider("檢索區塊數量 (top_k)", 1, 5, 3)
    
    st.divider()
    
    if st.button("🗑️ 清空當前對話", use_container_width=True):
        current_conv["messages"] = []
        st.rerun()

# ==========================================
# 🖥️ 主要聊天區域
# ==========================================

st.title(f"💬 {st.session_state.current_conversation}")
st.caption(f"建立時間：{current_conv['created_at']} | 訊息數：{len(messages)} | 模型：{model_name}")

chat_container = st.container()

with chat_container:
    if not messages:
        st.info("👋 歡迎使用 AI Agent 論文助手！請上傳 PDF 文件並開始提問。")
        
        st.markdown("**💡 快速開始範例：**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 統計關鍵字", use_container_width=True):
                st.session_state.quick_prompt = "請幫我統計這份文件中出現最多的關鍵字"
        
        with col2:
            if st.button("📝 摘要文件", use_container_width=True):
                st.session_state.quick_prompt = "請幫我摘要這份文件的主要內容"
        
        with col3:
            if st.button("🔍 深入分析", use_container_width=True):
                st.session_state.quick_prompt = "請分析這份文件的研究方法和結論"
    
    # 顯示對話歷史
    # ✅ 修正：使用 enumerate 取得索引 i，並加入 key 以避免 ID 重複錯誤
    for i, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if "thought_process" in msg and msg["thought_process"]:
                with st.expander("🔍 查看 Agent 執行過程"):
                    tab1, tab2, tab3 = st.tabs(["📋 執行計畫", "⚙️ 執行日誌", "📊 檢索內容"])
                    
                    with tab1:
                        st.json(msg["thought_process"])
                    
                    with tab2:
                        if "execution_log" in msg:
                            st.code(msg["execution_log"], language="text")
                    
                    with tab3:
                        if "retrieved_content" in msg:
                            # ✅ 這裡加入了 key=f"retrieved_{i}"
                            st.text_area("檢索到的相關內容", msg["retrieved_content"], height=200, key=f"retrieved_{i}")

# ==========================================
# 💬 底部輸入區
# ==========================================

if hasattr(st.session_state, 'quick_prompt'):
    prompt = st.session_state.quick_prompt
    delattr(st.session_state, 'quick_prompt')
else:
    prompt = st.chat_input("💭 請輸入您的問題...", key="chat_input")

# ==========================================
# 🤖 處理使用者輸入（完整 LangChain 流程）
# ==========================================
if prompt:
    if not gemini_api_key:
        st.error("❌ 請先在左側邊欄輸入 Google API Key！")
        st.stop()
    
    messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        try:
            client = genai.Client(api_key=gemini_api_key)
            
            # 工具卡定義
            TOOL_CARDS = """
            你擁有以下工具 (Tools) 可以使用：
            1. [Vector_Retriever]
               - 功能：使用向量相似度從知識庫檢索最相關的內容
               - 輸入：搜尋問題或關鍵字 (String)
               - 效果：將檢索到的文本存入 context['pdf_content']
               - 技術：Embedding + Cosine Similarity
            
            2. [Python_Interpreter]
               - 功能：執行 Python 程式碼進行計算、統計或資料處理
               - 輸入：Python 程式碼 (String)
               - 可用變數：context['pdf_content']
               - 可用函式：len, sum, max, min, sorted, print, str, int, float 等
            """
            
            recent_messages = messages[-6:] if len(messages) > 6 else messages
            history_str = "\n".join([f"{m['role']}: {m['content'][:200]}..." if len(m['content']) > 200 else f"{m['role']}: {m['content']}" for m in recent_messages])
            
            # Step 1: 規劃
            status_placeholder.info("🤔 正在分析問題並制定計畫...")
            
            planner_prompt = f"""
            你是一個智能 Agent Planner。請根據使用者問題制定執行計畫。
            {TOOL_CARDS}
            
            【對話歷史】：{history_str}
            【當前問題】："{prompt}"
            
            請輸出 JSON 計畫（不要包含 Markdown 標記）：
            {{
                "intent": "使用者意圖描述",
                "reasoning": "選擇這些步驟的原因",
                "steps": [
                    {{"tool": "Vector_Retriever", "args": "搜尋問題"}},
                    {{"tool": "Python_Interpreter", "args": "Python程式碼"}}
                ]
            }}
            """
            
            plan_resp = client.models.generate_content(model=model_name, contents=planner_prompt)
            raw_plan = clean_json_string(plan_resp.text)
            
            try:
                plan_data = json.loads(raw_plan) if raw_plan else {}
            except json.JSONDecodeError:
                plan_data = {"intent": "直接回答", "reasoning": "無法解析計畫", "steps": []}
            
            # Step 2: 執行工具（步驟 8-11）
            status_placeholder.info("⚙️ 正在執行向量檢索...")
            
            execution_logs = []
            context_data = {
                "pdf_content": "",
                "history": history_str,
                "user_query": prompt
            }
            
            retrieved_content = ""
            
            for i, step in enumerate(plan_data.get("steps", [])):
                tool = step.get("tool")
                args = step.get("args")
                
                if tool == "Vector_Retriever":
                    try:
                        # 步驟 8-11: Query → Embedding → Vector Similarity → Related Chunks
                        res = memory_manager.retrieve(args, top_k=top_k)
                        context_data["pdf_content"] = res
                        retrieved_content = res
                        execution_logs.append(f"【向量檢索】問題: {args}\n檢索到 {len(res)} 字符的相關內容")
                    except Exception as e:
                        execution_logs.append(f"【向量檢索錯誤】{e}")
                
                elif tool == "Python_Interpreter":
                    try:
                        old_stdout = sys.stdout
                        redirected_output = io.StringIO()
                        sys.stdout = redirected_output
                        
                        safe_builtins = {
                            "__builtins__": {
                                "len": len, "sum": sum, "max": max, "min": min,
                                "sorted": sorted, "print": print, "str": str,
                                "int": int, "float": float, "list": list,
                                "dict": dict, "set": set, "tuple": tuple,
                                "range": range, "enumerate": enumerate,
                                "zip": zip, "map": map, "filter": filter
                            }
                        }
                        
                        local_scope = {"context": context_data}
                        exec(args, safe_builtins, local_scope)
                        
                        sys.stdout = old_stdout
                        output = redirected_output.getvalue()
                        
                        if output.strip():
                            execution_logs.append(f"【Python執行】\n{output}")
                        else:
                            execution_logs.append("【Python執行】程式執行完成")
                    
                    except Exception as e:
                        sys.stdout = old_stdout
                        execution_logs.append(f"【Python錯誤】{e}")
            
            # ✅ 修正：如果沒有檢索到內容，預設檢索一次（保底方案）
            if not context_data["pdf_content"] and memory_manager.chunks:
                try:
                    context_data["pdf_content"] = memory_manager.retrieve(prompt, top_k=top_k)
                    retrieved_content = context_data["pdf_content"]
                    execution_logs.append(f"【自動檢索】使用問題本身進行檢索，檢索到 {len(retrieved_content)} 字符")
                except Exception as e:
                    execution_logs.append(f"【自動檢索失敗】{e}")
            
            # Step 3: 生成回答（步驟 13-15）
            status_placeholder.info("✍️ 正在生成回答...")
            
            final_context_log = "\n".join(execution_logs)
            
            # 步驟 13: Prompt Template
            final_prompt = f"""
            請根據以下資訊回答使用者問題：
            
            【使用者問題】：{prompt}
            【執行計畫】：{plan_data.get('intent', '未知')}
            【工具執行結果】：
            {final_context_log}
            
            【檢索到的相關內容】（步驟 12: Related Text Chunks）：
            {context_data['pdf_content'][:2000]}{"..." if len(context_data['pdf_content']) > 2000 else ""}
            
            請提供清晰、有條理的回答，並解釋執行結果的意義。
            """
            
            # 步驟 14-15: LLM → Answer
            final_resp = client.models.generate_content(model=model_name, contents=final_prompt)
            response_text = final_resp.text if final_resp.text else "(無回應)"
            
            status_placeholder.empty()
            response_placeholder.markdown(response_text)
            
            messages.append({
                "role": "assistant",
                "content": response_text,
                "thought_process": plan_data,
                "execution_log": final_context_log,
                "retrieved_content": retrieved_content
            })
            
            st.rerun()
        
        except Exception as e:
            status_placeholder.empty()
            response_placeholder.error(f"❌ 發生錯誤：{e}")
            
            with st.expander("🔧 錯誤詳情"):
                st.code(traceback.format_exc())
                st.write("**可能的解決方案：**")
                st.write("1. 檢查 API Key 是否正確")
                st.write("2. 確認網路連線正常")

                st.write("3. 安裝必要套件：pip install sentence-transformers")
