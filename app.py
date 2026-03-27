from pathlib import Path
import re
import time

import streamlit as st

from model_utils import generate_response_stream, load_model_and_tokenizer
from chat_history import (
    create_conversation,
    save_conversation,
    load_conversation,
    delete_conversation,
    list_conversations,
)




# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="AI Assistant — Local LLM",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_css(file_name):
    css_path = Path(__file__).resolve().parent / file_name
    if not css_path.exists():
        return ""
    return css_path.read_text(encoding="utf-8")


def extract_thinking_and_answer(buffer_text):
    """
    Parse streamed text into:
    - visible thinking summary between <think>...</think>
    - final answer outside think tags
    """
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    think_blocks = [
        chunk.strip()
        for chunk in think_pattern.findall(buffer_text)
        if chunk and chunk.strip()
    ]
    completed_thinking = "\n\n".join(think_blocks).strip()

    latest_open_start = buffer_text.lower().rfind("<think>")
    latest_close = buffer_text.lower().rfind("</think>")
    partial_thinking = ""
    answer_candidate = buffer_text

    if latest_open_start != -1 and latest_close < latest_open_start:
        partial_thinking = buffer_text[latest_open_start + len("<think>") :].strip()
        answer_candidate = buffer_text[:latest_open_start]

    answer_text = think_pattern.sub("", answer_candidate)
    answer_text = answer_text.replace("<think>", "").replace("</think>", "").strip()

    if partial_thinking:
        if completed_thinking:
            completed_thinking = f"{completed_thinking}\n\n{partial_thinking}"
        else:
            completed_thinking = partial_thinking

    return completed_thinking.strip(), answer_text.strip()


def _relative_time(iso_str: str) -> str:
    """Convert ISO timestamp to a human-friendly relative label."""
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(iso_str)
        now = datetime.now(timezone.utc)
        diff = now - dt
        seconds = diff.total_seconds()
        if seconds < 60:
            return "Just now"
        if seconds < 3600:
            m = int(seconds // 60)
            return f"{m}m ago"
        if seconds < 86400:
            h = int(seconds // 3600)
            return f"{h}h ago"
        if seconds < 604800:
            d = int(seconds // 86400)
            return f"{d}d ago"
        return dt.strftime("%b %d")
    except Exception:
        return ""


# ── Load Styles ───────────────────────────────────────────────
css_content = load_css("style.css")
if css_content:
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False
if "device_name" not in st.session_state:
    st.session_state.device_name = "Unknown"
if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None
if "pending_delete" not in st.session_state:
    st.session_state.pending_delete = None
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ── Generation defaults (hardcoded) ───────────────────────────
temperature = 0.7
max_tokens = 512
top_p = 0.9
max_history_messages = 12

# ── Theme Toggle ──────────────────────────────────────────────
is_dark = st.session_state.dark_mode
theme_icon = "☀️" if is_dark else "🌙"

# Apply dark-theme CSS class via inline script (runs in parent document, not iframe)
dark_class_css = ""
if is_dark:
    dark_class_css = """
    <style>
    .stApp {
        --bg-primary:     #0D0F14 !important;
        --bg-secondary:   #161921 !important;
        --bg-tertiary:    #1C1F2A !important;
        --bg-elevated:    rgba(34, 38, 51, 0.85) !important;
        --bg-glass:       rgba(22, 25, 33, 0.72) !important;
        --surface:        #1A1D26 !important;
        --surface-hover:  #22263A !important;
        --surface-active: #2A2E42 !important;
        --border-subtle:  rgba(255, 255, 255, 0.06) !important;
        --border-default: rgba(255, 255, 255, 0.10) !important;
        --border-strong:  rgba(255, 255, 255, 0.16) !important;
        --text-primary:   #E8E9ED !important;
        --text-secondary: #9CA3AF !important;
        --text-tertiary:  #6B7280 !important;
        --text-inverse:   #0D0F14 !important;
        --accent-hover:   #7F78FF !important;
        --accent-subtle:  rgba(108, 99, 255, 0.12) !important;
        --accent-glow:    rgba(108, 99, 255, 0.25) !important;
        --success:        #34D399 !important;
        --error:          #F87171 !important;
        --error-subtle:   rgba(248, 113, 113, 0.12) !important;
        --user-bg:        linear-gradient(135deg, rgba(108,99,255,0.14), rgba(108,99,255,0.06)) !important;
        --user-border:    rgba(108, 99, 255, 0.18) !important;
        --assistant-bg:   linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)) !important;
        --assistant-border: rgba(255, 255, 255, 0.06) !important;
        --hero-gradient:  linear-gradient(145deg, #1C1F2A, #161921) !important;
        --shadow-sm:   0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2) !important;
        --shadow-md:   0 4px 14px rgba(0,0,0,0.35), 0 2px 6px rgba(0,0,0,0.2) !important;
        --shadow-lg:   0 12px 40px rgba(0,0,0,0.45), 0 4px 12px rgba(0,0,0,0.25) !important;
        --sidebar-brand-color: #E8E9ED !important;
    }
    .stApp, .stApp [data-testid="stAppViewContainer"],
    .stApp [data-testid="stHeader"], .stApp [data-testid="stToolbar"] {
        background-color: #0D0F14 !important;
        color: #E8E9ED !important;
    }
    [data-testid="stSidebar"] {
        background: #161921 !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #E8E9ED !important;
    }
    .stApp::before {
        background:
            radial-gradient(ellipse 80% 50% at 10% 10%, rgba(108,99,255,0.06), transparent),
            radial-gradient(ellipse 60% 40% at 90% 20%, rgba(52,211,153,0.04), transparent),
            radial-gradient(ellipse 70% 60% at 50% 90%, rgba(108,99,255,0.03), transparent) !important;
    }
    /* ── Bottom bar / chat input container ── */
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    .stChatInput,
    [data-testid="stChatInput"],
    [data-testid="stChatInputContainer"] {
        background-color: #0D0F14 !important;
        background: #0D0F14 !important;
    }
    [data-testid="stChatInput"] {
        border-color: rgba(255, 255, 255, 0.10) !important;
        background: #161921 !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #E8E9ED !important;
        background: transparent !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #6B7280 !important;
    }
    </style>
    """
    st.markdown(dark_class_css, unsafe_allow_html=True)


def _switch_conversation(conv_id: str):
    """Save current conversation, then load the selected one."""
    # Save current if it has messages
    if st.session_state.active_conversation_id and st.session_state.messages:
        save_conversation(st.session_state.active_conversation_id, st.session_state.messages)

    # Load target
    conv = load_conversation(conv_id)
    if conv:
        st.session_state.active_conversation_id = conv_id
        st.session_state.messages = conv.get("messages", [])
    else:
        st.session_state.active_conversation_id = None
        st.session_state.messages = []


def _new_conversation():
    """Save current, create a fresh conversation."""
    if st.session_state.active_conversation_id and st.session_state.messages:
        save_conversation(st.session_state.active_conversation_id, st.session_state.messages)
    conv_id = create_conversation()
    st.session_state.active_conversation_id = conv_id
    st.session_state.messages = []


def _delete_conv(conv_id: str):
    """Delete a conversation and reset if it was active."""
    delete_conversation(conv_id)
    if st.session_state.active_conversation_id == conv_id:
        st.session_state.active_conversation_id = None
        st.session_state.messages = []


# ── Main Page Top Bar ─────────────────────────────────────────
top_left, top_right = st.columns([20, 1])
with top_right:
    # Use HTML to align to the right, and inject the button after
    st.markdown("<div style='display:flex;justify-content:flex-end;width:100%;'>", unsafe_allow_html=True)
    if st.button(theme_icon, key="theme_toggle", help="Toggle dark/light mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ── Hero Section ──────────────────────────────────────────────
st.markdown(
    """
    <section class='app-hero'>
        <p class='app-kicker'>realestate transection assistant</p>
    </section>
    """,
    unsafe_allow_html=True,
)

# ── Load Model ────────────────────────────────────────────────
try:
    with st.spinner("Starting inference engine..."):
        reasoning_mode = "off"
        client, _, device_name = load_model_and_tokenizer(
            "local-qwen",
            reasoning_mode=reasoning_mode,
            reasoning_format="deepseek-legacy",
            reasoning_budget=256,
        )
        if not st.session_state.model_ready:
            st.session_state.model_ready = True
            st.session_state.device_name = device_name
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.markdown(
    f"<span class='runtime-chip'>{st.session_state.device_name}</span>",
    unsafe_allow_html=True,
)

# ── Chat Display ──────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    role = message.get("role")
    content = message.get("content")
    if role in {"user", "assistant"} and content:
        with st.chat_message(role):
            st.markdown(content)

# ── Chat Input & Response ────────────────────────────────────
if prompt := st.chat_input("Message AI Assistant..."):
    prompt = prompt.strip()
    if not prompt:
        st.stop()

    # Auto-create a conversation if none is active
    if not st.session_state.active_conversation_id:
        conv_id = create_conversation()
        st.session_state.active_conversation_id = conv_id

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                streamer = generate_response_stream(
                    client,
                    "qwen-2.5-0.5b",
                    st.session_state.messages,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    max_history_messages=max_history_messages,
                )

                full_stream_text = ""
                final_answer_text = ""
                answer_placeholder = st.empty()

                with st.status("Processing...", expanded=False) as status:
                    st.write("Analyzing context...")
                    time.sleep(0.5)
                    st.write("Generating response...")
                    time.sleep(0.5)
                    status.update(
                        label="Streaming response", state="running", expanded=False
                    )

                for chunk in streamer:
                    piece = str(chunk)
                    full_stream_text += piece

                    thinking_text, final_answer_text = extract_thinking_and_answer(
                        full_stream_text
                    )

                    answer_placeholder.markdown(final_answer_text or "▍")

                # Finalize
                response_text = final_answer_text.strip()
                if not response_text:
                    response_text = "No response was generated. Please try again."

                answer_placeholder.markdown(response_text)

                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )

                # Auto-save after each response
                save_conversation(
                    st.session_state.active_conversation_id,
                    st.session_state.messages,
                )

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "I hit an internal error while generating a response. Please retry.",
                    }
                )
