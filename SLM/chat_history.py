"""
Persistent chat history manager.
Stores conversations as JSON files in a local `chat_history/` directory.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

HISTORY_DIR = Path(__file__).resolve().parent / "chat_history"
HISTORY_DIR.mkdir(exist_ok=True)


def _conv_path(conv_id: str) -> Path:
    return HISTORY_DIR / f"{conv_id}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_title(messages: list) -> str:
    """Derive a short title from the first user message."""
    for msg in messages:
        if msg.get("role") == "user":
            text = msg["content"].strip()
            if len(text) > 48:
                return text[:45] + "..."
            return text
    return "New conversation"


# ── CRUD ──────────────────────────────────────────────────────

def create_conversation() -> str:
    """Create and persist a new, empty conversation. Returns the ID."""
    conv_id = uuid.uuid4().hex[:12]
    data = {
        "id": conv_id,
        "title": "New conversation",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "messages": [],
    }
    _conv_path(conv_id).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return conv_id


def save_conversation(conv_id: str, messages: list) -> None:
    """Persist messages for an existing conversation, updating its title."""
    path = _conv_path(conv_id)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {
            "id": conv_id,
            "created_at": _now_iso(),
            "messages": [],
        }
    data["messages"] = messages
    data["updated_at"] = _now_iso()
    data["title"] = generate_title(messages)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_conversation(conv_id: str) -> dict | None:
    """Load a single conversation by ID. Returns None if not found."""
    path = _conv_path(conv_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def delete_conversation(conv_id: str) -> None:
    """Delete a conversation file."""
    path = _conv_path(conv_id)
    if path.exists():
        path.unlink()


def list_conversations() -> list[dict]:
    """
    Return all conversations sorted by last updated (newest first).
    Each item contains: id, title, updated_at, message_count.
    """
    conversations = []
    for file in HISTORY_DIR.glob("*.json"):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            conversations.append({
                "id": data.get("id", file.stem),
                "title": data.get("title", "Untitled"),
                "updated_at": data.get("updated_at", ""),
                "message_count": len(data.get("messages", [])),
            })
        except (json.JSONDecodeError, OSError):
            continue

    conversations.sort(key=lambda c: c["updated_at"], reverse=True)
    return conversations
