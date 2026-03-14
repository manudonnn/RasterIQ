import os
import pickle
import uuid
import datetime

class ChatManager:
    def __init__(self, data_dir="data/conversations"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def list_chats(self):
        """Return a sorted list of chats (newest first)."""
        chats = []
        for f in os.listdir(self.data_dir):
            if f.endswith(".pkl"):
                path = os.path.join(self.data_dir, f)
                try:
                    with open(path, 'rb') as file:
                        data = pickle.load(file)
                        # Derive a title from the first user message, or default
                        custom_title = data.get("custom_title")
                        if custom_title:
                            title = custom_title
                        else:
                            first_msg = next((m["content"] for m in data.get("messages", []) if m["role"] == "user"), "New Conversation")
                            title = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg
                        
                        chats.append({
                            "id": data["id"],
                            "title": title,
                            "updated_at": data.get("updated_at", 0),
                            "msg_count": len(data.get("messages", [])),
                            "is_important": data.get("is_important", False)
                        })
                except Exception:
                    continue
        
        # Sort by updated_at descending
        return sorted(chats, key=lambda x: x["updated_at"], reverse=True)
        
    def create_chat(self):
        """Create a new empty chat."""
        chat_id = str(uuid.uuid4())
        data = {
            "id": chat_id,
            "created_at": datetime.datetime.now().timestamp(),
            "updated_at": datetime.datetime.now().timestamp(),
            "messages": []
        }
        self.save_chat(chat_id, data["messages"])
        return chat_id
        
    def save_chat(self, chat_id, messages):
        """Persistent save of full rich python objects (plotly figs, etc)."""
        path = os.path.join(self.data_dir, f"{chat_id}.pkl")
        
        # If the file exists, preserve created_at and metadata
        created_at = datetime.datetime.now().timestamp()
        custom_title = None
        is_important = False
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    old_data = pickle.load(f)
                    created_at = old_data.get("created_at", created_at)
                    custom_title = old_data.get("custom_title")
                    is_important = old_data.get("is_important", False)
            except Exception:
                pass
                
        data = {
            "id": chat_id,
            "created_at": created_at,
            "updated_at": datetime.datetime.now().timestamp(),
            "messages": messages,
            "custom_title": custom_title,
            "is_important": is_important
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            
    def load_chat(self, chat_id):
        """Load messages for a chat ID."""
        path = os.path.join(self.data_dir, f"{chat_id}.pkl")
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    return data.get("messages", [])
            except Exception as e:
                print(f"Error loading chat: {e}")
        return []

    def delete_chat(self, chat_id):
        path = os.path.join(self.data_dir, f"{chat_id}.pkl")
        if os.path.exists(path):
            os.remove(path)

    def update_chat_metadata(self, chat_id, **kwargs):
        path = os.path.join(self.data_dir, f"{chat_id}.pkl")
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                for k, v in kwargs.items():
                    data[k] = v
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                print(f"Error updating metadata: {e}")
