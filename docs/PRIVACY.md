# Privacy & Local Storage

GroundUpGPT is designed with a strict local-first, privacy-by-default architecture.

## 1. Opt-In Chat Persistence (Default OFF)

- **Zero Background Writes**: By default, conversations exist exclusively in ephemeral React frontend state. The local SQLite database (`chat_history.db`) is completely inactive and records zero writes unless explicitly turned on.
- **Explicit Activation**: Chat persistence must be opted into via the **Settings** dialog by toggling **"Save chats locally"**.
- **User Control**: When enabled, previous conversations can be searched, renamed, or deleted directly from the sidebar. Deleting a conversation executes an immediate cascading deletion of all associated messages.

## 2. Temporary Knowledge Sessions & Document Ingestion

- **Ephemeral In-Memory Indexing**: When documents (`.txt`, `.md`, `.pdf`, `.docx`) are attached via the chat composer, text is parsed into bounded chunks (30–70 BPE tokens) stored in an in-memory session store (`src/user_knowledge.py`).
- **Session Isolation**: Each upload session receives an unpredictable UUID (`ksess_<hex>`). Chunks are segregated per session and capped at 16 KB total text.
- **Session Purge**: Closing the browser session or deleting the active conversation purges the associated in-memory knowledge chunks.

## 3. 100% Air-Gapped Local Processing

- **No Remote APIs**: All tokenization, inference, and document extraction execute within the local Python runtime.
- **Zero Telemetry**: GroundUpGPT contains no analytics trackers, crash-reporting daemons, or remote network calls.

## 4. Privacy Limitations & Guidance

- **Unencrypted Local Disk**: If chat persistence is enabled, conversations in `chat_history.db` are stored as unencrypted text on the local filesystem.
- **Single-Tenant Localhost**: The application does not provide per-user password protection or workspace isolation. Anyone with access to the host machine or local port can inspect history.
- **Sensitive Data Guidance**: Do not upload proprietary credentials, API keys, or sensitive personal information into local chat sessions.
