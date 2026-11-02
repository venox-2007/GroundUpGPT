"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  ChatMessage,
  ConversationSummary,
  ConversationDetail,
  GenerationSettings,
  ModelInfo,
  ModelProfilesResponse,
  AttachmentItem,
} from "./types";
import { Sidebar } from "./components/Sidebar";
import { EmptyState } from "./components/EmptyState";
import { Composer } from "./components/Composer";
import { ChatMessageItem } from "./components/ChatMessageItem";
import { SettingsModal } from "./components/SettingsModal";
import { streamSSE } from "./utils/sse";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") || "http://localhost:8000";

const DEFAULT_SETTINGS: GenerationSettings = {
  temperature: 0.8,
  top_k: 40,
  top_p: 0.9,
  max_new_tokens: 64,
  seed: null,
  repetition_penalty: 1.15,
  nativeOnly: false,
};

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [attachments, setAttachments] = useState<AttachmentItem[]>([]);
  const [knowledgeSessionId, setKnowledgeSessionId] = useState<string | null>(null);

  // Layout & UI state
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const [settingsOpen, setSettingsOpen] = useState<boolean>(false);
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  // History preference: Default OFF as strictly required
  const [saveHistory, setSaveHistory] = useState<boolean>(false);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);

  // Model & Server state
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [profiles, setProfiles] = useState<ModelProfilesResponse | null>(null);
  const [activeProfile, setActiveProfile] = useState<string | null>(null);
  const [isSwitchingModel, setIsSwitchingModel] = useState<boolean>(false);
  const [settings, setSettings] = useState<GenerationSettings>(DEFAULT_SETTINGS);

  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 1. Initial Load: read localStorage preferences and backend state
  useEffect(() => {
    // Check saved history preference
    const storedHistoryPref = localStorage.getItem("groundupgpt_save_history");
    if (storedHistoryPref !== null) {
      setSaveHistory(storedHistoryPref === "true");
    }

    // Check saved theme preference
    const storedTheme = localStorage.getItem("groundupgpt_theme");
    if (storedTheme === "light" || storedTheme === "dark") {
      setTheme(storedTheme);
      if (storedTheme === "light") {
        document.documentElement.classList.add("light");
      } else {
        document.documentElement.classList.remove("light");
      }
    }

    // Default sidebar closed on small viewports
    if (typeof window !== "undefined" && window.innerWidth < 768) {
      setSidebarOpen(false);
    }

    // Fetch initial backend state
    fetchModelInfo();
    fetchProfiles();
    fetchConversations();
  }, []);

  // Auto-scroll messages to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const fetchModelInfo = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/model/info`);
      if (res.ok) {
        const data = await res.json();
        setModelInfo(data);
        if (data.active_profile) setActiveProfile(data.active_profile);
      }
    } catch {
      // Backend offline or unreachable
    }
  };

  const fetchProfiles = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/model/profiles`);
      if (res.ok) {
        const data = await res.json();
        setProfiles(data);
        if (data.active_profile) setActiveProfile(data.active_profile);
      }
    } catch {
      // Backend offline
    }
  };

  const fetchConversations = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/history/conversations`);
      if (res.ok) {
        const data = await res.json();
        setConversations(data);
      }
    } catch {
      // ignore
    }
  };

  const handleToggleTheme = (newTheme: "dark" | "light") => {
    setTheme(newTheme);
    localStorage.setItem("groundupgpt_theme", newTheme);
    if (newTheme === "light") {
      document.documentElement.classList.add("light");
    } else {
      document.documentElement.classList.remove("light");
    }
  };

  const handleToggleSaveHistory = (val: boolean) => {
    setSaveHistory(val);
    localStorage.setItem("groundupgpt_save_history", String(val));
    if (val) {
      fetchConversations();
    }
  };

  // --- File Attachment Handler ---
  const handleAddFiles = async (files: FileList) => {
    setErrorBanner(null);

    // Ensure we have an active isolated session
    let sessId = knowledgeSessionId;
    if (!sessId) {
      try {
        const sRes = await fetch(`${API_BASE_URL}/api/knowledge/session`, { method: "POST" });
        if (sRes.ok) {
          const sData = await sRes.json();
          sessId = sData.session_id;
          setKnowledgeSessionId(sessId);
        } else {
          setErrorBanner("Failed to initialize document upload session.");
          return;
        }
      } catch (err) {
        setErrorBanner("Cannot reach server to initialize upload session.");
        return;
      }
    }

    if (!sessId) {
      setErrorBanner("Could not establish document session.");
      return;
    }

    // Process each selected file
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const tempId = `att-${Date.now()}-${i}`;

      const newAtt: AttachmentItem = {
        id: tempId,
        name: file.name,
        size: file.size,
        type: file.type || "text/plain",
        status: "uploading",
      };

      setAttachments((prev) => [...prev, newAtt]);

      // Upload and extract locally on server
      const formData = new FormData();
      formData.append("session_id", sessId);
      formData.append("file", file);

      try {
        const upRes = await fetch(`${API_BASE_URL}/api/knowledge/upload`, {
          method: "POST",
          body: formData,
        });

        if (upRes.ok) {
          const upData = await upRes.json();
          setAttachments((prev) =>
            prev.map((a) =>
              a.id === tempId
                ? {
                    ...a,
                    status: "ready",
                    sourceId: upData.source_id,
                    warning: upData.warning,
                  }
                : a
            )
          );
        } else {
          const errData = await upRes.json().catch(() => null);
          const detail = errData?.detail || `Upload failed (${upRes.status})`;
          setAttachments((prev) =>
            prev.map((a) =>
              a.id === tempId ? { ...a, status: "error", error: detail } : a
            )
          );
          setErrorBanner(`${file.name}: ${detail}`);
        }
      } catch (err: any) {
        setAttachments((prev) =>
          prev.map((a) =>
            a.id === tempId
              ? { ...a, status: "error", error: "Upload failed (network error)" }
              : a
          )
        );
        setErrorBanner(`Failed to upload ${file.name}`);
      }
    }
  };

  const handleRemoveAttachment = async (id: string) => {
    const target = attachments.find((a) => a.id === id);
    setAttachments((prev) => prev.filter((a) => a.id !== id));
    if (target?.sourceId && knowledgeSessionId) {
      try {
        await fetch(`${API_BASE_URL}/api/knowledge/source/${target.sourceId}`, {
          method: "DELETE",
        });
      } catch {
        // ignore
      }
    }
  };

  // --- Send Message & Stream Response ---
  const handleSubmit = async () => {
    const trimmedInput = input.trim();
    if (!trimmedInput && attachments.length === 0) return;
    if (isGenerating) return;

    setErrorBanner(null);

    // Save prompt and attachments reference
    const promptText = trimmedInput || `[Attached: ${attachments.map((a) => a.name).join(", ")}]`;
    const sentAttachments = [...attachments];

    // If history saving is ON and no active conversation, create one
    let targetConvId = activeConversationId;
    if (saveHistory && !targetConvId) {
      try {
        const titleText = promptText.length > 35 ? promptText.slice(0, 32) + "..." : promptText;
        const cRes = await fetch(`${API_BASE_URL}/api/history/conversations`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: titleText }),
        });
        if (cRes.ok) {
          const cData = await cRes.json();
          targetConvId = cData.id;
          setActiveConversationId(targetConvId);
          fetchConversations();
        }
      } catch {
        // continue without SQLite history if failed
      }
    }

    // User Message
    const userMsgId = `user-${Date.now()}`;
    const userMsg: ChatMessage = {
      id: userMsgId,
      role: "user",
      content: promptText,
      attachments: sentAttachments.length > 0 ? sentAttachments : undefined,
    };

    // Assistant Message Placeholder
    const assistantMsgId = `asst-${Date.now()}`;
    const initialAssistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      isStreaming: true,
      origin: "neural",
      modelProfile: activeProfile ?? undefined,
    };

    setMessages((prev) => [...prev, userMsg, initialAssistantMsg]);
    setInput("");
    setAttachments([]);
    setIsGenerating(true);

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    const requestMode = settings.nativeOnly ? "native" : "auto";

    const payload = {
      prompt: promptText,
      mode: requestMode,
      max_new_tokens: settings.max_new_tokens,
      temperature: settings.temperature,
      top_k: settings.top_k,
      top_p: settings.top_p,
      seed: settings.seed,
      repetition_penalty: settings.repetition_penalty,
      knowledge_session_id: knowledgeSessionId,
      conversation_id: saveHistory ? targetConvId : null,
    };

    try {
      const response = await fetch(`${API_BASE_URL}/api/assistant/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: abortController.signal,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        const errDetail = errorData?.detail || `Server error (${response.status})`;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? { ...m, isStreaming: false, content: `Error: ${errDetail}`, error: errDetail }
              : m
          )
        );
        setIsGenerating(false);
        return;
      }

      await streamSSE(
        response,
        {
          onToken: (tok) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId ? { ...m, content: m.content + tok.text } : m
              )
            );
          },
          onToolResult: (res) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId
                  ? {
                      ...m,
                      content: res.answer_text,
                      origin: res.origin,
                      modelProfile: res.model_profile ?? undefined,
                      toolName: res.tool_name ?? undefined,
                      factId: res.fact_id ?? undefined,
                      provenance: res.provenance ?? undefined,
                      evaluatedExpression: res.evaluated_expression ?? undefined,
                      errorCode: res.error_code ?? undefined,
                      modelGenerated: res.model_generated,
                      details: res.details ?? undefined,
                    }
                  : m
              )
            );
          },
          onDone: (done) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId
                  ? {
                      ...m,
                      isStreaming: false,
                      origin: done.origin || m.origin,
                      modelGenerated: done.model_generated ?? m.modelGenerated,
                      metrics: {
                        promptTokens: done.prompt_tokens,
                        generatedTokens: done.generated_tokens,
                        totalTokens: done.total_tokens,
                        activeContextTokens: done.active_context_tokens,
                        contextLimit: done.context_limit,
                        contextCropped: done.context_cropped,
                        elapsedTimeSec: done.elapsed_time_sec,
                        tokensPerSecond: done.tokens_per_second,
                      },
                    }
                  : m
              )
            );
            setIsGenerating(false);
            if (saveHistory) fetchConversations();
          },
          onError: (errMsg) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId
                  ? { ...m, isStreaming: false, error: errMsg, content: m.content || `Error: ${errMsg}` }
                  : m
              )
            );
            setIsGenerating(false);
          },
        },
        abortController.signal
      );
    } catch (err: any) {
      if (!abortController.signal.aborted) {
        const msg = err instanceof Error ? err.message : String(err);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? { ...m, isStreaming: false, error: msg, content: `Error: ${msg}` }
              : m
          )
        );
      }
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsGenerating(false);
    setMessages((prev) =>
      prev.map((m) => (m.isStreaming ? { ...m, isStreaming: false } : m))
    );
  };

  // --- Conversation Management ---
  const handleSelectConversation = async (id: string) => {
    if (isGenerating) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/history/conversations/${id}`);
      if (res.ok) {
        const data: ConversationDetail = await res.json();
        setActiveConversationId(data.id);
        const mapped: ChatMessage[] = (data.messages || []).map((m) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          origin: m.origin,
          modelProfile: m.model_profile ?? undefined,
          details: m.details,
        }));
        setMessages(mapped);
        setAttachments([]);
      }
    } catch {
      // ignore
    }
  };

  const handleNewChat = () => {
    if (isGenerating) return;
    setActiveConversationId(null);
    setMessages([]);
    setAttachments([]);
    setErrorBanner(null);
  };

  const handleRenameConversation = async (id: string, newTitle: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/history/conversations/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: newTitle }),
      });
      if (res.ok) {
        setConversations((prev) =>
          prev.map((c) => (c.id === id ? { ...c, title: newTitle } : c))
        );
      }
    } catch {
      // ignore
    }
  };

  const handleDeleteConversation = async (id: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/history/conversations/${id}`, {
        method: "DELETE",
      });
      if (res.ok) {
        setConversations((prev) => prev.filter((c) => c.id !== id));
        if (activeConversationId === id) {
          setActiveConversationId(null);
          setMessages([]);
        }
      }
    } catch {
      // ignore
    }
  };

  const handleSwitchProfile = async (profileKey: string) => {
    setIsSwitchingModel(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/model/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile_name: profileKey }),
      });
      if (res.ok) {
        setActiveProfile(profileKey);
        fetchModelInfo();
      }
    } catch {
      // ignore
    } finally {
      setIsSwitchingModel(false);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#212121] text-zinc-100 font-sans">
      {/* Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
        onRenameConversation={handleRenameConversation}
        onDeleteConversation={handleDeleteConversation}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      {/* Main Conversation Column */}
      <main className="relative flex flex-1 flex-col h-full min-w-0 bg-[#212121] overflow-hidden">
        {/* Top Navbar Header (Minimal) */}
        <header className="flex h-12 shrink-0 items-center justify-between px-3 sm:px-4 z-10">
          <div className="flex items-center gap-2">
            {!sidebarOpen && (
              <button
                type="button"
                onClick={() => setSidebarOpen(true)}
                title="Open sidebar"
                aria-label="Open sidebar"
                className="flex h-8 w-8 items-center justify-center rounded-lg text-zinc-400 hover:bg-zinc-800 hover:text-white transition"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-5 w-5"
                >
                  <rect width="18" height="18" x="3" y="3" rx="2" />
                  <path d="M9 3v18" />
                </svg>
              </button>
            )}
            <span className="text-xs font-medium text-zinc-400 select-none">
              {settings.nativeOnly
                ? "GroundUpGPT (Native Only)"
                : activeProfile
                ? `GroundUpGPT (${activeProfile})`
                : "GroundUpGPT"}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setSettingsOpen(true)}
              title="Settings"
              aria-label="Settings"
              className="flex h-8 w-8 items-center justify-center rounded-lg text-zinc-400 hover:bg-zinc-800 hover:text-white transition"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-4 w-4"
              >
                <circle cx="12" cy="12" r="3" />
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
              </svg>
            </button>
          </div>
        </header>

        {/* Error Notification Banner */}
        {errorBanner && (
          <div className="mx-4 mb-2 flex items-center justify-between rounded-xl border border-rose-800/80 bg-rose-950/70 px-3.5 py-2 text-xs text-rose-200">
            <span>{errorBanner}</span>
            <button
              type="button"
              onClick={() => setErrorBanner(null)}
              className="ml-2 text-rose-400 hover:text-white"
            >
              ✕
            </button>
          </div>
        )}

        {/* Middle Area: Welcome Empty State or Messages Stream */}
        <div className="flex-1 overflow-y-auto px-4 py-2 sm:px-6">
          <div className="mx-auto flex h-full max-w-3xl flex-col">
            {messages.length === 0 ? (
              <EmptyState />
            ) : (
              <div className="flex flex-col space-y-1 pb-6">
                {messages.map((m, idx) => (
                  <ChatMessageItem
                    key={m.id || idx}
                    message={m}
                    isLastAssistant={idx === messages.length - 1 && m.role === "assistant"}
                    isGenerating={isGenerating}
                    onRegenerate={handleSubmit}
                  />
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>

        {/* Bottom Composer */}
        <Composer
          input={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          onStop={handleStop}
          isGenerating={isGenerating}
          disabled={false}
          attachments={attachments}
          onAddFiles={handleAddFiles}
          onRemoveAttachment={handleRemoveAttachment}
        />
      </main>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        theme={theme}
        onToggleTheme={handleToggleTheme}
        saveHistory={saveHistory}
        onToggleSaveHistory={handleToggleSaveHistory}
        modelInfo={modelInfo}
        profiles={profiles}
        activeProfile={activeProfile}
        onSwitchProfile={handleSwitchProfile}
        settings={settings}
        onUpdateSettings={setSettings}
        isSwitchingModel={isSwitchingModel}
      />
    </div>
  );
}
