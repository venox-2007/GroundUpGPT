"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  ChatMessage,
  HealthStatus,
  ModelInfo,
  GenerationSettings,
  ModelProfileEntry,
  KnowledgeMode,
  KnowledgeSource,
} from "./types";
import { Sidebar } from "./components/Sidebar";
import { ChatMessageItem } from "./components/ChatMessageItem";
import { ChatInput } from "./components/ChatInput";
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
};

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [profiles, setProfiles] = useState<Record<string, ModelProfileEntry> | null>(null);
  const [activeProfile, setActiveProfile] = useState<string | null>(null);
  const [settings, setSettings] = useState<GenerationSettings>(DEFAULT_SETTINGS);
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(false);
  const [chatMode, setChatMode] = useState<"native" | "assisted">("native");
  const [knowledgeMode, setKnowledgeMode] = useState<KnowledgeMode>("off");
  const [knowledgeSources, setKnowledgeSources] = useState<KnowledgeSource[]>([]);
  const [knowledgePanelOpen, setKnowledgePanelOpen] = useState<boolean>(false);
  const [newDocName, setNewDocName] = useState<string>("");
  const [newDocContent, setNewDocContent] = useState<string>("");
  const [isAddingDoc, setIsAddingDoc] = useState<boolean>(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const fetchKnowledgeSources = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/knowledge/sources?session_id=default`);
      if (res.ok) {
        const data = await res.json();
        setKnowledgeSources(data);
      }
    } catch {
      // ignore
    }
  };

  const handleAddKnowledge = async () => {
    if (!newDocContent.trim()) return;
    try {
      setIsAddingDoc(true);
      const res = await fetch(`${API_BASE_URL}/api/knowledge/source`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: "default",
          display_name: newDocName.trim() || "pasted_note.txt",
          content: newDocContent,
          source_type: "txt",
        }),
      });
      if (res.ok) {
        setNewDocName("");
        setNewDocContent("");
        await fetchKnowledgeSources();
      } else {
        const err = await res.json().catch(() => ({ detail: "Upload failed" }));
        alert(`Failed to add knowledge document: ${err.detail}`);
      }
    } finally {
      setIsAddingDoc(false);
    }
  };

  const handleDeleteKnowledge = async (sourceId: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/knowledge/source/${sourceId}?session_id=default`, {
        method: "DELETE",
      });
      if (res.ok) {
        await fetchKnowledgeSources();
      }
    } catch {
      // ignore
    }
  };

  // Poll server health & model information on mount
  useEffect(() => {
    fetchKnowledgeSources();
    async function checkStatus() {
      try {
        const healthRes = await fetch(`${API_BASE_URL}/api/health`);
        if (healthRes.ok) {
          const healthData = await healthRes.json();
          setHealth(healthData);

          if (healthData.model_loaded) {
            const [infoRes, profRes] = await Promise.all([
              fetch(`${API_BASE_URL}/api/model/info`),
              fetch(`${API_BASE_URL}/api/model/profiles`),
            ]);
            if (infoRes.ok) {
              const infoData = await infoRes.json();
              setModelInfo(infoData);
            }
            if (profRes.ok) {
              const profData = await profRes.json();
              setProfiles(profData.profiles || null);
              if (profData.active_profile) {
                setActiveProfile(profData.active_profile);
              }
            }
          }
        } else {
          setHealth(null);
        }
      } catch (err) {
        setHealth(null);
      }
    }

    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll to bottom of conversation
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isGenerating]);

  // Execute generation stream from prompt
  const executeGeneration = async (promptText: string) => {
    if (!promptText.trim() || isGenerating) return;

    const userMessageId = `user-${Date.now()}`;
    const assistantMessageId = `assistant-${Date.now()}`;

    const userMsg: ChatMessage = {
      id: userMessageId,
      role: "user",
      content: promptText,
    };

    const assistantMsg: ChatMessage = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      isStreaming: true,
      modelProfile: modelInfo?.profile_name || activeProfile || "GroundUpGPT",
      mode: chatMode,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setIsGenerating(true);

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      const response = await fetch(`${API_BASE_URL}/api/assistant/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: promptText,
          mode: chatMode,
          knowledge_mode: knowledgeMode,
          knowledge_session_id: "default",
          max_new_tokens: settings.max_new_tokens,
          temperature: settings.temperature,
          top_k: settings.top_k,
          top_p: settings.top_p,
          seed: settings.seed,
          repetition_penalty: settings.repetition_penalty ?? 1.0,
        }),
        signal: abortController.signal,
      });

      if (!response.ok) {
        let errDetail = `HTTP ${response.status} ${response.statusText}`;
        try {
          const errJson = await response.json();
          if (errJson.detail) errDetail = errJson.detail;
        } catch {
          // ignore parsing error
        }
        throw new Error(errDetail);
      }

      await streamSSE(
        response,
        {
          onToken: (tok) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId
                  ? { ...m, content: m.content + tok.text, origin: m.origin || "neural" }
                  : m
              )
            );
          },
          onToolResult: (toolResult) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId
                  ? {
                      ...m,
                      content: toolResult.answer_text,
                      origin: toolResult.origin,
                      toolName: toolResult.tool_name ?? undefined,
                      factId: toolResult.fact_id ?? undefined,
                      provenance: toolResult.provenance ?? undefined,
                      evaluatedExpression: toolResult.evaluated_expression ?? undefined,
                      errorCode: toolResult.error_code ?? undefined,
                      modelGenerated: toolResult.model_generated,
                      details: toolResult.details ?? undefined,
                    }
                  : m
              )
            );
          },
          onDone: (done) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId
                  ? {
                      ...m,
                      isStreaming: false,
                      origin: m.origin || done.origin || "neural",
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
          },
          onError: (errMsg) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId
                  ? { ...m, isStreaming: false, error: errMsg }
                  : m
              )
            );
          },
        },
        abortController.signal
      );
    } catch (err: unknown) {
      if (abortController.signal.aborted) {
        // Stream deliberately halted by Stop button; preserve partial text
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessageId ? { ...m, isStreaming: false } : m
          )
        );
      } else {
        const message = err instanceof Error ? err.message : String(err);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessageId
              ? { ...m, isStreaming: false, error: message }
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
    }
  };

  const handleRegenerate = () => {
    // Find the last user prompt in message history
    const lastUserIndex = messages.findLastIndex((m) => m.role === "user");
    if (lastUserIndex === -1) return;

    const lastPrompt = messages[lastUserIndex].content;
    // Slice off the last prompt and subsequent messages, then resubmit
    setMessages((prev) => prev.slice(0, lastUserIndex));
    executeGeneration(lastPrompt);
  };

  const handleClearChat = () => {
    if (isGenerating) handleStop();
    setMessages([]);
  };

  const handleSwitchProfile = async (profile: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/model/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile }),
      });
      if (res.ok) {
        const info = await res.json();
        setModelInfo(info);
        setActiveProfile(profile);
      } else {
        const errData = await res.json().catch(() => ({ detail: "Unknown server error" }));
        console.error("Failed to switch model profile:", errData.detail);
        alert(`Failed to load model profile '${profile}': ${errData.detail || res.statusText}`);
      }
    } catch (err) {
      console.error("Failed to switch model profile:", err);
      alert(`Network error switching model profile: ${err}`);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100">
      {/* Sidebar Navigation */}
      <Sidebar
        health={health}
        modelInfo={modelInfo}
        profiles={profiles}
        activeProfileKey={activeProfile}
        settings={settings}
        onUpdateSettings={setSettings}
        onClearChat={handleClearChat}
        onSwitchProfile={handleSwitchProfile}
        isGenerating={isGenerating}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      {/* Main Chat Interface */}
      <main className="flex flex-1 flex-col overflow-hidden">
        {/* Top Navbar */}
        <header className="flex h-14 items-center justify-between border-b border-slate-800 bg-slate-900/60 px-4">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(true)}
              className="rounded p-1 text-slate-400 hover:text-slate-100 md:hidden"
              aria-label="Open sidebar"
            >
              ☰
            </button>
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-slate-200">
                GroundUpGPT
              </span>
              <span className={`rounded px-2 py-0.5 text-[10px] font-mono font-medium border ${
                modelInfo?.is_sft
                  ? "bg-amber-950 text-amber-300 border-amber-800"
                  : "bg-teal-950 text-teal-400 border-teal-800"
              }`}>
                {modelInfo?.is_sft
                  ? `${(modelInfo.total_parameters || 0) > 20000000 ? "27M" : "5M"} SFT #${modelInfo.sft_step ?? modelInfo.checkpoint_step} (Chat) • Active`
                  : modelInfo?.checkpoint_step !== undefined && modelInfo.checkpoint_step !== null
                  ? `Step ${modelInfo.checkpoint_step} • Active`
                  : "Active"}
              </span>
            </div>
          </div>

          {/* Mode & Knowledge Controls */}
          <div className="flex items-center gap-2">
            {/* Mode Selector Toggle */}
            <div className="flex items-center gap-1 rounded-lg bg-slate-950 p-1 border border-slate-800">
              <button
                type="button"
                onClick={() => setChatMode("native")}
                className={`rounded px-2.5 py-1 text-xs font-medium transition ${
                  chatMode === "native"
                    ? "bg-slate-800 text-slate-100 shadow-sm border border-slate-700"
                    : "text-slate-400 hover:text-slate-200"
                }`}
                title="Native GPT: Responses generated by your locally trained Transformer."
              >
                Native GPT
              </button>
              <button
                type="button"
                onClick={() => setChatMode("assisted")}
                className={`rounded px-2.5 py-1 text-xs font-medium transition ${
                  chatMode === "assisted"
                    ? "bg-indigo-950 text-indigo-300 shadow-sm border border-indigo-800"
                    : "text-slate-400 hover:text-slate-200"
                }`}
                title="Tool-Assisted: Uses local tools for supported calculations and facts; other conversations use the selected GPT."
              >
                Tool-Assisted
              </button>
            </div>

            {/* Knowledge Mode Toggle */}
            <div className="flex items-center gap-1 rounded-lg bg-slate-950 p-1 border border-slate-800">
              <span className="text-[10px] font-mono text-slate-500 uppercase px-1">Knowledge:</span>
              <button
                type="button"
                onClick={() => setKnowledgeMode("off")}
                className={`rounded px-2 py-0.5 text-xs font-medium transition ${
                  knowledgeMode === "off"
                    ? "bg-slate-800 text-slate-100 shadow-sm border border-slate-700"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Off
              </button>
              <button
                type="button"
                onClick={() => {
                  setKnowledgeMode("evidence");
                  setKnowledgePanelOpen(true);
                  fetchKnowledgeSources();
                }}
                className={`rounded px-2 py-0.5 text-xs font-medium transition ${
                  knowledgeMode === "evidence"
                    ? "bg-amber-950 text-amber-300 shadow-sm border border-amber-800"
                    : "text-slate-400 hover:text-slate-200"
                }`}
                title="Show Source Evidence: Returns exact matching document passage without model hallucination (0 model calls)."
              >
                Evidence
              </button>
              <button
                type="button"
                onClick={() => {
                  setKnowledgeMode("grounded");
                  setKnowledgePanelOpen(true);
                  fetchKnowledgeSources();
                }}
                className={`rounded px-2 py-0.5 text-xs font-medium transition ${
                  knowledgeMode === "grounded"
                    ? "bg-indigo-950 text-indigo-300 shadow-sm border border-indigo-800"
                    : "text-slate-400 hover:text-slate-200"
                }`}
                title="Experimental Grounded GPT: Injects retrieved passage into model prompt context."
              >
                Grounded
              </button>
              <button
                type="button"
                onClick={() => setKnowledgePanelOpen(!knowledgePanelOpen)}
                className={`ml-1 rounded px-1.5 py-0.5 text-[10px] font-mono border transition ${
                  knowledgePanelOpen
                    ? "border-teal-700 bg-teal-950 text-teal-300"
                    : "border-slate-800 bg-slate-900 text-slate-400 hover:text-slate-200"
                }`}
                title="Toggle Knowledge Document Manager"
              >
                Docs ({knowledgeSources.length}) {knowledgePanelOpen ? "▴" : "▾"}
              </button>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
            {health?.model_loaded ? (
              <span className="hidden sm:inline">
                {modelInfo?.total_parameters?.toLocaleString() ?? 0} parameters
              </span>
            ) : (
              <span className="text-amber-400">Model not loaded</span>
            )}
          </div>
        </header>

        {/* Collapsible Knowledge Document Manager Panel */}
        {knowledgePanelOpen && (
          <div className="border-b border-slate-800 bg-slate-900/90 px-4 py-3 text-xs">
            <div className="flex flex-col md:flex-row gap-4">
              {/* Document Ingestion Form */}
              <div className="flex-1 flex flex-col gap-2">
                <span className="font-semibold text-slate-200 font-mono text-[11px]">
                  + Add Document / Knowledge Passage (Max 16 KB)
                </span>
                <input
                  type="text"
                  placeholder="Document Title (e.g. project_apollo.txt)"
                  value={newDocName}
                  onChange={(e) => setNewDocName(e.target.value)}
                  className="rounded border border-slate-700 bg-slate-950 px-2.5 py-1 text-xs text-slate-100 placeholder-slate-500 focus:border-teal-500 focus:outline-none"
                />
                <textarea
                  rows={3}
                  placeholder="Paste factual text or notes here to ground responses..."
                  value={newDocContent}
                  onChange={(e) => setNewDocContent(e.target.value)}
                  className="rounded border border-slate-700 bg-slate-950 px-2.5 py-1 text-xs text-slate-100 placeholder-slate-500 focus:border-teal-500 focus:outline-none font-mono"
                />
                <div className="flex justify-between items-center">
                  <span className="text-[10px] text-slate-500 font-mono">
                    {newDocContent.length} chars (~{Math.round(newDocContent.length / 4)} tokens)
                  </span>
                  <button
                    type="button"
                    onClick={handleAddKnowledge}
                    disabled={isAddingDoc || !newDocContent.trim()}
                    className="rounded bg-teal-600 px-3 py-1 text-xs font-medium text-white hover:bg-teal-500 disabled:opacity-50"
                  >
                    {isAddingDoc ? "Ingesting..." : "Ingest Document"}
                  </button>
                </div>
              </div>

              {/* Ingested Documents List */}
              <div className="flex-1 flex flex-col gap-1.5 border-t md:border-t-0 md:border-l border-slate-800 md:pl-4">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-slate-200 font-mono text-[11px]">
                    Active Documents in Session ({knowledgeSources.length}/10)
                  </span>
                </div>
                {knowledgeSources.length === 0 ? (
                  <p className="text-[11px] text-slate-500 italic py-2">
                    No documents ingested yet. Add a passage on the left to enable document-attributed retrieval.
                  </p>
                ) : (
                  <div className="max-h-28 overflow-y-auto space-y-1.5 pr-1">
                    {knowledgeSources.map((s) => (
                      <div
                        key={s.source_id}
                        className="flex items-center justify-between rounded border border-slate-800 bg-slate-950/60 px-2.5 py-1 text-[11px]"
                      >
                        <div className="flex flex-col">
                          <span className="font-semibold text-slate-300 font-mono">{s.display_name}</span>
                          <span className="text-[9px] text-slate-500 font-mono">
                            {s.chunk_count} chunks • {s.character_count} chars • {s.token_count} tokens
                          </span>
                        </div>
                        <button
                          type="button"
                          onClick={() => handleDeleteKnowledge(s.source_id)}
                          className="rounded text-rose-400 hover:text-rose-300 text-xs px-1.5 py-0.5 hover:bg-rose-950/40"
                          title="Delete source"
                        >
                          ✕
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Mode Description Sub-Header */}
        <div className="border-b border-slate-800/80 bg-slate-900/40 px-4 py-1.5 text-[11px] text-slate-400 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-300">
              {chatMode === "native" ? "Native GPT" : "Tool-Assisted"}:
            </span>
            <span>
              {chatMode === "native"
                ? "Responses generated by your locally trained Transformer."
                : "Uses local tools for supported calculations and facts; other conversations use the selected GPT."}
            </span>
            {knowledgeMode !== "off" && (
              <span className="ml-2 rounded px-1.5 py-0.2 text-[10px] font-mono border border-amber-800 bg-amber-950/60 text-amber-300">
                Knowledge: {knowledgeMode === "evidence" ? "Evidence Mode (0 model calls)" : "Grounded GPT Mode"}
              </span>
            )}
          </div>
          <span className="hidden sm:inline font-mono text-[10px] text-slate-500">
            Mode: {chatMode} • Knowledge: {knowledgeMode}
          </span>
        </div>

        {/* Message Thread Scroll Area */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="mx-auto flex max-w-3xl flex-col">
            {messages.length === 0 ? (
              <div className="my-auto flex flex-col items-center justify-center py-16 text-center">
                <div className={`flex h-12 w-12 items-center justify-center rounded-xl border font-mono text-lg font-bold ${
                  modelInfo?.is_sft
                    ? "bg-amber-900/40 border-amber-700/60 text-amber-300"
                    : "bg-teal-900/40 border-teal-700/60 text-teal-300"
                }`}>
                  G
                </div>
                <h2 className="mt-4 text-lg font-semibold text-slate-100">
                  {modelInfo?.is_sft ? "GroundUpGPT Assistant (Experimental)" : "GroundUpGPT Web Completion"}
                </h2>
                <p className="mt-1.5 max-w-md text-xs leading-relaxed text-slate-400">
                  {modelInfo?.is_sft
                    ? "5.05M-parameter Transformer fine-tuned on Dolly 15k instruction pairs. Ask a question or command."
                    : "First-principles Transformer language model trained from scratch. Enter a starting prefix to observe next-token autoregressive completions."}
                </p>

                {/* Prompt Suggestions */}
                <div className="mt-8 flex flex-wrap justify-center gap-2">
                  {(modelInfo?.is_sft
                    ? ["What is Python?", "Explain gravity in simple terms", "Write a short poem about coding", "What are the primary colors?"]
                    : ["The ", "Once upon ", "In the ancient ", "GroundUpGPT "]
                  ).map((promptSample) => (
                    <button
                      key={promptSample}
                      onClick={() => executeGeneration(promptSample)}
                      className={`rounded-full border border-slate-800 bg-slate-900 px-3.5 py-1.5 text-xs text-slate-300 transition ${
                        modelInfo?.is_sft
                          ? "hover:border-amber-600 hover:text-amber-300"
                          : "hover:border-teal-600 hover:text-teal-300"
                      }`}
                    >
                      {promptSample} →
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((message, idx) => (
                <ChatMessageItem
                  key={message.id}
                  message={message}
                  isLastAssistant={
                    message.role === "assistant" && idx === messages.length - 1
                  }
                  isGenerating={isGenerating}
                  onRegenerate={handleRegenerate}
                />
              ))
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Chat Input Bar */}
        <ChatInput
          input={input}
          onChange={setInput}
          onSubmit={() => executeGeneration(input)}
          onStop={handleStop}
          isGenerating={isGenerating}
          disabled={!health?.model_loaded}
          isSft={modelInfo?.is_sft}
        />
      </main>
    </div>
  );
}
