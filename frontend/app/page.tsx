"use client";

import React, { useState, useEffect, useRef } from "react";
import { ChatMessage, HealthStatus, ModelInfo, GenerationSettings } from "./types";
import { Sidebar } from "./components/Sidebar";
import { ChatMessageItem } from "./components/ChatMessageItem";
import { ChatInput } from "./components/ChatInput";
import { streamSSE } from "./utils/sse";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") || "http://localhost:8000";

const DEFAULT_SETTINGS: GenerationSettings = {
  temperature: 1.0,
  top_k: null,
  top_p: 0.9,
  max_new_tokens: 50,
  seed: null,
};

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [settings, setSettings] = useState<GenerationSettings>(DEFAULT_SETTINGS);
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Poll server health & model information on mount
  useEffect(() => {
    async function checkStatus() {
      try {
        const healthRes = await fetch(`${API_BASE_URL}/api/health`);
        if (healthRes.ok) {
          const healthData = await healthRes.json();
          setHealth(healthData);

          if (healthData.model_loaded) {
            const infoRes = await fetch(`${API_BASE_URL}/api/model/info`);
            if (infoRes.ok) {
              const infoData = await infoRes.json();
              setModelInfo(infoData);
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
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setIsGenerating(true);

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      const response = await fetch(`${API_BASE_URL}/api/generate/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: promptText,
          max_new_tokens: settings.max_new_tokens,
          temperature: settings.temperature,
          top_k: settings.top_k,
          top_p: settings.top_p,
          seed: settings.seed,
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
                  ? { ...m, content: m.content + tok.text }
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
      }
    } catch (err) {
      console.error("Failed to switch model profile:", err);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100">
      {/* Sidebar Navigation */}
      <Sidebar
        health={health}
        modelInfo={modelInfo}
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
              <span className="rounded bg-teal-950 px-2 py-0.5 text-[10px] font-mono font-medium text-teal-400 border border-teal-800">
                Step 17 • Active
              </span>
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

        {/* Message Thread Scroll Area */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="mx-auto flex max-w-3xl flex-col">
            {messages.length === 0 ? (
              <div className="my-auto flex flex-col items-center justify-center py-16 text-center">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-teal-900/40 border border-teal-700/60 font-mono text-lg font-bold text-teal-300">
                  G
                </div>
                <h2 className="mt-4 text-lg font-semibold text-slate-100">
                  GroundUpGPT Web Completion
                </h2>
                <p className="mt-1.5 max-w-md text-xs leading-relaxed text-slate-400">
                  First-principles Transformer language model trained from scratch.
                  Enter a starting prefix to observe next-token autoregressive completions.
                </p>

                {/* Prompt Suggestions */}
                <div className="mt-8 flex flex-wrap justify-center gap-2">
                  {["The ", "Once upon ", "In the ancient ", "GroundUpGPT "].map(
                    (promptSample) => (
                      <button
                        key={promptSample}
                        onClick={() => executeGeneration(promptSample)}
                        className="rounded-full border border-slate-800 bg-slate-900 px-3.5 py-1.5 text-xs text-slate-300 transition hover:border-teal-600 hover:text-teal-300"
                      >
                        {promptSample} →
                      </button>
                    )
                  )}
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
        />
      </main>
    </div>
  );
}
