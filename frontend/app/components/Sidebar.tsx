"use client";

import React, { useState } from "react";
import { HealthStatus, ModelInfo, GenerationSettings, ModelProfileEntry, ConversationSummary } from "../types";

interface SidebarProps {
  health: HealthStatus | null;
  modelInfo: ModelInfo | null;
  profiles?: Record<string, ModelProfileEntry> | null;
  activeProfileKey?: string | null;
  settings: GenerationSettings;
  onUpdateSettings: (newSettings: GenerationSettings) => void;
  onClearChat: () => void;
  onSwitchProfile?: (profile: string) => void;
  isGenerating: boolean;
  isOpen: boolean;
  onClose: () => void;
  conversations?: ConversationSummary[];
  activeConversationId?: string | null;
  onSelectConversation?: (id: string) => void;
  onNewConversation?: () => void;
  onDeleteConversation?: (id: string) => void;
}

export function Sidebar({
  health,
  modelInfo,
  profiles,
  activeProfileKey,
  settings,
  onUpdateSettings,
  onClearChat,
  onSwitchProfile,
  isGenerating,
  isOpen,
  onClose,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
}: SidebarProps) {
  const [showSettings, setShowSettings] = useState(true);

  const isConnected = health?.status === "ok";
  const isLoaded = health?.model_loaded === true;

  const isProfileActive = (key: string): boolean => {
    if (!modelInfo) return false;
    // Rely strictly on backend-provided active profile identity:
    const activeKey = activeProfileKey || modelInfo.active_profile;
    if (activeKey) {
      if (activeKey === key) return true;
      if (key === "5m_step200" && (activeKey === "5m_initial" || activeKey === "5m")) return true;
      return false;
    }
    // Non-overlapping fallback only if active identity is absent:
    const is27m = (modelInfo.total_parameters || 0) > 20000000;
    const profileNameLower = (modelInfo.profile_name || "").toLowerCase();
    const isLatest = profileNameLower.includes("latest");
    const isAnchor = modelInfo.is_anchor_sft === true || profileNameLower.includes("anchor");

    if (key === "27m_anchors_best") {
      return modelInfo.is_sft === true && is27m && isAnchor && !isLatest;
    }
    if (key === "27m_anchors_latest") {
      return modelInfo.is_sft === true && is27m && isAnchor && isLatest;
    }
    if (key === "27m_sft_best") {
      return modelInfo.is_sft === true && is27m && !isAnchor && !isLatest;
    }
    if (key === "27m_sft_latest") {
      return modelInfo.is_sft === true && is27m && !isAnchor && isLatest;
    }
    if (key === "5m_sft_best") {
      return modelInfo.is_sft === true && !is27m && !isLatest;
    }
    if (key === "5m_sft_latest") {
      return modelInfo.is_sft === true && !is27m && isLatest;
    }
    if (key === "5m_best") {
      return !modelInfo.is_sft && modelInfo.checkpoint_step === 1000 && !isLatest;
    }
    if (key === "5m_latest") {
      return !modelInfo.is_sft && modelInfo.checkpoint_step === 1000 && isLatest;
    }
    if (key === "5m_phase2_best") {
      return !modelInfo.is_sft && modelInfo.checkpoint_step === 3000 && !isLatest;
    }
    if (key === "5m_phase2_latest") {
      return !modelInfo.is_sft && modelInfo.checkpoint_step === 3000 && isLatest;
    }
    if (key === "5m_step200") {
      return !modelInfo.is_sft && modelInfo.checkpoint_step === 200;
    }
    if (key === "demo") {
      return modelInfo.vocab_size < 4096;
    }
    return false;
  };

  const getProfileButtonLabel = (key: string, name: string): string => {
    if (key === "27m_anchors_best") return "27M Anchors Best (Experimental)";
    if (key === "27m_anchors_latest") return "27M Anchors Latest";
    if (key === "27m_sft_best") return "27M SFT Best (Chat)";
    if (key === "27m_sft_latest") return "27M SFT Latest";
    if (key === "5m_sft_best") return "5M SFT Best (Chat)";
    if (key === "5m_sft_latest") return "5M SFT Latest";
    if (key === "5m_phase2_best") return "Phase 2 Best";
    if (key === "5m_phase2_latest") return "Phase 2 (3K)";
    if (key === "5m_best") return "Step 1000";
    if (key === "5m_latest") return "Continued (1K)";
    if (key === "5m_step200" || key === "5m_initial") return "Step 200 (5M)";
    if (key === "demo") return "Demo (35K)";
    return name;
  };

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-950/70 md:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-72 flex-col justify-between border-r border-slate-800 bg-slate-900 px-4 py-5 transition-transform duration-200 ease-in-out md:static md:translate-x-0 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex flex-col gap-5 overflow-y-auto pr-1">
          {/* Header & Branding */}
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="flex h-6 w-6 items-center justify-center rounded bg-teal-600 font-mono text-xs font-bold text-white shadow-sm">
                  G
                </span>
                <h1 className="text-base font-semibold tracking-tight text-slate-100">
                  GroundUpGPT
                </h1>
              </div>
              <p className="mt-0.5 text-xs text-slate-400">
                Custom GPT from First Principles
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-slate-400 hover:text-slate-200 md:hidden"
              aria-label="Close sidebar"
            >
              ✕
            </button>
          </div>

          {/* New Conversation & Saved Sessions (Step 46) */}
          <div className="flex flex-col gap-2">
            <button
              type="button"
              onClick={onNewConversation}
              disabled={isGenerating}
              className="flex items-center justify-center gap-2 w-full rounded-md border border-teal-500/40 bg-teal-950/30 px-3 py-2 text-xs font-semibold text-teal-300 hover:bg-teal-900/50 transition disabled:opacity-50"
            >
              <span className="text-sm font-bold">+</span>
              <span>New Conversation</span>
            </button>

            {conversations && conversations.length > 0 && (
              <div className="flex flex-col gap-1 mt-1">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                  Saved Sessions ({conversations.length})
                </span>
                <div className="flex flex-col gap-1 max-h-36 overflow-y-auto pr-1">
                  {conversations.map((c) => {
                    const isActive = activeConversationId === c.id;
                    return (
                      <div
                        key={c.id}
                        className={`group flex items-center justify-between rounded px-2 py-1.5 text-xs transition cursor-pointer ${
                          isActive
                            ? "bg-slate-800 text-teal-300 font-medium border border-teal-500/40"
                            : "text-slate-300 hover:bg-slate-800/60"
                        }`}
                        onClick={() => onSelectConversation && onSelectConversation(c.id)}
                      >
                        <span className="truncate max-w-[170px]" title={c.title}>
                          {c.title}
                        </span>
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] text-slate-400">
                            {c.message_count}
                          </span>
                          {onDeleteConversation && (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                onDeleteConversation(c.id);
                              }}
                              className="text-slate-400 hover:text-rose-400 opacity-0 group-hover:opacity-100 transition p-0.5"
                              title="Delete conversation"
                            >
                              ✕
                            </button>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Model Status Pill */}
          <div className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950/60 p-2.5">
            <span className="text-xs font-medium text-slate-400">Status</span>
            <div className="flex items-center gap-1.5">
              <span
                className={`h-2 w-2 rounded-full ${
                  isLoaded
                    ? "bg-teal-400 ring-2 ring-teal-900"
                    : isConnected
                    ? "bg-amber-400 ring-2 ring-amber-900"
                    : "bg-rose-500 ring-2 ring-rose-950"
                }`}
              />
              <span className="text-xs font-medium text-slate-200">
                {isLoaded
                  ? `Ready (${modelInfo?.device?.toUpperCase() || "OK"})`
                  : isConnected
                  ? "Uninitialized"
                  : "Offline"}
              </span>
            </div>
          </div>

          {/* Model Profile Switcher */}
          {modelInfo && onSwitchProfile && (
            <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-2.5 text-xs">
              <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                Model Profile
              </div>
              <div className="grid grid-cols-2 gap-1.5 font-mono text-[11px]">
                {profiles && Object.keys(profiles).length > 0 ? (
                  Object.entries(profiles).map(([key, prof]) => {
                    const active = isProfileActive(key);
                    return (
                      <button
                        key={key}
                        type="button"
                        disabled={isGenerating || active}
                        onClick={() => onSwitchProfile(key)}
                        className={`rounded px-2 py-1.5 text-center font-medium transition-colors ${
                          active
                            ? "bg-teal-600 text-white font-semibold shadow-sm cursor-default"
                            : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                        }`}
                      >
                        {getProfileButtonLabel(key, prof.name)}
                      </button>
                    );
                  })
                ) : (
                  <>
                    <button
                      type="button"
                      disabled={isGenerating || isProfileActive("5m_best")}
                      onClick={() => onSwitchProfile("5m_best")}
                      className={`rounded px-2 py-1.5 text-center font-medium transition-colors ${
                        isProfileActive("5m_best")
                          ? "bg-teal-600 text-white font-semibold shadow-sm cursor-default"
                          : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                      }`}
                    >
                      Best (5M)
                    </button>
                    <button
                      type="button"
                      disabled={isGenerating || isProfileActive("demo")}
                      onClick={() => onSwitchProfile("demo")}
                      className={`rounded px-2 py-1.5 text-center font-medium transition-colors ${
                        isProfileActive("demo")
                          ? "bg-teal-600 text-white font-semibold shadow-sm cursor-default"
                          : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                      }`}
                    >
                      Demo Model
                    </button>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Model Architectural Specs */}
          {modelInfo && (
            <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3 text-xs text-slate-300">
              <div className="mb-2 flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-400">
                <span>Active Model</span>
                <span className="rounded border border-teal-800/60 bg-teal-950/80 px-1.5 py-0.5 font-mono text-[10px] font-medium text-teal-300">
                  {modelInfo.profile_name || modelInfo.model_name}
                </span>
              </div>
              <div className="space-y-1.5 font-mono text-[11px]">
                <div className="flex justify-between">
                  <span className="text-slate-500">Mode:</span>
                  <span className={`font-semibold ${modelInfo.is_sft ? "text-amber-400" : "text-teal-400"}`}>
                    {modelInfo.is_sft ? "Chat (Instruction-Tuned)" : "Completion (Pretrained LM)"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Parameters:</span>
                  <span className="text-slate-200">
                    {modelInfo.total_parameters.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Layers / Heads:</span>
                  <span className="text-slate-200">
                    {modelInfo.num_layers}L / {modelInfo.num_heads}H
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Context Window:</span>
                  <span className="text-teal-400">
                    {modelInfo.block_size} tokens
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Tokenizer:</span>
                  <span className="text-slate-200">
                    {modelInfo.tokenizer_type} ({modelInfo.tokenizer_vocab_size})
                  </span>
                </div>
                {modelInfo.is_sft ? (
                  <>
                    {modelInfo.is_anchor_sft ? (
                      <>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Pretrain Step:</span>
                          <span className="text-slate-300">
                            #{modelInfo.pretrain_steps ?? 5949}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Original SFT:</span>
                          <span className="text-slate-300">
                            #{modelInfo.original_sft_steps ?? 600}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Anchor SFT:</span>
                          <span className="text-amber-300 font-semibold">
                            +{modelInfo.anchor_sft_steps ?? 250} (Global #{modelInfo.global_step ?? 6799})
                          </span>
                        </div>
                        {modelInfo.anchor_val_loss !== undefined && modelInfo.anchor_val_loss !== null && (
                          <div className="flex justify-between">
                            <span className="text-slate-500">Anchor Loss:</span>
                            <span className="text-teal-400">
                              {modelInfo.anchor_val_loss.toFixed(4)} (PPL {modelInfo.anchor_val_perplexity?.toFixed(1) ?? "4.4"})
                            </span>
                          </div>
                        )}
                        {modelInfo.dolly_val_loss !== undefined && modelInfo.dolly_val_loss !== null && (
                          <div className="flex justify-between">
                            <span className="text-slate-500">Dolly Loss:</span>
                            <span className="text-slate-300">
                              {modelInfo.dolly_val_loss.toFixed(4)} (PPL {modelInfo.dolly_val_perplexity?.toFixed(1) ?? "17.9"})
                            </span>
                          </div>
                        )}
                        {modelInfo.lm_val_loss !== undefined && modelInfo.lm_val_loss !== null && (
                          <div className="flex justify-between">
                            <span className="text-slate-500">LM Benchmark:</span>
                            <span className="text-slate-300">
                              {modelInfo.lm_val_loss.toFixed(4)} (PPL {modelInfo.lm_val_perplexity?.toFixed(1) ?? "39.6"})
                            </span>
                          </div>
                        )}
                      </>
                    ) : (
                      <>
                        {modelInfo.source_step !== undefined && modelInfo.source_step !== null && (
                          <div className="flex justify-between">
                            <span className="text-slate-500">Pretrain Step:</span>
                            <span className="text-slate-300">
                              #{modelInfo.source_step}
                            </span>
                          </div>
                        )}
                        {modelInfo.source_val_loss !== undefined && modelInfo.source_val_loss !== null && (
                          <div className="flex justify-between">
                            <span className="text-slate-500">Pretrain Loss:</span>
                            <span className="text-slate-300">
                              {modelInfo.source_val_loss.toFixed(4)}
                            </span>
                          </div>
                        )}
                        {modelInfo.sft_step !== undefined && modelInfo.sft_step !== null && (
                          <div className="flex justify-between">
                            <span className="text-slate-500">SFT Step:</span>
                            <span className="text-amber-300">
                              #{modelInfo.sft_step} (Global #{modelInfo.global_step ?? modelInfo.checkpoint_step})
                            </span>
                          </div>
                        )}
                        {modelInfo.sft_val_loss !== undefined && modelInfo.sft_val_loss !== null && (
                          <div className="flex justify-between">
                            <span className="text-slate-500">SFT Loss:</span>
                            <span className="text-teal-400">
                              {modelInfo.sft_val_loss.toFixed(4)}
                            </span>
                          </div>
                        )}
                        {modelInfo.sft_val_perplexity !== undefined && modelInfo.sft_val_perplexity !== null && (
                          <div className="flex justify-between">
                            <span className="text-slate-500">SFT Perplexity:</span>
                            <span className="text-teal-400">
                              {modelInfo.sft_val_perplexity.toFixed(1)}
                            </span>
                          </div>
                        )}
                        {modelInfo.lm_val_loss !== undefined && modelInfo.lm_val_loss !== null && (
                          <div className="flex justify-between">
                            <span className="text-slate-500">LM Benchmark:</span>
                            <span className="text-slate-300">
                              {modelInfo.lm_val_loss.toFixed(4)} (PPL {modelInfo.lm_val_perplexity?.toFixed(1)})
                            </span>
                          </div>
                        )}
                      </>
                    )}
                  </>
                ) : (
                  <>
                    {modelInfo.checkpoint_step !== undefined && modelInfo.checkpoint_step !== null && (
                      <div className="flex justify-between">
                        <span className="text-slate-500">Step:</span>
                        <span className="text-slate-200">
                          #{modelInfo.checkpoint_step}
                        </span>
                      </div>
                    )}
                    {modelInfo.original_val_loss !== undefined && modelInfo.original_val_loss !== null && (
                      <div className="flex justify-between">
                        <span className="text-slate-500">Orig Val:</span>
                        <span className="text-teal-400">
                          {modelInfo.original_val_loss.toFixed(4)}
                        </span>
                      </div>
                    )}
                    {modelInfo.expanded_val_loss !== undefined && modelInfo.expanded_val_loss !== null && (
                      <div className="flex justify-between">
                        <span className="text-slate-500">Exp Val:</span>
                        <span className="text-teal-400">
                          {modelInfo.expanded_val_loss.toFixed(4)}
                        </span>
                      </div>
                    )}
                    {modelInfo.checkpoint_perplexity !== undefined && modelInfo.checkpoint_perplexity !== null && (
                      <div className="flex justify-between">
                        <span className="text-slate-500">Perplexity:</span>
                        <span className="text-teal-400">
                          {modelInfo.checkpoint_perplexity.toFixed(1)}
                        </span>
                      </div>
                    )}
                    {modelInfo.checkpoint_val_loss !== undefined && modelInfo.checkpoint_val_loss !== null && modelInfo.original_val_loss === undefined && (
                      <div className="flex justify-between">
                        <span className="text-slate-500">Val Loss:</span>
                        <span className="text-teal-400">
                          {modelInfo.checkpoint_val_loss.toFixed(4)}
                        </span>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          )}

          {/* Inference Settings Accordion */}
          <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
            <button
              type="button"
              onClick={() => setShowSettings(!showSettings)}
              className="flex w-full items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-400 hover:text-slate-200"
            >
              <span>Sampling Settings</span>
              <span>{showSettings ? "−" : "+"}</span>
            </button>

            {showSettings && (
              <div className="mt-3 space-y-3.5 text-xs">
                {/* Temperature */}
                <div>
                  <div className="flex justify-between text-slate-300">
                    <span>Temperature</span>
                    <span className="font-mono text-teal-400">
                      {settings.temperature === 0
                        ? "0.0 (Greedy)"
                        : settings.temperature.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.05"
                    value={settings.temperature}
                    disabled={isGenerating}
                    onChange={(e) =>
                      onUpdateSettings({
                        ...settings,
                        temperature: parseFloat(e.target.value),
                      })
                    }
                    className="mt-1 w-full accent-teal-500"
                  />
                </div>

                {/* Top-p Nucleus */}
                <div>
                  <div className="flex justify-between text-slate-300">
                    <span>Top-p (Nucleus)</span>
                    <span className="font-mono text-teal-400">
                      {settings.top_p !== null ? settings.top_p.toFixed(2) : "1.00"}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={settings.top_p ?? 1.0}
                    disabled={isGenerating}
                    onChange={(e) =>
                      onUpdateSettings({
                        ...settings,
                        top_p: parseFloat(e.target.value),
                      })
                    }
                    className="mt-1 w-full accent-teal-500"
                  />
                </div>

                {/* Top-k */}
                <div>
                  <div className="flex justify-between text-slate-300">
                    <span>Top-k</span>
                    <span className="font-mono text-teal-400">
                      {settings.top_k !== null ? settings.top_k : "Disabled"}
                    </span>
                  </div>
                  <input
                    type="number"
                    min="1"
                    max="100"
                    placeholder="None"
                    value={settings.top_k ?? ""}
                    disabled={isGenerating}
                    onChange={(e) => {
                      const val = e.target.value ? parseInt(e.target.value, 10) : null;
                      onUpdateSettings({ ...settings, top_k: val });
                    }}
                    className="mt-1 w-full rounded border border-slate-800 bg-slate-900 px-2 py-1 text-slate-200 focus:border-teal-500 focus:outline-none"
                  />
                </div>

                {/* Max New Tokens */}
                <div>
                  <div className="flex justify-between text-slate-300">
                    <span>Max New Tokens</span>
                    <span className="font-mono text-teal-400">
                      {settings.max_new_tokens}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="128"
                    step="5"
                    value={settings.max_new_tokens}
                    disabled={isGenerating}
                    onChange={(e) =>
                      onUpdateSettings({
                        ...settings,
                        max_new_tokens: parseInt(e.target.value, 10),
                      })
                    }
                    className="mt-1 w-full accent-teal-500"
                  />
                </div>

                {/* Repetition Penalty */}
                <div>
                  <div className="flex justify-between text-slate-300">
                    <span>Repetition Penalty</span>
                    <span className="font-mono text-teal-400">
                      {(settings.repetition_penalty ?? 1.0).toFixed(2)}
                      {(settings.repetition_penalty ?? 1.0) === 1.0 ? " (Off)" : ""}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1.0"
                    max="2.0"
                    step="0.05"
                    value={settings.repetition_penalty ?? 1.0}
                    disabled={isGenerating}
                    onChange={(e) =>
                      onUpdateSettings({
                        ...settings,
                        repetition_penalty: parseFloat(e.target.value),
                      })
                    }
                    className="mt-1 w-full accent-teal-500"
                  />
                </div>

                {/* Optional Seed */}
                <div>
                  <div className="flex justify-between text-slate-300">
                    <span>Random Seed</span>
                    <span className="font-mono text-teal-400">
                      {settings.seed !== null ? settings.seed : "Random"}
                    </span>
                  </div>
                  <input
                    type="number"
                    placeholder="Random (None)"
                    value={settings.seed ?? ""}
                    disabled={isGenerating}
                    onChange={(e) => {
                      const val = e.target.value ? parseInt(e.target.value, 10) : null;
                      onUpdateSettings({ ...settings, seed: val });
                    }}
                    className="mt-1 w-full rounded border border-slate-800 bg-slate-900 px-2 py-1 text-slate-200 focus:border-teal-500 focus:outline-none"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Educational Disclaimer */}
          <div className="rounded-lg border border-slate-800/80 bg-slate-950/30 p-2.5 text-[11px] leading-relaxed text-slate-400">
            {modelInfo?.is_sft ? (
              <>
                <span className="font-medium text-amber-300">Instruction-Tuned Model:</span>{" "}
                Fine-tuned on Databricks Dolly 15k (CC BY-SA 3.0). Formatted into
                user-assistant turns with stop delimiters. Responses may be simple or
                repetitive given the 5.05M educational parameter scale.
              </>
            ) : (
              <>
                <span className="font-medium text-slate-300">Pretrained LM:</span>{" "}
                Trained on literary English text without instruction tuning. Predicts raw
                next-token continuations from prompts.
              </>
            )}
            <div className="mt-2 text-[10px] text-slate-400 border-t border-slate-800/60 pt-1.5">
              <span className="font-medium text-slate-300">Single-Turn Prompt Context:</span>{" "}
              Model is conditioned on the active turn only to strictly adhere to the 256-token limit.
            </div>
          </div>
        </div>

        {/* Bottom Actions */}
        <div className="border-t border-slate-800 pt-4">
          <button
            type="button"
            onClick={onClearChat}
            disabled={isGenerating}
            className="w-full rounded-md border border-slate-700 bg-slate-800/80 px-3 py-2 text-xs font-medium text-slate-200 transition hover:bg-slate-700 disabled:opacity-50"
          >
            Clear Conversation
          </button>
        </div>
      </aside>
    </>
  );
}
