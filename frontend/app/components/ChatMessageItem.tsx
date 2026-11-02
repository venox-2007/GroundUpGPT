"use client";

import React, { useState } from "react";
import { ChatMessage } from "../types";

interface ChatMessageItemProps {
  message: ChatMessage;
  isLastAssistant: boolean;
  isGenerating: boolean;
  onRegenerate?: () => void;
}

export function ChatMessageItem({
  message,
  isLastAssistant,
  isGenerating,
  onRegenerate,
}: ChatMessageItemProps) {
  const isUser = message.role === "user";
  const [showFactDetails, setShowFactDetails] = useState(false);

  // Determine origin display parameters
  const origin = message.origin;
  const isCalculator = origin === "safe_calculator";
  const isLocalFact = origin === "local_fact";
  const isToolError = origin === "tool_error";
  const isDocumentExcerpt = origin === "document_excerpt";
  const isGroundedNeural = origin === "grounded_neural";
  const isNeural = !isUser && (origin === "neural" || isGroundedNeural || (!isCalculator && !isLocalFact && !isToolError && !isDocumentExcerpt));

  const getHeaderBadge = () => {
    if (isUser) {
      return {
        initial: "U",
        iconBg: "bg-slate-700 text-slate-200",
        label: "You (Prompt)",
        tag: null,
      };
    }
    if (isCalculator) {
      return {
        initial: "C",
        iconBg: "bg-sky-600 text-white",
        label: "SafeCalculator — Computed",
        tag: "Computed Math",
        tagBg: "border-sky-800 bg-sky-950/60 text-sky-300",
      };
    }
    if (isLocalFact) {
      return {
        initial: "F",
        iconBg: "bg-emerald-600 text-white",
        label: "Local Fact — Curated Record",
        tag: "Curated Fact",
        tagBg: "border-emerald-800 bg-emerald-950/60 text-emerald-300",
      };
    }
    if (isDocumentExcerpt) {
      return {
        initial: "D",
        iconBg: "bg-amber-600 text-white",
        label: `User Document — ${message.provenance || "Knowledge Excerpt"}`,
        tag: "Excerpt (0 Model Calls)",
        tagBg: "border-amber-800 bg-amber-950/60 text-amber-300",
      };
    }
    if (isGroundedNeural) {
      const profileText = message.modelProfile
        ? `GroundUpGPT (${message.modelProfile})`
        : "GroundUpGPT";
      return {
        initial: "K",
        iconBg: "bg-indigo-600 text-white",
        label: `${profileText} — Grounded Neural`,
        tag: "Grounded GPT",
        tagBg: "border-indigo-800 bg-indigo-950/60 text-indigo-300",
      };
    }
    if (isToolError) {
      return {
        initial: "!",
        iconBg: "bg-rose-600 text-white",
        label: "SafeCalculator — Error",
        tag: "Tool Error",
        tagBg: "border-rose-800 bg-rose-950/60 text-rose-300",
      };
    }
    // Neural default
    const profileText = message.modelProfile
      ? `GroundUpGPT (${message.modelProfile})`
      : "GroundUpGPT";
    return {
      initial: "G",
      iconBg: "bg-teal-600 text-white",
      label: `${profileText} — Neural`,
      tag: "Neural",
      tagBg: "border-slate-700 bg-slate-800/60 text-slate-300",
    };
  };

  const badge = getHeaderBadge();

  return (
    <div
      className={`group flex w-full flex-col gap-1.5 py-4 ${
        isUser ? "items-end" : "items-start"
      }`}
    >
      {/* Role and Truthful Origin Header */}
      <div className="flex items-center gap-2 px-1 text-[11px] font-medium text-slate-400">
        <span
          className={`flex h-4 w-4 items-center justify-center rounded text-[10px] font-bold ${badge.iconBg}`}
        >
          {badge.initial}
        </span>
        <span className="font-semibold text-slate-300">{badge.label}</span>
        {badge.tag && (
          <span
            className={`rounded px-1.5 py-0.2 text-[9px] font-mono border ${badge.tagBg}`}
          >
            {badge.tag}
          </span>
        )}
      </div>

      {/* Message Content Bubble */}
      <div
        className={`max-w-[85%] rounded-lg px-4 py-3 text-sm leading-relaxed sm:max-w-[75%] ${
          isUser
            ? "border border-slate-700 bg-slate-800 text-slate-100"
            : isToolError
            ? "border border-rose-900/60 bg-slate-900/90 text-rose-200"
            : "border border-slate-800 bg-slate-900/90 text-slate-200"
        }`}
      >
        {/* SafeCalculator expression breakdown */}
        {!isUser && isCalculator && message.evaluatedExpression && (
          <div className="mb-2.5 flex items-center gap-2 rounded border border-sky-900/60 bg-sky-950/30 px-2.5 py-1 text-xs font-mono text-sky-400">
            <span className="text-slate-400">Expression:</span>
            <span className="font-semibold text-sky-200">
              {message.evaluatedExpression}
            </span>
            <span className="text-slate-400">=</span>
            <span className="font-bold text-sky-100">{message.content}</span>
          </div>
        )}

        <div className="whitespace-pre-wrap font-sans">
          {message.content}
          {message.isStreaming && (
            <span className="inline-block animate-pulse text-teal-400">▍</span>
          )}
        </div>

        {/* Local Fact Expandable Details */}
        {!isUser && isLocalFact && (
          <div className="mt-2.5 border-t border-slate-800/80 pt-2 text-xs">
            <button
              type="button"
              onClick={() => setShowFactDetails(!showFactDetails)}
              className="flex items-center gap-1 text-[11px] font-mono text-emerald-400 hover:text-emerald-300 transition"
            >
              <span>{showFactDetails ? "▴ Hide Fact Details" : "▾ Curated Fact Details"}</span>
              {message.factId && (
                <span className="text-slate-500 font-mono">({message.factId})</span>
              )}
            </button>
            {showFactDetails && (
              <div className="mt-2 space-y-1.5 rounded border border-slate-800 bg-slate-950/70 p-2.5 font-mono text-[11px] text-slate-300">
                {message.factId && (
                  <div className="flex gap-2">
                    <span className="text-slate-500 w-24">Fact ID:</span>
                    <span className="text-emerald-300 font-semibold">{message.factId}</span>
                  </div>
                )}
                {message.details?.entity && (
                  <div className="flex gap-2">
                    <span className="text-slate-500 w-24">Subject:</span>
                    <span className="text-slate-200">{message.details.entity}</span>
                  </div>
                )}
                <div className="flex gap-2">
                  <span className="text-slate-500 w-24">Provenance:</span>
                  <span className="text-slate-300">
                    {message.provenance || "curated_groundup_facts_v1"}
                  </span>
                </div>
                <div className="flex gap-2">
                  <span className="text-slate-500 w-24">Version:</span>
                  <span className="text-slate-300">{message.details?.version || "1.0"}</span>
                </div>
                <div className="flex gap-2">
                  <span className="text-slate-500 w-24">Model gen:</span>
                  <span className="text-amber-400 font-semibold">false (Deterministic Local)</span>
                </div>
                {message.details?.supporting_passage && (
                  <div className="mt-1.5 pt-1.5 border-t border-slate-800/80 text-[11px] font-sans text-slate-400 italic">
                    "{message.details.supporting_passage}"
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* User Document Excerpt or Grounded Details */}
        {!isUser && (isDocumentExcerpt || isGroundedNeural) && (
          <div className="mt-2.5 border-t border-slate-800/80 pt-2 text-xs">
            <button
              type="button"
              onClick={() => setShowFactDetails(!showFactDetails)}
              className={`flex items-center gap-1 text-[11px] font-mono transition ${
                isDocumentExcerpt
                  ? "text-amber-400 hover:text-amber-300"
                  : "text-indigo-400 hover:text-indigo-300"
              }`}
            >
              <span>{showFactDetails ? "▴ Hide Source Evidence" : "▾ Source Document Attribution"}</span>
              {message.provenance && (
                <span className="text-slate-400 font-mono">({message.provenance})</span>
              )}
            </button>
            {showFactDetails && (
              <div className="mt-2 space-y-1.5 rounded border border-slate-800 bg-slate-950/70 p-2.5 font-mono text-[11px] text-slate-300">
                <div className="flex gap-2">
                  <span className="text-slate-500 w-28">Source Document:</span>
                  <span className="text-amber-300 font-semibold">{message.provenance || message.details?.source_display_name || "Knowledge"}</span>
                </div>
                {message.details?.chunk_id && (
                  <div className="flex gap-2">
                    <span className="text-slate-500 w-28">Chunk ID:</span>
                    <span className="text-slate-200">{message.details.chunk_id}</span>
                  </div>
                )}
                {message.details?.start_char !== undefined && message.details?.end_char !== undefined && (
                  <div className="flex gap-2">
                    <span className="text-slate-500 w-28">Offset Span:</span>
                    <span className="text-slate-200">[{message.details.start_char} : {message.details.end_char}]</span>
                  </div>
                )}
                {message.details?.score !== undefined && (
                  <div className="flex gap-2">
                    <span className="text-slate-500 w-28">BM25 Score:</span>
                    <span className="text-slate-200 font-semibold">{message.details.score}</span>
                  </div>
                )}
                <div className="flex gap-2">
                  <span className="text-slate-500 w-28">Execution:</span>
                  <span className="text-slate-300">
                    {isDocumentExcerpt ? "Raw Document Passage (0 Model Calls)" : "Grounded Prompt Conditioning"}
                  </span>
                </div>
                {message.details?.conflict_detected && (
                  <div className="mt-2 rounded border border-rose-900/80 bg-rose-950/60 p-2 text-[11px] text-rose-300">
                    <span className="font-bold">⚠️ Contradiction Warning: </span>
                    {message.details?.conflict_details?.message || "Contradictory information detected across sources."}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Error message card */}
        {message.error && (
          <div className="mt-2.5 rounded border border-rose-900/60 bg-rose-950/40 p-2 text-xs text-rose-300">
            <span className="font-semibold">Error: </span>
            {message.error}
          </div>
        )}

        {/* Real Backend Telemetry Metrics */}
        {!isUser && message.metrics && (
          <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-slate-800/80 pt-2 font-mono text-[11px] text-slate-400">
            {isNeural && message.metrics.generatedTokens !== undefined && (
              <span>
                Tokens:{" "}
                <span className="text-slate-200 font-semibold">
                  {message.metrics.generatedTokens}
                </span>
              </span>
            )}
            {isNeural && message.metrics.tokensPerSecond !== undefined && (
              <span>
                Speed:{" "}
                <span className="text-teal-400 font-semibold">
                  {message.metrics.tokensPerSecond.toFixed(1)} tok/s
                </span>
              </span>
            )}
            {message.metrics.elapsedTimeSec !== undefined && (
              <span>
                Time:{" "}
                <span className="text-slate-200">
                  {message.metrics.elapsedTimeSec.toFixed(2)}s
                </span>
              </span>
            )}
            {!isNeural && (
              <span className="text-slate-400">
                Origin:{" "}
                <span className="text-slate-200 font-semibold">
                  {message.toolName || origin || "Tool"} (0 model calls)
                </span>
              </span>
            )}
            {isNeural &&
              message.metrics.activeContextTokens !== undefined &&
              message.metrics.contextLimit !== undefined && (
                <span className="flex items-center gap-1">
                  Context:{" "}
                  <span className="text-slate-200">
                    {message.metrics.activeContextTokens}/
                    {message.metrics.contextLimit}
                  </span>
                  {message.metrics.contextCropped && (
                    <span className="rounded bg-amber-950 px-1 py-0.2 text-[10px] font-semibold text-amber-400 border border-amber-800">
                      Cropped
                    </span>
                  )}
                </span>
              )}
          </div>
        )}
      </div>

      {/* Regenerate Action Button (for neural generations) */}
      {!isUser && isLastAssistant && !isGenerating && onRegenerate && isNeural && (
        <button
          type="button"
          onClick={onRegenerate}
          className="mt-1 flex items-center gap-1 px-1 text-[11px] text-slate-400 transition hover:text-teal-400"
          title="Regenerate this response with current settings"
        >
          <span>↺ Regenerate response</span>
        </button>
      )}
    </div>
  );
}
