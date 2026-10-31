"use client";

import React from "react";
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

  return (
    <div
      className={`group flex w-full flex-col gap-1.5 py-4 ${
        isUser ? "items-end" : "items-start"
      }`}
    >
      {/* Role Identifier Header */}
      <div className="flex items-center gap-2 px-1 text-[11px] font-medium text-slate-400">
        <span
          className={`flex h-4 w-4 items-center justify-center rounded text-[10px] font-bold ${
            isUser ? "bg-slate-700 text-slate-200" : "bg-teal-600 text-white"
          }`}
        >
          {isUser ? "U" : "G"}
        </span>
        <span>{isUser ? "You (Prompt)" : "GroundUpGPT (Completion)"}</span>
      </div>

      {/* Message Content Bubble */}
      <div
        className={`max-w-[85%] rounded-lg px-4 py-3 text-sm leading-relaxed sm:max-w-[75%] ${
          isUser
            ? "border border-slate-700 bg-slate-800 text-slate-100"
            : "border border-slate-800 bg-slate-900/90 text-slate-200"
        }`}
      >
        <div className="whitespace-pre-wrap font-sans">
          {message.content}
          {message.isStreaming && (
            <span className="inline-block animate-pulse text-teal-400">▍</span>
          )}
        </div>

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
            {message.metrics.generatedTokens !== undefined && (
              <span>
                Tokens:{" "}
                <span className="text-slate-200 font-semibold">
                  {message.metrics.generatedTokens}
                </span>
              </span>
            )}
            {message.metrics.tokensPerSecond !== undefined && (
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
            {message.metrics.activeContextTokens !== undefined &&
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

      {/* Regenerate Action Button */}
      {!isUser && isLastAssistant && !isGenerating && onRegenerate && (
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
