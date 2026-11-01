"use client";

import React, { useRef, useEffect } from "react";

interface ChatInputProps {
  input: string;
  onChange: (val: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  isGenerating: boolean;
  disabled: boolean;
  isSft?: boolean;
}

export function ChatInput({
  input,
  onChange,
  onSubmit,
  onStop,
  isGenerating,
  disabled,
  isSft,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea height up to a max
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [input]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isGenerating && input.trim()) {
        onSubmit();
      }
    }
  };

  return (
    <div className="mx-auto w-full max-w-3xl px-4 pb-4 pt-2">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!isGenerating && input.trim()) {
            onSubmit();
          }
        }}
        className="relative flex flex-col rounded-xl border border-slate-700 bg-slate-900 shadow-lg focus-within:border-teal-500"
      >
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            disabled
              ? "Connecting to GroundUpGPT backend..."
              : isGenerating
              ? "Model is generating tokens..."
              : isSft
              ? "Ask GroundUpGPT anything (experimental instruction-tuned model)..."
              : "Type a prompt to complete... (e.g., 'The ')"
          }
          disabled={disabled || isGenerating}
          rows={1}
          className="w-full resize-none bg-transparent px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none disabled:opacity-50"
        />

        <div className="flex items-center justify-between border-t border-slate-800/80 px-3 py-2">
          <span className="text-[11px] text-slate-500">
            Press <kbd className="rounded border border-slate-700 bg-slate-800 px-1 py-0.5 text-[10px] text-slate-300">Enter</kbd> to send, <kbd className="rounded border border-slate-700 bg-slate-800 px-1 py-0.5 text-[10px] text-slate-300">Shift+Enter</kbd> for newline
          </span>

          <div className="flex items-center gap-2">
            {isGenerating ? (
              <button
                type="button"
                onClick={onStop}
                className="flex items-center gap-1.5 rounded-md bg-rose-600 px-3 py-1.5 text-xs font-semibold text-white shadow transition hover:bg-rose-500"
              >
                <span className="h-2 w-2 rounded-sm bg-white" />
                Stop
              </button>
            ) : (
              <button
                type="submit"
                disabled={disabled || !input.trim()}
                className="flex items-center gap-1.5 rounded-md bg-teal-600 px-3.5 py-1.5 text-xs font-semibold text-white shadow transition hover:bg-teal-500 disabled:opacity-40"
              >
                Send
                <span aria-hidden="true">→</span>
              </button>
            )}
          </div>
        </div>
      </form>
    </div>
  );
}
