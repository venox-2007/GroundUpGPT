"use client";

import React, { useRef, useEffect } from "react";
import { AttachmentItem } from "../types";

interface ComposerProps {
  input: string;
  onChange: (val: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  isGenerating: boolean;
  disabled: boolean;
  attachments: AttachmentItem[];
  onAddFiles: (files: FileList) => void;
  onRemoveAttachment: (id: string) => void;
}

export function Composer({
  input,
  onChange,
  onSubmit,
  onStop,
  isGenerating,
  disabled,
  attachments,
  onAddFiles,
  onRemoveAttachment,
}: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-resize textarea height
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [input]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isGenerating && (input.trim() || attachments.length > 0)) {
        onSubmit();
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onAddFiles(e.target.files);
      // Reset input value so same file can be re-selected if removed
      e.target.value = "";
    }
  };

  const hasContent = input.trim().length > 0 || attachments.length > 0;

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="mx-auto w-full max-w-3xl px-3 sm:px-4 pb-3">
      {/* Hidden File Picker */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".txt,.md,.markdown,.pdf,.docx"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Main Composer Box */}
      <div className="relative flex flex-col rounded-[26px] border border-zinc-700/70 bg-[#2f2f32]/90 shadow-lg transition-all focus-within:border-zinc-500 focus-within:bg-[#2f2f32]">
        {/* Attachment Chips Tray */}
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 px-3 pt-3">
            {attachments.map((att) => (
              <div
                key={att.id}
                className={`group flex items-center gap-2 rounded-xl border px-2.5 py-1.5 text-xs transition ${
                  att.status === "error"
                    ? "border-rose-800 bg-rose-950/50 text-rose-200"
                    : "border-zinc-700 bg-zinc-800/80 text-zinc-200"
                }`}
              >
                {/* File Icon */}
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-3.5 w-3.5 text-zinc-400"
                >
                  <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                  <polyline points="14 2 14 8 20 8" />
                </svg>

                <div className="flex flex-col">
                  <span className="max-w-[140px] truncate font-medium sm:max-w-[200px]" title={att.name}>
                    {att.name}
                  </span>
                  <span className="text-[10px] text-zinc-400">
                    {att.status === "uploading"
                      ? "Processing..."
                      : att.status === "error"
                      ? (att.error || "Failed")
                      : formatFileSize(att.size)}
                  </span>
                </div>

                {/* Remove Attachment Button */}
                <button
                  type="button"
                  onClick={() => onRemoveAttachment(att.id)}
                  aria-label={`Remove ${att.name}`}
                  className="ml-1 rounded-full p-0.5 text-zinc-400 hover:bg-zinc-700 hover:text-white transition"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="h-3.5 w-3.5"
                  >
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Input Text Row */}
        <div className="flex items-end gap-2 px-3 py-2">
          {/* Plus / Attachment Button */}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled || isGenerating}
            title="Attach file (TXT, MD, PDF, DOCX)"
            aria-label="Attach file"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-zinc-300 hover:bg-zinc-700/60 hover:text-white transition disabled:opacity-40"
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
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
          </button>

          {/* Multiline auto-expanding textarea */}
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              disabled
                ? "Connecting to GroundUpGPT..."
                : isGenerating
                ? "Generating response..."
                : "Message GroundUpGPT..."
            }
            disabled={disabled || isGenerating}
            rows={1}
            className="max-h-[180px] min-h-[36px] flex-1 resize-none bg-transparent py-2 px-1 text-sm text-zinc-100 placeholder-zinc-400 focus:outline-none disabled:opacity-50"
          />

          {/* Send or Stop Button */}
          {isGenerating ? (
            <button
              type="button"
              onClick={onStop}
              title="Stop generation"
              aria-label="Stop generation"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-zinc-200 text-zinc-900 hover:bg-white shadow transition"
            >
              <span className="h-3 w-3 rounded-xs bg-zinc-900" />
            </button>
          ) : (
            <button
              type="button"
              onClick={onSubmit}
              disabled={disabled || !hasContent}
              title="Send message"
              aria-label="Send message"
              className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full transition shadow-sm ${
                hasContent && !disabled
                  ? "bg-white text-zinc-950 hover:bg-zinc-200 cursor-pointer"
                  : "bg-zinc-700/60 text-zinc-500 cursor-not-allowed opacity-60"
              }`}
            >
              {/* Up-arrow icon */}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.4"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-4 w-4"
              >
                <line x1="12" y1="19" x2="12" y2="5" />
                <polyline points="5 12 12 5 19 12" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Subtle ChatGPT-style disclaimer */}
      <div className="mt-2 text-center text-[11px] text-zinc-500 select-none">
        GroundUpGPT can make mistakes. Check important info.
      </div>
    </div>
  );
}
