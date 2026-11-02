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
  const [copied, setCopied] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);

  // Copy message text
  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Determine truthful origin badge
  const origin = message.origin;
  const isCalculator = origin === "safe_calculator";
  const isLocalFact = origin === "local_fact";
  const isToolError = origin === "tool_error";
  const isDocument = origin === "document_excerpt";

  const getAttributionPill = () => {
    if (isUser) return null;
    if (isCalculator) {
      return {
        label: "Calculated",
        className: "bg-sky-950/60 text-sky-300 border-sky-800/80",
      };
    }
    if (isLocalFact) {
      return {
        label: "Curated Fact",
        className: "bg-emerald-950/60 text-emerald-300 border-emerald-800/80",
      };
    }
    if (isDocument) {
      return {
        label: message.provenance ? `From ${message.provenance}` : "From your file",
        className: "bg-amber-950/60 text-amber-300 border-amber-800/80",
      };
    }
    if (isToolError) {
      return {
        label: "Tool Error",
        className: "bg-rose-950/60 text-rose-300 border-rose-800/80",
      };
    }
    // Neural model response
    const profile = message.modelProfile ? message.modelProfile : "GroundUpGPT";
    return {
      label: profile,
      className: "bg-zinc-800/80 text-zinc-300 border-zinc-700/60",
    };
  };

  const badge = getAttributionPill();

  return (
    <div className={`group flex w-full flex-col py-3 ${isUser ? "items-end" : "items-start"}`}>
      {/* Turn Header / Attribution */}
      {!isUser && badge && (
        <div className="mb-1.5 flex items-center gap-2 px-1 text-xs">
          <span
            className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium tracking-wide ${badge.className}`}
          >
            {badge.label}
          </span>
          {isCalculator && message.evaluatedExpression && (
            <span className="font-mono text-[11px] text-zinc-400">
              {message.evaluatedExpression}
            </span>
          )}
        </div>
      )}

      {/* Message Content Container */}
      <div
        className={`relative max-w-[92%] sm:max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? "bg-[#2f2f32] text-zinc-100 rounded-br-sm shadow-xs"
            : "text-zinc-200"
        }`}
      >
        {/* If user attached files, display attachment tags in user bubble */}
        {isUser && message.attachments && message.attachments.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-1.5 border-b border-zinc-600/40 pb-2">
            {message.attachments.map((att) => (
              <span
                key={att.id}
                className="inline-flex items-center gap-1 rounded-md bg-zinc-800/90 px-2 py-0.5 text-[11px] text-zinc-300 font-medium"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="h-3 w-3 text-zinc-400"
                >
                  <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
                  <polyline points="14 2 14 8 20 8" />
                </svg>
                <span className="truncate max-w-[150px]">{att.name}</span>
              </span>
            ))}
          </div>
        )}

        {/* Content body rendered with Markdown support */}
        <div className="markdown-content">
          <RenderMarkdown content={message.content} />
          {message.isStreaming && (
            <span className="inline-block animate-pulse text-zinc-400 ml-1">▍</span>
          )}
        </div>

        {/* Details drawer for Local Facts or Documents (Expandable) */}
        {!isUser && (isLocalFact || isDocument) && message.details && (
          <div className="mt-2.5 border-t border-zinc-800/80 pt-2 text-xs">
            <button
              type="button"
              onClick={() => setDetailsOpen(!detailsOpen)}
              className="text-[11px] font-mono text-zinc-400 hover:text-zinc-200 transition"
            >
              {detailsOpen ? "▴ Hide attribution source" : "▾ View attribution source"}
            </button>
            {detailsOpen && (
              <div className="mt-1.5 rounded-lg border border-zinc-800 bg-zinc-900/80 p-2.5 font-mono text-[11px] text-zinc-300 space-y-1">
                {message.factId && <div><span className="text-zinc-500">Fact ID:</span> {message.factId}</div>}
                {message.provenance && <div><span className="text-zinc-500">Source:</span> {message.provenance}</div>}
                {message.details.chunk_id && <div><span className="text-zinc-500">Chunk:</span> {message.details.chunk_id}</div>}
                {message.details.score !== undefined && <div><span className="text-zinc-500">Relevance Score:</span> {message.details.score}</div>}
              </div>
            )}
          </div>
        )}

        {/* Action icons row (Copy, Regenerate) */}
        {!isUser && !message.isStreaming && (
          <div className="mt-2 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              type="button"
              onClick={handleCopy}
              title="Copy response"
              className="flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 transition"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-3 w-3"
              >
                <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
                <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
              </svg>
              <span>{copied ? "Copied" : "Copy"}</span>
            </button>

            {isLastAssistant && !isGenerating && onRegenerate && (
              <button
                type="button"
                onClick={onRegenerate}
                title="Regenerate response"
                className="flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 transition"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="h-3 w-3"
                >
                  <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                  <path d="M3 3v5h5" />
                  <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
                  <path d="M16 21h5v-5" />
                </svg>
                <span>Regenerate</span>
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Lightweight, robust markdown parser rendering headings, lists, code blocks, bold, italics, tables.
 */
function RenderMarkdown({ content }: { content: string }) {
  if (!content) return null;

  // Split into code blocks and normal text blocks
  const parts: React.ReactNode[] = [];
  const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      parts.push(
        <TextSection key={`text-${lastIndex}`} raw={content.slice(lastIndex, match.index)} />
      );
    }
    const lang = match[1] || "code";
    const code = match[2];
    parts.push(<CodeBlock key={`code-${match.index}`} language={lang} code={code} />);
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < content.length) {
    parts.push(
      <TextSection key={`text-${lastIndex}`} raw={content.slice(lastIndex)} />
    );
  }

  return <div className="space-y-2">{parts}</div>;
}

function CodeBlock({ language, code }: { language: string; code: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopyCode = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-2.5 overflow-hidden rounded-xl border border-zinc-700/80 bg-zinc-950 font-mono text-xs">
      <div className="flex items-center justify-between border-b border-zinc-800 bg-zinc-900/90 px-3 py-1.5 text-zinc-400">
        <span className="text-[11px] font-sans lowercase">{language || "code"}</span>
        <button
          type="button"
          onClick={handleCopyCode}
          className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-white transition"
        >
          <span>{copied ? "Copied!" : "Copy code"}</span>
        </button>
      </div>
      <pre className="overflow-x-auto p-3 text-zinc-200 leading-normal">
        <code>{code.trimEnd()}</code>
      </pre>
    </div>
  );
}

function TextSection({ raw }: { raw: string }) {
  const lines = raw.split("\n");
  const elements: React.ReactNode[] = [];
  let currentList: React.ReactNode[] = [];
  let isNumbered = false;

  const flushList = () => {
    if (currentList.length > 0) {
      if (isNumbered) {
        elements.push(
          <ol key={`ol-${elements.length}`} className="list-decimal pl-5 space-y-1 my-1.5">
            {currentList}
          </ol>
        );
      } else {
        elements.push(
          <ul key={`ul-${elements.length}`} className="list-disc pl-5 space-y-1 my-1.5">
            {currentList}
          </ul>
        );
      }
      currentList = [];
    }
  };

  lines.forEach((line, idx) => {
    const trimmed = line.trim();

    // Headers
    if (trimmed.startsWith("### ")) {
      flushList();
      elements.push(
        <h3 key={`h3-${idx}`} className="text-sm font-semibold text-white mt-2 mb-1">
          {formatInline(trimmed.slice(4))}
        </h3>
      );
      return;
    }
    if (trimmed.startsWith("## ")) {
      flushList();
      elements.push(
        <h2 key={`h2-${idx}`} className="text-base font-semibold text-white mt-2 mb-1">
          {formatInline(trimmed.slice(3))}
        </h2>
      );
      return;
    }
    if (trimmed.startsWith("# ")) {
      flushList();
      elements.push(
        <h1 key={`h1-${idx}`} className="text-lg font-bold text-white mt-2 mb-1">
          {formatInline(trimmed.slice(2))}
        </h1>
      );
      return;
    }

    // Bullet lists
    if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      if (isNumbered && currentList.length > 0) flushList();
      isNumbered = false;
      currentList.push(
        <li key={`li-${idx}`}>{formatInline(trimmed.slice(2))}</li>
      );
      return;
    }

    // Numbered lists
    const numMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
    if (numMatch) {
      if (!isNumbered && currentList.length > 0) flushList();
      isNumbered = true;
      currentList.push(
        <li key={`li-${idx}`}>{formatInline(numMatch[2])}</li>
      );
      return;
    }

    flushList();

    // Empty line creates break
    if (!trimmed) {
      elements.push(<div key={`br-${idx}`} className="h-1.5" />);
      return;
    }

    // Normal paragraph
    elements.push(
      <p key={`p-${idx}`} className="leading-relaxed">
        {formatInline(line)}
      </p>
    );
  });

  flushList();
  return <div className="space-y-1">{elements}</div>;
}

/**
 * Format inline bold, italic, and inline code.
 */
function formatInline(text: string): React.ReactNode {
  // Split inline code `...`
  const codeParts = text.split(/(`[^`]+`)/);
  return codeParts.map((part, i) => {
    if (part.startsWith("`") && part.endsWith("`") && part.length > 1) {
      return (
        <code
          key={i}
          className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[11px] text-zinc-200 border border-zinc-700/60"
        >
          {part.slice(1, -1)}
        </code>
      );
    }

    // Process bold **...**
    const boldParts = part.split(/(\*\*[^*]+\*\*)/);
    return (
      <span key={i}>
        {boldParts.map((bPart, j) => {
          if (bPart.startsWith("**") && bPart.endsWith("**") && bPart.length > 3) {
            return (
              <strong key={j} className="font-semibold text-white">
                {bPart.slice(2, -2)}
              </strong>
            );
          }
          return bPart;
        })}
      </span>
    );
  });
}
