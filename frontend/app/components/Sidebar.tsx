"use client";

import React, { useState } from "react";
import { ConversationSummary } from "../types";

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  onRenameConversation: (id: string, newTitle: string) => Promise<void>;
  onDeleteConversation: (id: string) => Promise<void>;
  onOpenSettings: () => void;
}

export function Sidebar({
  isOpen,
  onToggle,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewChat,
  onRenameConversation,
  onDeleteConversation,
  onOpenSettings,
}: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);

  const filteredConversations = conversations.filter((c) =>
    c.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleStartRename = (c: ConversationSummary, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(c.id);
    setEditingTitle(c.title);
    setMenuOpenId(null);
  };

  const handleSaveRename = async (id: string) => {
    if (editingTitle.trim()) {
      await onRenameConversation(id, editingTitle.trim());
    }
    setEditingId(null);
  };

  const handleKeyDownRename = (e: React.KeyboardEvent, id: string) => {
    if (e.key === "Enter") {
      handleSaveRename(id);
    } else if (e.key === "Escape") {
      setEditingId(null);
    }
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          onClick={onToggle}
          className="fixed inset-0 z-40 bg-black/60 md:hidden backdrop-blur-xs transition-opacity"
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex h-full w-[260px] flex-col border-r border-zinc-800/80 bg-[#18181b] p-3 transition-transform duration-200 ease-in-out md:static md:translate-x-0 ${
          isOpen ? "translate-x-0" : "-translate-x-full md:hidden"
        }`}
      >
        {/* Top Header: Brand & Collapse Toggle */}
        <div className="flex items-center justify-between pb-3 px-1">
          <div className="flex items-center gap-2.5">
            {/* Sidebar toggle icon (matches mockup) */}
            <button
              type="button"
              onClick={onToggle}
              title="Close sidebar"
              aria-label="Toggle sidebar"
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
            <span className="font-semibold text-base text-zinc-100 tracking-tight">
              GroundUpGPT
            </span>
          </div>
        </div>

        {/* New Chat Button */}
        <button
          type="button"
          onClick={onNewChat}
          className="flex w-full items-center gap-2.5 rounded-lg border border-zinc-700/60 bg-zinc-800/40 px-3 py-2 text-sm font-medium text-zinc-200 shadow-xs hover:bg-zinc-800 hover:text-white transition"
        >
          {/* Edit / Compose Icon */}
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-4 w-4 text-zinc-300"
          >
            <path d="M12 20h9" />
            <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
          </svg>
          <span>New chat</span>
        </button>

        {/* Search Chats Input */}
        <div className="relative mt-2.5">
          <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-2.5 text-zinc-400">
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
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.3-4.3" />
            </svg>
          </div>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search chats"
            className="w-full rounded-lg bg-zinc-800/30 py-1.5 pl-8 pr-3 text-xs text-zinc-200 placeholder-zinc-500 focus:bg-zinc-800/80 focus:outline-none focus:ring-1 focus:ring-zinc-600 transition"
          />
        </div>

        {/* Recent Conversations List */}
        <div className="mt-3 flex-1 overflow-y-auto pr-1">
          <div className="px-1.5 py-1 text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
            Recent
          </div>

          <div className="mt-1 space-y-0.5">
            {filteredConversations.length === 0 ? (
              <div className="px-2 py-3 text-xs text-zinc-500 italic">
                {searchQuery ? "No matching chats" : "No recent chats"}
              </div>
            ) : (
              filteredConversations.map((c) => {
                const isActive = c.id === activeConversationId;
                const isEditing = c.id === editingId;
                const isMenuOpen = c.id === menuOpenId;

                return (
                  <div
                    key={c.id}
                    className="group relative flex items-center rounded-lg"
                  >
                    {isEditing ? (
                      <input
                        type="text"
                        autoFocus
                        value={editingTitle}
                        onChange={(e) => setEditingTitle(e.target.value)}
                        onBlur={() => handleSaveRename(c.id)}
                        onKeyDown={(e) => handleKeyDownRename(e, c.id)}
                        className="w-full rounded-md bg-zinc-800 px-2.5 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-zinc-500"
                      />
                    ) : (
                      <>
                        <button
                          type="button"
                          onClick={() => onSelectConversation(c.id)}
                          className={`flex w-full items-center truncate rounded-lg px-2.5 py-2 text-left text-xs transition ${
                            isActive
                              ? "bg-zinc-800 text-white font-medium"
                              : "text-zinc-300 hover:bg-zinc-800/60 hover:text-white"
                          }`}
                          title={c.title}
                        >
                          <span className="truncate">{c.title || "Untitled chat"}</span>
                        </button>

                        {/* Three-dots menu trigger */}
                        <div className="absolute right-1">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setMenuOpenId(isMenuOpen ? null : c.id);
                            }}
                            className={`rounded p-1 text-zinc-400 hover:bg-zinc-700 hover:text-white transition ${
                              isMenuOpen || isActive ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                            }`}
                            title="Chat options"
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
                              <circle cx="12" cy="12" r="1" />
                              <circle cx="19" cy="12" r="1" />
                              <circle cx="5" cy="12" r="1" />
                            </svg>
                          </button>

                          {/* Context Menu Dropdown */}
                          {isMenuOpen && (
                            <div
                              onClick={(e) => e.stopPropagation()}
                              className="absolute right-0 top-6 z-50 w-28 rounded-lg border border-zinc-700 bg-zinc-800 py-1 shadow-xl text-xs text-zinc-200"
                            >
                              <button
                                type="button"
                                onClick={(e) => handleStartRename(c, e)}
                                className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left hover:bg-zinc-700 hover:text-white"
                              >
                                <svg
                                  xmlns="http://www.w3.org/2000/svg"
                                  viewBox="0 0 24 24"
                                  fill="none"
                                  stroke="currentColor"
                                  strokeWidth="2"
                                  className="h-3 w-3"
                                >
                                  <path d="M12 20h9" />
                                  <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
                                </svg>
                                <span>Rename</span>
                              </button>
                              <button
                                type="button"
                                onClick={async (e) => {
                                  e.stopPropagation();
                                  setMenuOpenId(null);
                                  await onDeleteConversation(c.id);
                                }}
                                className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-rose-400 hover:bg-rose-950/60 hover:text-rose-200"
                              >
                                <svg
                                  xmlns="http://www.w3.org/2000/svg"
                                  viewBox="0 0 24 24"
                                  fill="none"
                                  stroke="currentColor"
                                  strokeWidth="2"
                                  className="h-3 w-3"
                                >
                                  <path d="M3 6h18" />
                                  <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                                  <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                                </svg>
                                <span>Delete</span>
                              </button>
                            </div>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Bottom Item: Settings */}
        <div className="border-t border-zinc-800/80 pt-2">
          <button
            type="button"
            onClick={onOpenSettings}
            className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-xs font-medium text-zinc-300 hover:bg-zinc-800/80 hover:text-white transition"
          >
            {/* Gear Icon (matches mockup) */}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-4 w-4 text-zinc-400"
            >
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
            <span>Settings</span>
          </button>
        </div>
      </aside>
    </>
  );
}
