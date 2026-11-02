"use client";

import React from "react";

export function EmptyState() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-4 text-center select-none">
      {/* ChatGPT-style clean circular speech bubble icon */}
      <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-full border border-zinc-700/80 bg-zinc-800/40 text-zinc-300 shadow-sm">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="h-7 w-7"
        >
          <path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z" />
        </svg>
      </div>

      <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-zinc-100">
        What can I help with?
      </h1>
      <p className="mt-2 text-sm text-zinc-400 font-normal">
        Your own locally trained AI
      </p>
    </div>
  );
}
