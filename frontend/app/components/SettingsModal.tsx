"use client";

import React, { useState } from "react";
import { GenerationSettings, ModelInfo, ModelProfilesResponse } from "../types";

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  theme: "dark" | "light";
  onToggleTheme: (t: "dark" | "light") => void;
  saveHistory: boolean;
  onToggleSaveHistory: (val: boolean) => void;
  modelInfo: ModelInfo | null;
  profiles: ModelProfilesResponse | null;
  activeProfile: string | null;
  onSwitchProfile: (profileKey: string) => Promise<void>;
  settings: GenerationSettings;
  onUpdateSettings: (newSettings: GenerationSettings) => void;
  isSwitchingModel: boolean;
}

export function SettingsModal({
  isOpen,
  onClose,
  theme,
  onToggleTheme,
  saveHistory,
  onToggleSaveHistory,
  modelInfo,
  profiles,
  activeProfile,
  onSwitchProfile,
  settings,
  onUpdateSettings,
  isSwitchingModel,
}: SettingsModalProps) {
  const [activeTab, setActiveTab] = useState<"general" | "model" | "advanced">("general");
  const [advancedExpanded, setAdvancedExpanded] = useState(false);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        onClick={onClose}
        className="fixed inset-0 bg-black/65 backdrop-blur-xs transition-opacity"
      />

      {/* Dialog Window */}
      <div className="relative w-full max-w-lg rounded-2xl border border-zinc-700/80 bg-[#1e1e22] shadow-2xl text-zinc-200 overflow-hidden z-10 flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800/80 px-5 py-4">
          <h2 className="text-base font-semibold text-white tracking-tight">Settings</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close settings"
            className="rounded-lg p-1 text-zinc-400 hover:bg-zinc-800 hover:text-white transition"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-4 w-4"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-zinc-800/80 px-5">
          <button
            type="button"
            onClick={() => setActiveTab("general")}
            className={`border-b-2 py-2.5 px-3 text-xs font-medium transition ${
              activeTab === "general"
                ? "border-white text-white"
                : "border-transparent text-zinc-400 hover:text-zinc-200"
            }`}
          >
            General
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("model")}
            className={`border-b-2 py-2.5 px-3 text-xs font-medium transition ${
              activeTab === "model"
                ? "border-white text-white"
                : "border-transparent text-zinc-400 hover:text-zinc-200"
            }`}
          >
            Model
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("advanced")}
            className={`border-b-2 py-2.5 px-3 text-xs font-medium transition ${
              activeTab === "advanced"
                ? "border-white text-white"
                : "border-transparent text-zinc-400 hover:text-zinc-200"
            }`}
          >
            Advanced
          </button>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5 text-sm">
          {/* GENERAL TAB */}
          {activeTab === "general" && (
            <div className="space-y-4">
              {/* Theme selector */}
              <div className="flex items-center justify-between py-1">
                <div>
                  <div className="font-medium text-zinc-100">Theme</div>
                  <div className="text-xs text-zinc-400">Choose appearance style</div>
                </div>
                <div className="flex rounded-lg border border-zinc-700 bg-zinc-800 p-0.5 text-xs">
                  <button
                    type="button"
                    onClick={() => onToggleTheme("dark")}
                    className={`rounded-md px-3 py-1 font-medium transition ${
                      theme === "dark"
                        ? "bg-zinc-700 text-white shadow-xs"
                        : "text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    Dark
                  </button>
                  <button
                    type="button"
                    onClick={() => onToggleTheme("light")}
                    className={`rounded-md px-3 py-1 font-medium transition ${
                      theme === "light"
                        ? "bg-zinc-700 text-white shadow-xs"
                        : "text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    Light
                  </button>
                </div>
              </div>

              {/* Local chat history opt-in toggle */}
              <div className="border-t border-zinc-800 pt-4 flex items-start justify-between gap-4">
                <div>
                  <div className="font-medium text-zinc-100">Save chats locally</div>
                  <div className="text-xs text-zinc-400 mt-0.5 leading-relaxed">
                    When turned on, your conversations are saved to local SQLite history so you can review them later.
                    When turned off, conversations stay in temporary memory only and are never saved to disk.
                  </div>
                </div>
                <button
                  type="button"
                  role="switch"
                  aria-checked={saveHistory}
                  onClick={() => onToggleSaveHistory(!saveHistory)}
                  className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                    saveHistory ? "bg-emerald-600" : "bg-zinc-700"
                  }`}
                >
                  <span
                    className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow-lg ring-0 transition duration-200 ease-in-out ${
                      saveHistory ? "translate-x-5" : "translate-x-0"
                    }`}
                  />
                </button>
              </div>
            </div>
          )}

          {/* MODEL TAB */}
          {activeTab === "model" && (
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-zinc-400 mb-1.5">
                  Active Model Checkpoint
                </label>
                <div className="rounded-lg border border-zinc-700/80 bg-zinc-800/60 p-3 text-xs">
                  <div className="font-semibold text-white">
                    {modelInfo?.model_name || "GroundUpGPT"}
                  </div>
                  <div className="mt-1 text-zinc-400 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[11px]">
                    <span>Params: {(modelInfo?.total_parameters || 0).toLocaleString()}</span>
                    <span>Layers: {modelInfo?.num_layers || 0}</span>
                    <span>Context: {modelInfo?.block_size || 256} tokens</span>
                    <span>Device: {modelInfo?.device || "cpu"}</span>
                  </div>
                </div>
              </div>

              {/* Switch Model Profile */}
              {profiles && (
                <div className="border-t border-zinc-800 pt-4">
                  <label className="block text-xs font-medium text-zinc-300 mb-1.5">
                    Switch Experimental Profile
                  </label>
                  <select
                    value={activeProfile || ""}
                    disabled={isSwitchingModel}
                    onChange={(e) => onSwitchProfile(e.target.value)}
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-800/90 px-3 py-2 text-xs text-white focus:outline-none focus:ring-1 focus:ring-zinc-500 disabled:opacity-50"
                  >
                    {Object.entries(profiles.profiles).map(([key, prof]) => (
                      <option key={key} value={key} disabled={!prof.checkpoint_exists}>
                        {prof.name} {!prof.checkpoint_exists ? "(Missing Checkpoint)" : ""}
                      </option>
                    ))}
                  </select>
                  {isSwitchingModel && (
                    <div className="mt-2 text-xs text-teal-400">Loading checkpoint...</div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* ADVANCED TAB */}
          {activeTab === "advanced" && (
            <div className="space-y-4">
              {/* Native GPT Only Toggle */}
              <div className="flex items-start justify-between gap-4 py-1">
                <div>
                  <div className="font-medium text-zinc-100">Native GPT only</div>
                  <div className="text-xs text-zinc-400 mt-0.5 leading-relaxed">
                    Bypasses automatic tools (SafeCalculator, Local Facts) and forces all requests strictly through the neural model. Useful for raw model evaluation.
                  </div>
                </div>
                <button
                  type="button"
                  role="switch"
                  aria-checked={Boolean(settings.nativeOnly)}
                  onClick={() =>
                    onUpdateSettings({ ...settings, nativeOnly: !settings.nativeOnly })
                  }
                  className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                    settings.nativeOnly ? "bg-amber-600" : "bg-zinc-700"
                  }`}
                >
                  <span
                    className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow-lg ring-0 transition duration-200 ease-in-out ${
                      settings.nativeOnly ? "translate-x-5" : "translate-x-0"
                    }`}
                  />
                </button>
              </div>

              {/* Collapsed sampling controls */}
              <div className="border-t border-zinc-800 pt-3">
                <button
                  type="button"
                  onClick={() => setAdvancedExpanded(!advancedExpanded)}
                  className="flex w-full items-center justify-between text-xs font-medium text-zinc-400 hover:text-white transition"
                >
                  <span>Sampling Parameters</span>
                  <span>{advancedExpanded ? "▴ Hide" : "▾ Expand"}</span>
                </button>

                {advancedExpanded && (
                  <div className="mt-3 space-y-3 rounded-lg border border-zinc-800 bg-zinc-900/60 p-3 text-xs">
                    {/* Temperature */}
                    <div>
                      <div className="flex justify-between text-[11px] text-zinc-400">
                        <span>Temperature</span>
                        <span className="font-mono text-zinc-200">{settings.temperature.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0.1"
                        max="1.5"
                        step="0.05"
                        value={settings.temperature}
                        onChange={(e) =>
                          onUpdateSettings({ ...settings, temperature: parseFloat(e.target.value) })
                        }
                        className="w-full mt-1 accent-zinc-400"
                      />
                    </div>

                    {/* Top-k */}
                    <div>
                      <div className="flex justify-between text-[11px] text-zinc-400">
                        <span>Top-k</span>
                        <span className="font-mono text-zinc-200">{settings.top_k || "Disabled"}</span>
                      </div>
                      <input
                        type="range"
                        min="1"
                        max="80"
                        step="1"
                        value={settings.top_k || 40}
                        onChange={(e) =>
                          onUpdateSettings({ ...settings, top_k: parseInt(e.target.value) })
                        }
                        className="w-full mt-1 accent-zinc-400"
                      />
                    </div>

                    {/* Top-p */}
                    <div>
                      <div className="flex justify-between text-[11px] text-zinc-400">
                        <span>Top-p (Nucleus)</span>
                        <span className="font-mono text-zinc-200">
                          {settings.top_p ? settings.top_p.toFixed(2) : "Disabled"}
                        </span>
                      </div>
                      <input
                        type="range"
                        min="0.1"
                        max="1.0"
                        step="0.05"
                        value={settings.top_p || 0.9}
                        onChange={(e) =>
                          onUpdateSettings({ ...settings, top_p: parseFloat(e.target.value) })
                        }
                        className="w-full mt-1 accent-zinc-400"
                      />
                    </div>

                    {/* Repetition Penalty */}
                    <div>
                      <div className="flex justify-between text-[11px] text-zinc-400">
                        <span>Repetition Penalty</span>
                        <span className="font-mono text-zinc-200">
                          {settings.repetition_penalty ? settings.repetition_penalty.toFixed(2) : "1.00"}
                        </span>
                      </div>
                      <input
                        type="range"
                        min="1.0"
                        max="1.8"
                        step="0.05"
                        value={settings.repetition_penalty || 1.15}
                        onChange={(e) =>
                          onUpdateSettings({
                            ...settings,
                            repetition_penalty: parseFloat(e.target.value),
                          })
                        }
                        className="w-full mt-1 accent-zinc-400"
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-zinc-800/80 px-5 py-3 text-right">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg bg-zinc-700 hover:bg-zinc-600 px-4 py-1.5 text-xs font-medium text-white transition"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
