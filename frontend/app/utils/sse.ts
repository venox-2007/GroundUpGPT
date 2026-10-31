import { MessageMetrics } from "../types";

export interface TokenEventPayload {
  token_id: number;
  text: string;
  step: number;
  generated_tokens: number;
}

export interface DoneEventPayload {
  prompt_tokens: number;
  generated_tokens: number;
  total_tokens: number;
  active_context_tokens: number;
  context_limit: number;
  context_cropped: boolean;
  elapsed_time_sec: number;
  tokens_per_second: number;
}

export interface ErrorEventPayload {
  error: string;
}

export interface StreamCallbacks {
  onToken: (token: TokenEventPayload) => void;
  onDone: (done: DoneEventPayload) => void;
  onError: (error: string) => void;
}

/**
 * Robustly parse SSE stream from a fetch Response body across chunk boundaries.
 */
export async function streamSSE(
  response: Response,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  if (!response.body) {
    callbacks.onError("Response body is not readable.");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      if (signal?.aborted) {
        break;
      }

      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      // Process complete double-newline separated event blocks
      let boundaryIndex: number;
      while ((boundaryIndex = buffer.indexOf("\n\n")) !== -1) {
        const eventBlock = buffer.slice(0, boundaryIndex).trim();
        buffer = buffer.slice(boundaryIndex + 2);

        if (!eventBlock) continue;

        parseEventBlock(eventBlock, callbacks);
      }
    }

    // Flush any trailing event
    if (buffer.trim()) {
      parseEventBlock(buffer.trim(), callbacks);
    }
  } catch (err: unknown) {
    if (signal?.aborted) {
      // User aborted stream via Stop button; not an error
      return;
    }
    const message = err instanceof Error ? err.message : String(err);
    callbacks.onError(message);
  } finally {
    reader.releaseLock();
  }
}

function parseEventBlock(eventBlock: string, callbacks: StreamCallbacks): void {
  const lines = eventBlock.split("\n");
  let eventType = "message";
  let dataStr = "";

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventType = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataStr = line.slice(5).trim();
    }
  }

  if (!dataStr) return;

  try {
    const parsed = JSON.parse(dataStr);
    if (eventType === "token") {
      callbacks.onToken(parsed as TokenEventPayload);
    } else if (eventType === "done") {
      callbacks.onDone(parsed as DoneEventPayload);
    } else if (eventType === "error") {
      callbacks.onError(parsed.error || "Unknown server error during streaming.");
    }
  } catch (err) {
    console.error("Failed to parse SSE JSON payload:", dataStr, err);
  }
}
