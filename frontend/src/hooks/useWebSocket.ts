import { useCallback, useEffect, useRef, useState } from "react";
import type { WebSocketMessage } from "../types";

const BASE_URL = "ws://localhost:8000";
const MAX_MESSAGES = 200;
const MAX_BACKOFF_MS = 30_000;

interface UseWebSocketResult {
  messages: WebSocketMessage[];
  isConnected: boolean;
  reset: () => void;
}

export function useWebSocket(experimentId: string | null): UseWebSocketResult {
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef(1_000);
  const mountedRef = useRef(true);

  const reset = useCallback(() => {
    setMessages([]);
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    if (!experimentId) return;

    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      if (!mountedRef.current) return;

      const url = `${BASE_URL}/ws/v1/experiment-stream/${experimentId}`;
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) {
          ws.close();
          return;
        }
        backoffRef.current = 1_000;
        setIsConnected(true);
      };

      ws.onmessage = (event: MessageEvent<string>) => {
        if (!mountedRef.current) return;
        try {
          const msg = JSON.parse(event.data) as WebSocketMessage;
          setMessages((prev) => {
            const next = [...prev, msg];
            return next.length > MAX_MESSAGES ? next.slice(-MAX_MESSAGES) : next;
          });
        } catch {
          // ignore malformed frames
        }
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        setIsConnected(false);
        const delay = backoffRef.current;
        backoffRef.current = Math.min(delay * 2, MAX_BACKOFF_MS);
        reconnectTimer = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      mountedRef.current = false;
      if (reconnectTimer !== null) clearTimeout(reconnectTimer);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
      setIsConnected(false);
    };
  }, [experimentId]);

  return { messages, isConnected, reset };
}
