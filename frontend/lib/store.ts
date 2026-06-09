import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { AuthState, WebSocketState, StreamMessage } from '@/types';

interface StoreState extends AuthState, WebSocketState {
  setApiKey: (key: string) => void;
  clearApiKey: () => void;
  setStreamToken: (token: string) => void;
  setConnected: (connected: boolean) => void;
  setConnecting: (connecting: boolean) => void;
  setError: (error: string | null) => void;
  addMessage: (message: StreamMessage) => void;
  setMessages: (messages: StreamMessage[]) => void;
  setReplayComplete: (complete: boolean) => void;
  clearMessages: () => void;
  latencyHistory: number[];
  addLatency: (latency: number) => void;
  setLatencyHistory: (history: number[]) => void;
}

export const useStore = create<StoreState>()(
  persist(
    (set, get) => ({
      // Auth state
      apiKey: null,
      streamToken: null,
      isAuthenticated: false,

      // WebSocket state
      isConnected: false,
      isConnecting: false,
      error: null,
      lastMessage: null,
      messages: [],
      replayComplete: false,

      // Latency history for charts
      latencyHistory: [],

      // Actions
      setApiKey: (key: string) =>
        set({ apiKey: key, isAuthenticated: true }),

      clearApiKey: () =>
        set({
          apiKey: null,
          streamToken: null,
          isAuthenticated: false,
          isConnected: false,
          messages: [],
          latencyHistory: [],
        }),

      setStreamToken: (token: string) => set({ streamToken: token }),

      setConnected: (connected: boolean) => set({ isConnected: connected }),

      setConnecting: (connecting: boolean) => set({ isConnecting: connecting }),

      setError: (error: string | null) => set({ error }),

      addMessage: (message: StreamMessage) =>
        set((state) => ({
          messages: [message, ...state.messages].slice(0, 500),
          lastMessage: message,
          latencyHistory: [...state.latencyHistory, message.latency_ms].slice(-200),
        })),

      setMessages: (messages: StreamMessage[]) =>
        set({
          messages,
          latencyHistory: messages.map((m) => m.latency_ms).slice(-200),
        }),

      setReplayComplete: (complete: boolean) => set({ replayComplete: complete }),

      clearMessages: () => set({ messages: [], latencyHistory: [] }),

      addLatency: (latency: number) =>
        set((state) => ({
          latencyHistory: [...state.latencyHistory, latency].slice(-200),
        })),

      setLatencyHistory: (history: number[]) => set({ latencyHistory: history }),
    }),
    {
      name: 'fraud-dashboard-store',
      partialize: (state) => ({
        apiKey: state.apiKey,
      }),
    }
  )
);