'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import type { StreamMessage } from '@/types';
import { useStore } from '@/lib/store';

interface WebSocketClientOptions {
  token: string;
  since?: string;
  onMessage: (message: StreamMessage) => void;
  onReplayComplete: () => void;
  onError: (error: string) => void;
  onConnect: () => void;
  onDisconnect: () => void;
}

export function useWebSocket({
  token,
  since,
  onMessage,
  onReplayComplete,
  onError,
  onConnect,
  onDisconnect,
}: WebSocketClientOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const maxReconnectAttempts = 10;
  const baseReconnectDelay = 1000;

  const { setConnected, setConnecting, setError, addMessage, setReplayComplete } =
    useStore();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setConnecting(true);
    setError(null);

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const wsUrl = apiUrl.replace('http', 'ws');
    const sinceParam = since ? `&since=${encodeURIComponent(since)}` : '';
    const url = `${wsUrl}/stream?token=${token}${sinceParam}`;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        setConnecting(false);
        setReconnectAttempts(0);
        onConnect();
      };

      ws.onmessage = (event) => {
        try {
          const message: StreamMessage = JSON.parse(event.data);
          
          if (message.type === 'replay') {
            onMessage(message);
          } else if (message.type === 'live') {
            onMessage(message);
            addMessage(message);
          } else if (message.type === 'error') {
            onError(message.data || 'Unknown stream error');
          }
        } catch (err) {
          console.error('Failed to parse stream message:', err);
        }
      };

      ws.onclose = (event) => {
        setConnected(false);
        setConnecting(false);
        onDisconnect();

        if (event.code !== 1000 && reconnectAttempts < maxReconnectAttempts) {
          const delay = Math.min(
            baseReconnectDelay * Math.pow(1.5, reconnectAttempts),
            30000
          );
          reconnectTimeoutRef.current = setTimeout(() => {
            setReconnectAttempts((prev) => prev + 1);
            connect();
          }, delay);
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError('Connection error. Attempting to reconnect...');
      };
    } catch (err) {
      setError('Failed to establish connection');
      setConnecting(false);
    }
  }, [
    token,
    since,
    reconnectAttempts,
    setConnected,
    setConnecting,
    setError,
    setReconnectAttempts,
    onConnect,
    onDisconnect,
    onMessage,
    addMessage,
  ]);

  useEffect(() => {
    if (!token) return;

    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted');
        wsRef.current = null;
      }
    };
  }, [token, since, connect]);

  const sendMessage = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { sendMessage, reconnectAttempts };
}

export function useStream(since?: string) {
  const { streamToken, isConnected, isConnecting, error, messages, setReplayComplete } =
    useStore();

  const handleMessage = useCallback((message: StreamMessage) => {
    // Messages are added to store via WebSocket client
  }, []);

  const handleReplayComplete = useCallback(() => {
    setReplayComplete(true);
  }, [setReplayComplete]);

  const handleError = useCallback((error: string) => {
    console.error('Stream error:', error);
  }, []);

  const handleConnect = useCallback(() => {
    console.log('Stream connected');
  }, []);

  const handleDisconnect = useCallback(() => {
    console.log('Stream disconnected');
  }, []);

  useWebSocket({
    token: streamToken || '',
    since,
    onMessage: handleMessage,
    onReplayComplete: handleReplayComplete,
    onError: handleError,
    onConnect: handleConnect,
    onDisconnect: handleDisconnect,
  });

  return {
    isConnected,
    isConnecting,
    error,
    messages,
  };
}