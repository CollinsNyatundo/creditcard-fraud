'use client';

import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { useStore } from '@/lib/store';
import { api, setApiKeyCookie, clearApiKeyCookie, getApiKeyFromCookie } from '@/lib/api';

interface AuthContextType {
  apiKey: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (key: string) => Promise<void>;
  logout: () => void;
  refreshToken: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const { apiKey, isAuthenticated, setApiKey, clearApiKey, setStreamToken } = useStore();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const storedKey = getApiKeyFromCookie();
    if (storedKey) {
      setApiKey(storedKey);
      refreshToken();
    }
    setIsLoading(false);
  }, [setApiKey]);

  const refreshToken = async () => {
    try {
      const { token } = await api.getStreamToken();
      setStreamToken(token);
    } catch (err) {
      console.error('Failed to get stream token:', err);
    }
  };

  const login = async (key: string) => {
    setApiKey(key);
    setApiKeyCookie(key);
    await refreshToken();
  };

  const logout = () => {
    clearApiKey();
    clearApiKeyCookie();
  };

  return (
    <AuthContext.Provider
      value={{
        apiKey,
        isAuthenticated,
        isLoading,
        login,
        logout,
        refreshToken,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}