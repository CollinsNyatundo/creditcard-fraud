import type {
  PredictRequest,
  PredictResponse,
  StreamTokenResponse,
  HealthResponse,
  SHAPResponse,
  ConfigResponse,
  MetricsResponse,
} from '@/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function fetchWithAuth<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const apiKey = typeof window !== 'undefined' 
    ? document.cookie
        .split('; ')
        .find((row) => row.startsWith('api_key='))
        ?.split('=')[1]
    : null;

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  };

  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export const api = {
  health: () => fetchWithAuth<HealthResponse>('/health'),

  predict: (data: PredictRequest) =>
    fetchWithAuth<PredictResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  getStreamToken: () => fetchWithAuth<StreamTokenResponse>('/auth/stream-token', {
    method: 'POST',
  }),

  getSHAP: (predictionId: string) =>
    fetchWithAuth<SHAPResponse>(`/shap/${predictionId}`),

  getConfig: (key: string) =>
    fetchWithAuth<ConfigResponse>(`/config/${key}`),

  setConfig: (key: string, value: string) =>
    fetchWithAuth<ConfigResponse>(`/config/${key}`, {
      method: 'PUT',
      body: JSON.stringify({ value }),
    }),

  getMetrics: () => fetchWithAuth<MetricsResponse>('/metrics'),

  getAllConfigs: () =>
    fetchWithAuth<ConfigResponse[]>('/config'),
};

export function setApiKeyCookie(key: string) {
  if (typeof window !== 'undefined') {
    document.cookie = `api_key=${key}; path=/; max-age=${60 * 60 * 24 * 30}; SameSite=Lax`;
  }
}

export function clearApiKeyCookie() {
  if (typeof window !== 'undefined') {
    document.cookie = 'api_key=; path=/; max-age=0';
  }
}

export function getApiKeyFromCookie(): string | null {
  if (typeof window === 'undefined') return null;
  return (
    document.cookie
      .split('; ')
      .find((row) => row.startsWith('api_key='))
      ?.split('=')[1] || null
  );
}