'use client';

import { useState, useEffect, Suspense } from 'react';
import { AuthProvider, useAuth } from '@/lib/auth-context';
import { MetricsCard } from '@/components/metric-card';
import { LiveFeed } from '@/components/live-feed';
import { LatencyChart } from '@/components/latency-chart';
import { ThresholdTuner } from '@/components/threshold-tuner';
import { PredictForm } from '@/components/predict-form';
import { useStore } from '@/lib/store';
import { Metrics, StreamMessage } from '@/types';

function DashboardContent() {
  const { apiKey, login } = useAuth();
  const { messages, latencyHistory, isConnected, setMessages } = useStore();
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [threshold, setThreshold] = useState(0.5);
  const [showApiKeyInput, setShowApiKeyInput] = useState(false);
  const [inputApiKey, setInputApiKey] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!apiKey) {
      setShowApiKeyInput(true);
      return;
    }

    const fetchInitialData = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // Fetch health
        const healthRes = await fetch('/api/backend/health');
        if (!healthRes.ok) throw new Error('Backend health check failed');

        // Fetch metrics
        const metricsRes = await fetch('/api/backend/metrics');
        if (metricsRes.ok) {
          const metricsData = await metricsRes.json();
          setMetrics(metricsData);
        }

        // Get stream token and connect
        const tokenRes = await fetch('/api/backend/auth/stream-token', {
          method: 'POST',
          headers: {
            'X-API-Key': apiKey,
          },
        });

        if (!tokenRes.ok) {
          throw new Error('Failed to get stream token');
        }

        const { token } = await tokenRes.json();
        connectWebSocket(token);

        // Fetch config
        const configRes = await fetch('/api/backend/config/shap_trigger_threshold', {
          headers: { 'X-API-Key': apiKey },
        });
        if (configRes.ok) {
          const config = await configRes.json();
          if (config.value) {
            setThreshold(parseFloat(config.value));
          }
        }
      } catch (err) {
        console.error('Failed to initialize dashboard:', err);
        setError(err instanceof Error ? err.message : 'Failed to connect to backend');
        setShowApiKeyInput(true);
      } finally {
        setIsLoading(false);
      }
    };

    fetchInitialData();
  }, [apiKey]);

  const connectWebSocket = (token: string) => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/api/backend/stream?token=${token}`;
    
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const message: StreamMessage = JSON.parse(event.data);
        if (message.type === 'replay' || message.type === 'live') {
          setMessages((prev) => [message, ...prev].slice(0, 500));
        }
      } catch (err) {
        console.error('Failed to parse message:', err);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
    };
  };

  const handleApiKeySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputApiKey.trim()) return;

    try {
      await login(inputApiKey.trim());
      setShowApiKeyInput(false);
    } catch (err) {
      setError('Invalid API key');
    }
  };

  const handleThresholdChange = async (value: number) => {
    try {
      const res = await fetch('/api/backend/config/shap_trigger_threshold', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey || '',
        },
        body: JSON.stringify({ value: value.toString() }),
      });
      if (res.ok) {
        setThreshold(value);
      }
    } catch (err) {
      console.error('Failed to update threshold:', err);
    }
  };

  if (showApiKeyInput) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle>Connect to Backend</CardTitle>
            <CardDescription>
              Enter your API key to connect to the fraud detection API
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleApiKeySubmit} className="space-y-4">
              <div>
                <label htmlFor="apiKey" className="text-sm font-medium">
                  API Key
                </label>
                <input
                  id="apiKey"
                  type="password"
                  value={inputApiKey}
                  onChange={(e) => setInputApiKey(e.target.value)}
                  className="mt-1 w-full px-3 py-2 bg-background border border-input rounded-md"
                  placeholder="Enter API key..."
                  autoFocus
                />
              </div>
              {error && (
                <p className="text-sm text-destructive">{error}</p>
              )}
              <Button type="submit" className="w-full">
                Connect
              </Button>
            </form>
            <p className="mt-4 text-xs text-muted-foreground">
              Get your API key from the backend service configuration.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
      </div>
    );
  }

  const fraudCount = messages.filter(m => m.is_flagged).length;
  const totalVolume = messages.reduce((sum, m) => sum + m.amount, 0);

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center">
                <Shield className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h1 className="text-xl font-bold">Fraud Detection</h1>
                <p className="text-xs text-muted-foreground">Real-time ML inference</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className={`status-badge ${isConnected ? 'status-live' : 'status-offline'}`}>
                <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-success animate-pulse' : 'bg-destructive'}`} />
                {isConnected ? 'LIVE' : 'OFFLINE'}
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowApiKeyInput(true)}
              >
                Switch Key
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6 space-y-6">
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricsCard
            title="Transactions"
            value={messages.length}
            subtitle="Total processed"
            icon={<Activity className="w-5 h-5" />}
          />
          <MetricsCard
            title="Fraud Alerts"
            value={fraudCount}
            subtitle={`${((fraudCount / Math.max(messages.length, 1)) * 100).toFixed(2)}% flag rate`}
            icon={<AlertTriangle className="w-5 h-5 text-fraud" />}
            trend={fraudCount > 0 ? 'up' : 'neutral'}
          />
          <MetricsCard
            title="Total Volume"
            value={formatCurrency(totalVolume)}
            subtitle="Transaction amount"
            icon={<DollarSign className="w-5 h-5" />}
          />
          <MetricsCard
            title="Avg Latency"
            value={formatLatency(metrics?.latency_p95 || 0)}
            subtitle="P95 inference time"
            icon={<Zap className="w-5 h-5" />}
            trend={(metrics?.latency_p95 || 0) < 10 ? 'up' : 'down'}
          />
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Live Transaction Feed</CardTitle>
                  <span className="text-sm text-muted-foreground">
                    {messages.length} transactions
                  </span>
                </div>
              </CardHeader>
              <CardContent>
                <LiveFeed messages={messages} isConnected={isConnected} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Latency Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <LatencyChart data={latencyHistory} />
              </CardContent>
            </Card>
          </div>

          <div className="space-y-6">
            <ThresholdTuner
              currentThreshold={threshold}
              onChange={handleThresholdChange}
            />

            <Card>
              <CardHeader>
                <CardTitle>Test Prediction</CardTitle>
              </CardHeader>
              <CardContent>
                <PredictForm />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>System Health</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">API Status</span>
                    <span className={`status-badge ${(metrics?.error_rate || 0) < 0.1 ? 'status-live' : 'status-delayed'}`}>
                      {(metrics?.error_rate || 0) < 0.1 ? 'Healthy' : 'Degraded'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Queue Depth</span>
                    <span className="font-mono text-sm">{metrics?.queue_depth || 0}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Throughput</span>
                    <span className="font-mono text-sm">{(metrics?.throughput_rps || 0).toFixed(1)} RPS</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Connections</span>
                    <span className="font-mono text-sm">{metrics?.active_connections || 0}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>
      </main>
    </div>
  );
}

export default function Home() {
  return (
    <AuthProvider>
      <DashboardContent />
    </AuthProvider>
  );
}