'use client';

import { useState, useEffect } from 'react';
import { useStore } from '@/lib/store';

export default function Home() {
  const { messages, latencyHistory, isConnected, apiKey, setApiKey, clearApiKey } = useStore();
  const [inputKey, setInputKey] = useState('');
  const [showKeyInput, setShowKeyInput] = useState(false);
  const [metrics, setMetrics] = useState({ latency_p95: 0, throughput_rps: 0, queue_depth: 0, active_connections: 0 });

  useEffect(() => {
    if (!apiKey) {
      setShowKeyInput(true);
      return;
    }

    const fetchMetrics = async () => {
      try {
        const res = await fetch('/api/backend/metrics', { headers: { 'X-API-Key': apiKey } });
        if (res.ok) {
          const data = await res.json();
          setMetrics(data);
        }
      } catch (err) {
        console.error('metrics fetch failed', err);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, [apiKey]);

  const connectKey = () => {
    if (!inputKey.trim()) return;
    setApiKey(inputKey.trim());
    setShowKeyInput(false);
  };

  const fraudCount = messages.filter((m) => m.is_flagged).length;
  const totalVolume = messages.reduce((sum, m) => sum + m.amount, 0);

  if (showKeyInput) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="w-full max-w-md rounded-xl border border-border bg-card p-6 shadow-sm">
          <h1 className="text-xl font-bold">Connect to backend</h1>
          <p className="mt-2 text-sm text-muted-foreground">Enter your API key to view live predictions.</p>
          <input
            className="mt-4 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            value={inputKey}
            onChange={(e) => setInputKey(e.target.value)}
            placeholder="X-API-Key"
          />
          <button
            onClick={connectKey}
            className="mt-4 w-full rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground"
          >
            Connect
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/50">
        <div className="mx-auto max-w-6xl px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Fraud Detection</h1>
            <p className="text-xs text-muted-foreground">Real-time ML inference</p>
          </div>
          <div className="flex items-center gap-3">
            <span className={`inline-flex items-center gap-2 rounded-full px-2 py-1 text-xs ${isConnected ? 'bg-success/20 text-success' : 'bg-destructive/20 text-destructive'}`}>
              <span className={`h-2 w-2 rounded-full ${isConnected ? 'bg-success' : 'bg-destructive'}`} />
              {isConnected ? 'LIVE' : 'OFFLINE'}
            </span>
            <button
              onClick={() => { clearApiKey(); setShowKeyInput(true); }}
              className="rounded-md border border-border px-3 py-1.5 text-xs hover:bg-accent"
            >
              Switch key
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6 space-y-6">
        <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-border bg-card p-4">
            <p className="text-xs text-muted-foreground">Transactions</p>
            <p className="text-2xl font-mono font-bold">{messages.length}</p>
            <p className="text-xs text-muted-foreground">Total processed</p>
          </div>
          <div className="rounded-xl border border-border bg-card p-4">
            <p className="text-xs text-muted-foreground">Fraud Alerts</p>
            <p className="text-2xl font-mono font-bold">{fraudCount}</p>
            <p className="text-xs text-muted-foreground">{messages.length ? ((fraudCount / messages.length) * 100).toFixed(2) : '0.00'}% flag rate</p>
          </div>
          <div className="rounded-xl border border-border bg-card p-4">
            <p className="text-xs text-muted-foreground">Total Volume</p>
            <p className="text-2xl font-mono font-bold">${totalVolume.toFixed(2)}</p>
            <p className="text-xs text-muted-foreground">Transaction amount</p>
          </div>
          <div className="rounded-xl border border-border bg-card p-4">
            <p className="text-xs text-muted-foreground">P95 Latency</p>
            <p className="text-2xl font-mono font-bold">{metrics.latency_p95.toFixed(2)}ms</p>
            <p className="text-xs text-muted-foreground">Inference time</p>
          </div>
        </section>

        <section className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2 space-y-4">
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="flex items-center justify-between">
                <h2 className="font-semibold">Live Feed</h2>
                <span className="text-xs text-muted-foreground">{messages.length} messages</span>
              </div>
              <div className="mt-3 space-y-2">
                {messages.length === 0 && <p className="text-sm text-muted-foreground">Waiting for transactions...</p>}
                {messages.map((m) => (
                  <div key={m.prediction_id} className="flex items-center justify-between rounded-lg border border-border/60 bg-muted/30 px-3 py-2 text-sm">
                    <div className="flex items-center gap-3">
                      <span className={`h-2 w-2 rounded-full ${m.is_flagged ? 'bg-destructive' : 'bg-success'}`} />
                      <span className="font-mono">{m.card_id}</span>
                      <span className={`rounded-full px-2 py-0.5 text-xs ${m.is_flagged ? 'bg-destructive/20 text-destructive' : 'bg-success/20 text-success'}`}>{m.is_flagged ? 'FRAUD' : 'CLEAR'}</span>
                    </div>
                    <div className="text-right">
                      <div className="font-mono">${m.amount.toFixed(2)}</div>
                      <div className="text-xs text-muted-foreground">{m.latency_ms.toFixed(2)}ms</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="rounded-xl border border-border bg-card p-4">
              <h2 className="font-semibold">System</h2>
              <div className="mt-3 space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Queue</span>
                  <span className="font-mono">{metrics.queue_depth}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Throughput</span>
                  <span className="font-mono">{metrics.throughput_rps.toFixed(1)} RPS</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Connections</span>
                  <span className="font-mono">{metrics.active_connections}</span>
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-border bg-card p-4">
              <h2 className="font-semibold">Test Predict</h2>
              <PredictInline />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function PredictInline() {
  const { apiKey } = useStore();
  const [amount, setAmount] = useState('10');
  const [hour, setHour] = useState('12');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ prediction_id: string; is_fraud: boolean; fraud_probability: number; latency_ms: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch('/api/backend/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey || '',
        },
        body: JSON.stringify({ card_id: `demo-${Date.now()}`, amount: parseFloat(amount) || 0, hour: parseInt(hour, 10) || 12 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Prediction failed');
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={submit} className="mt-3 space-y-3">
      <input
        className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
        type="number"
        value={amount}
        onChange={(e) => setAmount(e.target.value)}
      />
      <input
        className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
        type="number"
        min="0"
        max="23"
        value={hour}
        onChange={(e) => setHour(e.target.value)}
      />
      <button
        disabled={loading || !apiKey}
        className="w-full rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
      >
        {loading ? 'Running...' : 'Run Prediction'}
      </button>
      {error && <p className="text-sm text-destructive">{error}</p>}
      {result && (
        <div className="space-y-1 rounded-md border border-border bg-muted/30 p-3 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Verdict</span>
            <span className={result.is_fraud ? 'text-destructive' : 'text-success'}>{result.is_fraud ? 'Fraud' : 'Clear'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Probability</span>
            <span className="font-mono">{(result.fraud_probability * 100).toFixed(2)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Latency</span>
            <span className="font-mono">{result.latency_ms.toFixed(2)}ms</span>
          </div>
        </div>
      )}
    </form>
  );
}