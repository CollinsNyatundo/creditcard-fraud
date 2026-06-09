'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Loader2 } from 'lucide-react';
import { useAuth } from '@/lib/auth-context';
import { api } from '@/lib/api';
import { formatCurrency, formatLatency, formatProbability } from '@/lib/utils';

export function PredictForm() {
  const { apiKey } = useAuth();
  const [cardId, setCardId] = useState('');
  const [amount, setAmount] = useState('');
  const [hour, setHour] = useState('12');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    prediction_id: string;
    is_fraud: boolean;
    fraud_probability: number;
    latency_ms: number;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const amountValue = parseFloat(amount);
      const hourValue = parseInt(hour, 10);

      const response = await api.predict({
        card_id: cardId.trim() || `demo-${Date.now()}`,
        amount: isNaN(amountValue) ? 0 : amountValue,
        hour: isNaN(hourValue) ? 12 : hourValue,
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Test Prediction</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium" htmlFor="cardId">
              Card ID
            </label>
            <input
              id="cardId"
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              placeholder="Optional"
              value={cardId}
              onChange={(event) => setCardId(event.target.value)}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium" htmlFor="amount">
              Amount
            </label>
            <input
              id="amount"
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              type="number"
              min="0"
              step="0.01"
              placeholder="0.00"
              value={amount}
              onChange={(event) => setAmount(event.target.value)}
              required
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium" htmlFor="hour">
              Hour of day
            </label>
            <input
              id="hour"
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              type="number"
              min="0"
              max="23"
              value={hour}
              onChange={(event) => setHour(event.target.value)}
              required
            />
          </div>

          <Button type="submit" className="w-full" disabled={loading || !apiKey}>
            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {apiKey ? 'Run Prediction' : 'Connect API Key First'}
          </Button>
        </form>

        {error && <p className="mt-3 text-sm text-destructive">{error}</p>}

        {result && (
          <div className="mt-4 space-y-2 rounded-md border border-border bg-muted/30 p-3 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Verdict</span>
              <span className={result.is_fraud ? 'text-destructive' : 'text-success'}>
                {result.is_fraud ? 'Fraud' : 'Clear'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Probability</span>
              <span className="font-mono">{formatProbability(result.fraud_probability)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Latency</span>
              <span className="font-mono">{formatLatency(result.latency_ms)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">ID</span>
              <span className="font-mono">{result.prediction_id}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}