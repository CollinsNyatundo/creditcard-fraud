'use client';

import { MetricsCard } from '@/components/metric-card';
import { LiveFeed } from '@/components/live-feed';
import { LatencyChart } from '@/components/latency-chart';
import { useStore } from '@/lib/store';
import { formatCurrency, formatLatency } from '@/lib/utils';
import { Activity, ShieldAlert, TrendingUp, DollarSign, Zap } from 'lucide-react';
import { useEffect, useState } from 'react';

const REPLAY_WINDOW = 300; // seconds (5 minutes)

export default function ReplayPage() {
  const { messages, latencyHistory, isConnected } = useStore();
  const [since, setSince] = useState<string>('');
  const [isReplaying, setIsReplaying] = useState(false);

  useEffect(() => {
    const cutoff = new Date(Date.now() - REPLAY_WINDOW * 1000).toISOString();
    setSince(cutoff);
  }, []);

  const replayMessages = messages.filter((m) => m.type === 'replay');
  const liveMessages = messages.filter((m) => m.type === 'live');

  const fraudCount = replayMessages.filter((m) => m.is_flagged).length;
  const totalVolume = replayMessages.reduce((sum, m) => sum + m.amount, 0);
  const avgLatency = replayMessages.length
    ? replayMessages.reduce((sum, m) => sum + m.latency_ms, 0) / replayMessages.length
    : 0;

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold">Replay Mode</h1>
              <p className="text-xs text-muted-foreground">
                Time-travel through transaction history
              </p>
            </div>
            <div className="flex items-center gap-4">
              <div className={`status-badge ${isConnected ? 'status-live' : 'status-offline'}`}>
                <span
                  className={`w-2 h-2 rounded-full ${
                    isConnected ? 'bg-success animate-pulse' : 'bg-destructive'
                  }`}
                />
                {isConnected ? 'LIVE' : 'OFFLINE'}
              </div>
              <Clock className="w-4 h-4 text-muted-foreground" />
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6 space-y-6">
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricsCard
            title="Replay Transactions"
            value={replayMessages.length}
            subtitle={`${REPLAY_WINDOW}s window`}
            icon={<Activity className="w-5 h-5" />}
          />
          <MetricsCard
            title="Fraud Alerts"
            value={fraudCount}
            subtitle={`${((fraudCount / Math.max(replayMessages.length, 1)) * 100).toFixed(2)}% flag rate`}
            icon={<ShieldAlert className="w-5 h-5 text-fraud" />}
          />
          <MetricsCard
            title="Replay Volume"
            value={formatCurrency(totalVolume)}
            subtitle="Transaction amount"
            icon={<TrendingUp className="w-5 h-5" />}
          />
          <MetricsCard
            title="Avg Latency"
            value={formatLatency(avgLatency)}
            subtitle="Average inference time"
            icon={<Zap className="w-5 h-5" />}
          />
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Replay Feed</CardTitle>
                <CardDescription>
                  Showing transactions from the last {REPLAY_WINDOW} seconds
                </CardDescription>
              </CardHeader>
              <CardContent>
                <LiveFeed
                  messages={replayMessages}
                  isConnected={isConnected}
                  maxMessages={200}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Replay Latency</CardTitle>
              </CardHeader>
              <CardContent>
                <LatencyChart data={latencyHistory.slice(0, 200)} />
              </CardContent>
            </Card>
          </div>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Replay Controls</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="text-sm font-medium">Since (ISO-8601)</label>
                  <input
                    type="text"
                    value={since}
                    onChange={(e) => setSince(e.target.value)}
                    className="mt-1 w-full px-3 py-2 bg-background border border-input rounded-md font-mono text-sm"
                    placeholder="2024-01-01T00:00:00Z"
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    Select a timestamp to replay from
                  </p>
                </div>

                <div className="space-y-2">
                  {[
                    { label: 'Last 5 min', seconds: 300 },
                    { label: 'Last 15 min', seconds: 900 },
                    { label: 'Last 1 hour', seconds: 3600 },
                    { label: 'Last 6 hours', seconds: 21600 },
                  ].map((preset) => (
                    <button
                      key={preset.seconds}
                      onClick={() =>
                        setSince(
                          new Date(Date.now() - preset.seconds * 1000).toISOString()
                        )
                      }
                      className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors ${
                        Math.abs(Date.now() - new Date(since).getTime() - preset.seconds * 1000) < 1000
                          ? 'bg-primary/20 border border-primary'
                          : 'bg-muted/50 hover:bg-muted'
                      }`}
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Live Stream</CardTitle>
                <CardDescription>
                  Concurrent live transactions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {liveMessages.slice(0, 10).map((msg) => (
                    <div
                      key={msg.prediction_id}
                      className="flex items-center justify-between p-2 rounded-md bg-muted/50"
                    >
                      <div className="flex items-center gap-2">
                        <span
                          className={`w-2 h-2 rounded-full ${
                            msg.is_flagged ? 'bg-fraud' : 'bg-success'
                          }`}
                        />
                        <span className="font-mono text-xs">{msg.card_id}</span>
                      </div>
                      <span className="text-xs text-muted-foreground font-mono">
                        {formatLatency(msg.latency_ms)}
                      </span>
                    </div>
                  ))}
                  {liveMessages.length === 0 && (
                    <p className="text-sm text-muted-foreground">No live transactions</p>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </section>
      </main>
    </div>
  );
}