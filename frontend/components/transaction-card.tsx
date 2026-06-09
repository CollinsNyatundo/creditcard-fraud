'use client';

import { motion } from 'framer-motion';
import { Badge } from '@/components/ui/badge';
import { formatCurrency, formatLatency, formatProbability, formatTimestamp, getProbabilityColor } from '@/lib/utils';
import type { StreamMessage } from '@/types';

interface TransactionCardProps {
  data: StreamMessage;
  index: number;
}

export function TransactionCard({ data, index }: TransactionCardProps) {
  const probColor = getProbabilityColor(data.fraud_probability);
  const latencyColor = data.latency_ms > 10 ? 'hsl(var(--fraud))' : 
                       data.latency_ms > 5 ? 'hsl(var(--warning))' : 'hsl(var(--success))';

  return (
    <motion.div
      initial={{ opacity: 0, x: -20, y: 10 }}
      animate={{ opacity: 1, x: 0, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.02 }}
      className="glass rounded-xl p-4 border-l-4 transition-all duration-300 hover:shadow-lg"
      style={{
        borderLeftColor: data.is_flagged ? 'hsl(var(--fraud))' : 'hsl(var(--success))',
        background: `linear-gradient(90deg, ${data.is_flagged ? 'hsl(var(--fraud)/0.05)' : 'hsl(var(--success)/0.05)'}, transparent)`,
      }}
    >
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4 flex-1 min-w-0">
          <div
            className="probability-ring w-10 h-10 flex-shrink-0"
            style={{
              background: `conic-gradient(${probColor} ${data.fraud_probability * 360}deg, hsl(var(--muted)) 0deg)`,
            }}
          >
            <div className="w-8 h-8 rounded-full bg-card flex items-center justify-center">
              <span className="text-xs font-mono font-bold" style={{ color: probColor }}>
                {formatProbability(data.fraud_probability)}
              </span>
            </div>
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-mono text-sm font-medium truncate max-w-[150px]">
                {data.card_id}
              </span>
              <Badge
                variant={data.is_flagged ? 'destructive' : 'success'}
                className="text-xs"
              >
                {data.is_flagged ? 'FRAUD' : 'CLEAR'}
              </Badge>
              <Badge variant="outline" className="text-xs">
                {data.type === 'replay' ? 'REPLAY' : 'LIVE'}
              </Badge>
            </div>
            <div className="flex items-center gap-4 mt-1 text-sm text-muted-foreground">
              <span className="font-mono">{formatCurrency(data.amount)}</span>
              <span className="hidden sm:inline">{formatTimestamp(data.created_at)}</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-6 text-right flex-shrink-0">
          <div>
            <div className="text-xs text-muted-foreground">LATENCY</div>
            <div className="font-mono font-medium" style={{ color: latencyColor }}>
              {formatLatency(data.latency_ms)}
            </div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground">ID</div>
            <div className="font-mono text-xs truncate max-w-[120px]">
              {data.prediction_id.slice(0, 12)}...
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}