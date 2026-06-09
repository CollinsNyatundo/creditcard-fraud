'use client';

import { cn } from '@/lib/utils';
import { Badge } from './badge';

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: number;
  trend?: 'up' | 'down' | 'neutral';
  unit?: string;
  icon?: React.ReactNode;
  subtitle?: string;
}

export function MetricCard({
  title,
  value,
  change,
  trend = 'neutral',
  unit,
  icon,
  subtitle,
}: MetricCardProps) {
  const trendColors = {
    up: 'text-success',
    down: 'text-fraud',
    neutral: 'text-muted-foreground',
  };

  const trendIcons = {
    up: (
      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
      </svg>
    ),
    down: (
      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
      </svg>
    ),
    neutral: (
      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14" />
      </svg>
    ),
  };

  return (
    <div className="metric-card">
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-muted-foreground truncate">{title}</p>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-2xl font-mono font-bold tabular-nums">{value}</span>
            {unit && <span className="text-sm text-muted-foreground">{unit}</span>}
          </div>
          {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
          {change !== undefined && (
            <div className="flex items-center gap-1 mt-2">
              <span className={cn('text-xs font-medium', trendColors[trend])}>
                {trendIcons[trend]}
                {change >= 0 ? '+' : ''}{change.toFixed(1)}%
              </span>
              <span className="text-xs text-muted-foreground">vs last hour</span>
            </div>
          )}
        </div>
        {icon && <div className="text-muted-foreground/50">{icon}</div>}
      </div>
    </div>
  );
}