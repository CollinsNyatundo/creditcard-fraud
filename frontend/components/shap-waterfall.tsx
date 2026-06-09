'use client';

import { motion } from 'framer-motion';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import type { SHAPResponse } from '@/types';

interface SHAPWaterfallProps {
  data: SHAPResponse | null;
  loading: boolean;
}

export function SHAPWaterfall({ data, loading }: SHAPWaterfallProps) {
  if (loading || !data) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            SHAP Explanation
            <span className="animate-pulse">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1 }}
                className="h-8 bg-muted/50 rounded animate-pulse"
              />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  const features = data.feature_names
    .map((name, index) => ({
      name,
      value: data.shap_values[index],
    }))
    .filter((f) => Math.abs(f.value) > 0.001)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 15);

  const maxValue = Math.max(...features.map((f) => Math.abs(f.value)));

  return (
    <Card className="h-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            SHAP Waterfall
            <Badge variant="outline" className="text-xs">
              {data.prediction_id.slice(0, 8)}
            </Badge>
          </CardTitle>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>Base: {data.base_value.toFixed(4)}</span>
            <span>Pred: {data.prediction_value.toFixed(4)}</span>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {features.map((feature, index) => (
            <motion.div
              key={feature.name}
              initial={{ opacity: 0, x: feature.value < 0 ? 20 : -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.03, duration: 0.3 }}
              className="group"
            >
              <div className="flex items-center gap-3">
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-mono text-muted-foreground truncate pr-2">
                    {feature.name}
                  </p>
                  <div className="relative h-2 rounded-full bg-muted overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{
                        width: `${(Math.abs(feature.value) / maxValue) * 100}%`,
                      }}
                      transition={{ delay: index * 0.03 + 0.1, duration: 0.5 }}
                      className="h-full rounded-full"
                      style={{
                        background: feature.value > 0
                          ? 'hsl(var(--fraud))'
                          : 'hsl(var(--success))',
                      }}
                    />
                  </div>
                </div>
                <span
                  className="text-xs font-mono font-medium w-16 text-right"
                  style={{ color: feature.value > 0 ? 'hsl(var(--fraud))' : 'hsl(var(--success))' }}
                >
                  {feature.value > 0 ? '+' : ''}{feature.value.toFixed(4)}
                </span>
              </div>
            </motion.div>
          ))}
        </div>
        <div className="mt-4 pt-4 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
          <span>
            <span className="inline-block w-3 h-3 rounded bg-fraud mr-1" /> Pushes toward fraud
          </span>
          <span>
            <span className="inline-block w-3 h-3 rounded bg-success mr-1" /> Pushes toward clear
          </span>
        </div>
      </CardContent>
    </Card>
  );
}