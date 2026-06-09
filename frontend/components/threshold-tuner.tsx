'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import type { ThresholdTunerProps } from '@/types';

export function ThresholdTuner({ currentThreshold, onChange }: ThresholdTunerProps) {
  const [localThreshold, setLocalThreshold] = useState(currentThreshold);
  const [isUpdating, setIsUpdating] = useState(false);

  const handleSubmit = async () => {
    setIsUpdating(true);
    try {
      await onChange(localThreshold);
      setLocalThreshold(localThreshold);
    } catch (err) {
      console.error('Failed to update threshold:', err);
      setLocalThreshold(currentThreshold);
    } finally {
      setIsUpdating(false);
    }
  };

  const getThresholdColor = (value: number) => {
    if (value >= 0.8) return 'hsl(var(--fraud))';
    if (value >= 0.5) return 'hsl(var(--warning))';
    return 'hsl(var(--success))';
  };

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Decision Threshold
          <span className="text-sm font-mono text-muted-foreground">
            ({currentThreshold.toFixed(2)})
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span className="text-muted-foreground">Conservative</span>
              <span className="font-mono" style={{ color: getThresholdColor(localThreshold) }}>
                {localThreshold.toFixed(2)}
              </span>
              <span className="text-muted-foreground">Aggressive</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={localThreshold}
              onChange={(e) => setLocalThreshold(parseFloat(e.target.value))}
              className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-background"
              style={{
                background: `linear-gradient(to right, hsl(var(--success)), hsl(var(--warning)), hsl(var(--fraud)))`,
              }}
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>0.00</span>
              <span>0.50</span>
              <span>1.00</span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2 text-center">
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-3 rounded-lg bg-muted/50"
            >
              <div className="text-2xl font-mono font-bold text-success">High Recall</div>
              <div className="text-xs text-muted-foreground">Catch more fraud</div>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="p-3 rounded-lg bg-muted/50"
            >
              <div className="text-2xl font-mono font-bold text-warning">Balanced</div>
              <div className="text-xs text-muted-foreground">F1-optimal</div>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="p-3 rounded-lg bg-muted/50"
            >
              <div className="text-2xl font-mono font-bold text-fraud">High Precision</div>
              <div className="text-xs text-muted-foreground">Fewer false alarms</div>
            </motion.div>
          </div>

          <Button
            onClick={handleSubmit}
            disabled={isUpdating || localThreshold === currentThreshold}
            className="w-full"
            loading={isUpdating}
          >
            Apply Threshold
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}