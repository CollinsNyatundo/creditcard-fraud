'use client';

import { useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TransactionCard } from './transaction-card';
import type { StreamMessage } from '@/types';
import { cn } from '@/lib/utils';

interface LiveFeedProps {
  messages: StreamMessage[];
  isConnected: boolean;
  isLoading?: boolean;
  maxMessages?: number;
}

export function LiveFeed({ messages, isConnected, isLoading, maxMessages = 100 }: LiveFeedProps) {
  const displayMessages = useMemo(
    () => messages.slice(0, maxMessages),
    [messages, maxMessages]
  );

  if (displayMessages.length === 0 && !isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
        <div className="animate-pulse mb-4">
          <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <p className="text-sm">Waiting for transactions...</p>
        <p className="text-xs mt-1">
          {isConnected ? 'Live stream active' : 'Connecting...'}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2 max-h-[calc(100vh-280px)] overflow-y-auto scrollbar-hide pr-2">
      <AnimatePresence initial={false}>
        {displayMessages.map((message, index) => (
          <motion.div
            key={message.prediction_id}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
          >
            <TransactionCard data={message} index={index} />
          </motion.div>
        ))}
      </AnimatePresence>
      
      {isLoading && (
        <div className="flex items-center justify-center py-4">
          <div className="animate-spin rounded-full h-6 w-6 border-2 border-primary border-t-transparent" />
        </div>
      )}
    </div>
  );
}