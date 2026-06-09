'use client';

import { useRef, useEffect, useMemo } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import { cn } from '@/lib/utils';

interface LatencyChartProps {
  data: number[];
  maxPoints?: number;
  height?: number;
}

export function LatencyChart({ data, maxPoints = 200, height = 120 }: LatencyChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const uplotRef = useRef<uPlot | null>(null);

  const series = useMemo(() => [
    {
      label: 'Latency (ms)',
      stroke: 'hsl(var(--primary))',
      fill: 'hsla(var(--primary), 0.1)',
      width: 1,
      paths: (u: uPlot) => {
        const { points } = u;
        const [x, y] = u.series[1];
        const px = Math.round;
        let path = '';
        const dataX = x as number[];
        const dataY = y as number[];
        const len = dataX.length;
        
        if (len === 0) return '';
        
        const xScale = u.bbox.width / (dataX[len - 1] - dataX[0]);
        
        path += `M${px(dataX[0] * xScale)},${px(u.bbox.height - dataY[0] * u.bbox.height)}`;
        
        for (let i = 1; i < len; i++) {
          const xPos = px((dataX[i] - dataX[0]) * xScale);
          const yPos = px(u.bbox.height - dataY[i] * u.bbox.height);
          path += `L${xPos},${yPos}`;
        }
        return new Path2D(path);
      },
    },
  ], []);

  const options = useMemo(() => ({
    width: containerRef.current?.clientWidth || 800,
    height,
    scales: {
      x: { time: false },
      y: {
        min: 0,
        auto: true,
      },
    },
    series: [
      {},
      {
        ...series[0],
        points: {
          show: false,
        },
      },
    ],
    axes: [
      { show: false },
      {
        show: true,
        stroke: 'hsl(var(--muted))',
        grid: { stroke: 'hsl(var(--border))' },
        ticks: { stroke: 'hsl(var(--muted))' },
        size: 40,
      },
    ],
    plugins: [],
    cursor: {
      show: true,
      x: true,
      y: true,
      drag: {
        setScale: false,
      },
    },
    hooks: {
      draw: [
        (u: uPlot) => {
          const ctx = u.ctx;
          const [x, y] = u.series[1];
          const dataX = x as number[];
          const dataY = y as number[];
          
          if (dataX.length < 2) return;
          
          ctx.strokeStyle = 'hsl(var(--warning))';
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(0, u.bbox.height - 5 * (u.bbox.height / (u.scales.y.max || 10)));
          ctx.lineTo(u.bbox.width, u.bbox.height - 5 * (u.bbox.height / (u.scales.y.max || 10)));
          ctx.stroke();
          ctx.setLineDash([]);
          
          ctx.strokeStyle = 'hsl(var(--success))';
          ctx.beginPath();
          ctx.moveTo(0, u.bbox.height - 1 * (u.bbox.height / (u.scales.y.max || 10)));
          ctx.lineTo(u.bbox.width, u.bbox.height - 1 * (u.bbox.height / (u.scales.y.max || 10)));
          ctx.stroke();
        },
      ],
    },
  }), [series, height]);

  useEffect(() => {
    if (!containerRef.current) return;

    const initData = [
      Array.from({ length: data.length }, (_, i) => i),
      data.slice(-maxPoints),
    ];

    uplotRef.current = new uPlot(options, initData, containerRef.current);

    return () => {
      uplotRef.current?.destroy();
      uplotRef.current = null;
    };
  }, [options]);

  useEffect(() => {
    if (!uplotRef.current) return;
    
    const newData = [
      Array.from({ length: data.length }, (_, i) => i),
      data.slice(-maxPoints),
    ];
    
    uplotRef.current.setData(newData);
  }, [data, maxPoints]);

  return (
    <div
      ref={containerRef}
      className={cn('w-full', 'bg-card/50 rounded-lg')}
      style={{ height }}
    />
  );
}