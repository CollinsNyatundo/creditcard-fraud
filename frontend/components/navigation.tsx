'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { Shield, Activity, BarChart2, Settings, RefreshCw } from 'lucide-react';

const navItems = [
  { href: '/', label: 'Dashboard', icon: Activity },
  { href: '/analyst', label: 'Analyst', icon: Shield },
  { href: '/ops', label: 'Operations', icon: BarChart2 },
  { href: '/model', label: 'Model', icon: Settings },
  { href: '/replay', label: 'Replay', icon: RefreshCw },
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-border bg-card/30 backdrop-blur-sm sticky top-0 z-40">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-14">
          <Link href="/" className="flex items-center gap-2 font-bold text-lg">
            <Shield className="w-6 h-6 text-primary" />
            <span className="hidden sm:inline">Fraud Detection</span>
          </Link>

          <div className="flex items-center gap-1">
            {navItems.map((item) => {
              const isActive = pathname === item.href || 
                (item.href !== '/' && pathname.startsWith(item.href));
              
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    'px-3 py-2 rounded-md text-sm font-medium transition-colors',
                    'hover:bg-accent hover:text-accent-foreground',
                    isActive && 'bg-primary/10 text-primary'
                  )}
                >
                  <item.icon className="w-4 h-4 sm:mr-2 inline" />
                  <span className="hidden md:inline">{item.label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}