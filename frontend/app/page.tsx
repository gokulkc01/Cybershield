'use client';

import { useEffect, useState } from 'react';
import { useAppStore } from '@/lib/store';
import DashboardPage from '@/components/pages/dashboard';
import HostInsightsPage from '@/components/pages/host-insights';
import MutationLabPage from '@/components/pages/mutation-lab';
import RobustnessAnalyticsPage from '@/components/pages/robustness-analytics';
import ModelDemoPage from '@/components/pages/model-demo';
import UploadPage from '@/components/pages/upload';
import Navigation from '@/components/layout/navigation';

export default function Home() {
    const selectedPage = useAppStore((state) => state.selectedPage);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100">
                <div className="text-center">
                    <div className="animate-pulse">
                        <div className="h-12 w-12 bg-cyber-600 rounded-lg mx-auto mb-4"></div>
                        <p className="text-neutral-600">Loading CyberShield...</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
            <Navigation />
            <main className="pt-16">
                {selectedPage === 'dashboard' && <DashboardPage />}
                {selectedPage === 'analysis' && <HostInsightsPage />}
                {selectedPage === 'host-insights' && <HostInsightsPage />}
                {selectedPage === 'mutation-lab' && <MutationLabPage />}
                {selectedPage === 'robustness' && <RobustnessAnalyticsPage />}
                {selectedPage === 'model-demo' && <ModelDemoPage />}
                {selectedPage === 'upload' && <UploadPage />}
            </main>
        </div>
    );
}
