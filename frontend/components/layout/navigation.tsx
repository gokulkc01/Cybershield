/**
 * Navigation Component
 *
 * Main navigation bar with page selector and branding.
 */

'use client';

import { useState } from 'react';
import { Menu, Shield, X } from 'lucide-react';
import { AppState, useAppStore } from '@/lib/store';

type PageOption = AppState['selectedPage'];

export default function Navigation() {
    const selectedPage = useAppStore((state) => state.selectedPage);
    const setSelectedPage = useAppStore((state) => state.setSelectedPage);
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

    const pages: { id: PageOption; label: string; icon: string }[] = [
        { id: 'dashboard', label: 'Dashboard', icon: 'DB' },
        { id: 'analysis', label: 'Host Insights', icon: 'HI' },
        { id: 'mutation-lab', label: 'Mutation Lab', icon: 'ML' },
        { id: 'red-agent-demo', label: 'Red-Agent', icon: 'RL' },
        { id: 'robustness', label: 'Robustness', icon: 'RB' },
        { id: 'model-demo', label: 'Model Demo', icon: 'MD' },
        { id: 'upload', label: 'Upload', icon: 'UP' },
    ];

    const handlePageChange = (page: PageOption) => {
        setSelectedPage(page);
        setMobileMenuOpen(false);
    };

    return (
        <nav className="fixed top-0 left-0 right-0 bg-white border-b border-neutral-200 shadow-sm z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    <div className="flex items-center gap-2">
                        <div className="flex items-center justify-center w-8 h-8 bg-gradient-to-br from-cyber-600 to-cyber-800 rounded-lg">
                            <Shield className="w-5 h-5 text-white" />
                        </div>
                        <div className="hidden sm:block">
                            <h1 className="text-lg font-bold text-neutral-900">CyberShield v2</h1>
                            <p className="text-xs text-neutral-500">Behavioral Adversarial Robustness</p>
                        </div>
                    </div>

                    <div className="hidden md:flex items-center gap-1">
                        {pages.map((page) => (
                            <button
                                key={page.id}
                                onClick={() => handlePageChange(page.id)}
                                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${selectedPage === page.id
                                        ? 'bg-cyber-100 text-cyber-700'
                                        : 'text-neutral-700 hover:bg-neutral-100'
                                    }`}
                            >
                                <span className="mr-1 text-[10px] font-bold">{page.icon}</span>
                                {page.label}
                            </button>
                        ))}
                    </div>

                    <button
                        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                        className="md:hidden p-2 hover:bg-neutral-100 rounded-lg"
                    >
                        {mobileMenuOpen ? (
                            <X className="w-5 h-5" />
                        ) : (
                            <Menu className="w-5 h-5" />
                        )}
                    </button>
                </div>

                {mobileMenuOpen && (
                    <div className="md:hidden border-t border-neutral-200 py-2">
                        {pages.map((page) => (
                            <button
                                key={page.id}
                                onClick={() => handlePageChange(page.id)}
                                className={`w-full text-left px-4 py-2 text-sm font-medium transition-colors ${selectedPage === page.id
                                        ? 'bg-cyber-100 text-cyber-700'
                                        : 'text-neutral-700 hover:bg-neutral-100'
                                    }`}
                            >
                                <span className="mr-2 text-[10px] font-bold">{page.icon}</span>
                                {page.label}
                            </button>
                        ))}
                    </div>
                )}
            </div>
        </nav>
    );
}
