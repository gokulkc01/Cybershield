import type { Metadata } from 'next';
import { Toaster } from 'react-hot-toast';
import '@/styles/globals.css';

export const metadata: Metadata = {
    title: 'CyberShield v2 - Behavioral Adversarial Robustness Intelligence',
    description: 'Behavioral C2 detection and adversarial robustness evaluation platform',
    icons: {
        icon: '/favicon.ico',
    },
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" suppressHydrationWarning>
            <body>
                {children}
                <Toaster position="top-right" />
            </body>
        </html>
    );
}
