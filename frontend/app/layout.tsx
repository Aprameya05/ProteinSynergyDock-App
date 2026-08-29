import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import Link from 'next/link';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'ProteinSynergyDock | GATv2 Synergy & ADMET AI System',
  description:
    'Production AI system for predicting dual-drug synergy and ADMET profiles guided by GATv2 Graph Neural Networks and molecular docking.',
  keywords: [
    'Synergy Prediction',
    'Drug Combination',
    'Graph Neural Network',
    'ADMET Profile',
    'Oncology',
    'Cheminformatics',
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} dark`}>
      <body className="bg-[#0a0a0f] text-slate-100 antialiased flex flex-col min-h-screen selection:bg-violet-500/30 selection:text-violet-200">
        {/* Navigation Bar */}
        <header className="sticky top-0 z-50 backdrop-blur-md bg-[#0a0a0f]/80 border-b border-white/10">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <Link href="/" className="flex items-center space-x-3 group">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-violet-600 to-emerald-500 p-[1px] shadow-violet-glow transition-transform group-hover:scale-105">
                <div className="w-full h-full bg-[#0a0a0f] rounded-[11px] flex items-center justify-center">
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-emerald-400 font-bold text-lg">
                    Ψ
                  </span>
                </div>
              </div>
              <div className="flex flex-col">
                <span className="font-bold tracking-tight text-white group-hover:text-violet-300 transition-colors">
                  ProteinSynergyDock
                </span>
                <span className="text-[10px] tracking-wider font-mono text-emerald-400 uppercase font-semibold">
                  v3.0 GATv2 AI Engine
                </span>
              </div>
            </Link>

            <nav className="flex items-center space-x-6">
              <Link
                href="/"
                className="text-sm text-slate-300 hover:text-white transition-colors"
              >
                Platform
              </Link>
              <Link
                href="/predict"
                className="text-sm font-medium text-slate-200 hover:text-white transition-colors"
              >
                Predictor UI
              </Link>
              <a
                href="https://github.com/Aprameya05/ProteinSynergyDock-App"
                target="_blank"
                rel="noreferrer"
                className="text-sm text-slate-400 hover:text-slate-200 transition-colors hidden sm:block"
              >
                Docs
              </a>
              <Link
                href="/predict"
                className="relative inline-flex items-center justify-center p-0.5 overflow-hidden text-xs font-semibold rounded-lg group bg-gradient-to-br from-violet-600 to-emerald-500 group-hover:from-violet-500 group-hover:to-emerald-400 text-white shadow-violet-glow"
              >
                <span className="relative px-4 py-2 transition-all ease-in duration-75 bg-[#0a0a0f] rounded-[6px] group-hover:bg-opacity-0">
                  Launch Demo
                </span>
              </Link>
            </nav>
          </div>
        </header>

        {/* Ambient Glow Effects */}
        <div className="fixed top-12 left-1/4 w-96 h-96 glow-orb-violet -z-10 animate-pulse-glow" />
        <div className="fixed bottom-12 right-1/4 w-96 h-96 glow-orb-emerald -z-10 animate-pulse-glow" style={{ animationDelay: '2s' }} />

        {/* Main Content Area */}
        <main className="flex-grow">{children}</main>

        {/* Footer */}
        <footer className="border-t border-white/10 bg-[#07070b] py-8 text-center text-xs text-slate-500">
          <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row justify-between items-center space-y-4 sm:space-y-0">
            <div>
              <p className="font-medium text-slate-400">
                ProteinSynergyDock GATv2 AI System &bull; Research & Development Edition
              </p>
              <p className="mt-1 text-slate-600">
                Trained on ~800k Drug Pairs (DrugComb v2, NCI ALMANAC, ONEIL, SynergyFinder)
              </p>
            </div>
            <div className="flex items-center space-x-4 text-slate-400 font-mono text-[11px]">
              <span className="px-2 py-1 rounded bg-violet-950/60 border border-violet-800/40 text-violet-300">
                A100 Accelerated
              </span>
              <span className="px-2 py-1 rounded bg-emerald-950/60 border border-emerald-800/40 text-emerald-300">
                RDKit ADMET
              </span>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
