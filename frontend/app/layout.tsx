import type { Metadata } from 'next';
import Link from 'next/link';
import './globals.css';

export const metadata: Metadata = {
  title: 'ProteinSynergyDock | GATv2 Drug Synergy Engine',
  description: 'Scientific drug synergy prediction — GATv2 GNN, AutoDock Vina, ADMET, FHIR R4, CDS Hooks.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body style={{ background: 'var(--bg)', color: 'var(--text)' }} className="antialiased flex flex-col min-h-screen">
        {/* Top Bar */}
        <header style={{ borderBottom: '1px solid var(--border)', background: '#000' }}
          className="sticky top-0 z-50 h-12 flex items-center">
          <div className="w-full max-w-screen-2xl mx-auto px-6 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-3">
              <div style={{ border: '1px solid var(--cyan)', color: 'var(--cyan)', width: 28, height: 28 }}
                className="flex items-center justify-center text-sm font-bold">Ψ</div>
              <div>
                <span style={{ color: 'var(--text)' }} className="text-xs font-bold tracking-wider uppercase">ProteinSynergyDock</span>
                <span style={{ color: 'var(--muted)', marginLeft: 8 }} className="text-[10px] tracking-widest uppercase">v3.0</span>
              </div>
            </Link>
            <nav className="flex items-center gap-6">
              <Link href="/" style={{ color: 'var(--muted)', fontSize: 11, letterSpacing: '0.1em' }}
                className="uppercase hover:text-white transition-colors">Platform</Link>
              <Link href="/predict" style={{ color: 'var(--muted)', fontSize: 11, letterSpacing: '0.1em' }}
                className="uppercase hover:text-white transition-colors">Predictor</Link>
              <a href="https://github.com/Aprameya05/ProteinSynergyDock-App" target="_blank" rel="noreferrer"
                style={{ color: 'var(--muted)', fontSize: 11, letterSpacing: '0.1em' }}
                className="uppercase hover:text-white transition-colors hidden sm:block">GitHub</a>
              <Link href="/predict"
                style={{ background: 'var(--cyan)', color: '#000', fontSize: 10, letterSpacing: '0.12em', padding: '5px 14px' }}
                className="font-bold uppercase">Launch</Link>
            </nav>
          </div>
        </header>

        <main className="flex-grow">{children}</main>

        <footer style={{ borderTop: '1px solid var(--border)', background: '#000' }} className="py-5">
          <div className="max-w-screen-2xl mx-auto px-6 flex flex-col sm:flex-row justify-between items-center gap-3">
            <span style={{ color: 'var(--muted)', fontSize: 10, letterSpacing: '0.1em' }} className="uppercase">
              ProteinSynergyDock · Research Edition · Not for clinical use
            </span>
            <div className="flex gap-3">
              {['GATv2 GNN', 'NCI ALMANAC', 'RDKit ADMET', 'FHIR R4'].map(t => (
                <span key={t} style={{ border: '1px solid var(--border2)', color: 'var(--muted)', fontSize: 9, padding: '2px 8px', letterSpacing: '0.1em' }}
                  className="uppercase">{t}</span>
              ))}
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
