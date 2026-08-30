import type { Metadata } from 'next';
import Link from 'next/link';
import './globals.css';

export const metadata: Metadata = {
  title: 'ProteinSynergyDock — Drug Synergy Engine',
  description: 'GATv2 GNN drug synergy prediction. AutoDock Vina · ADMET · FHIR R4 · CDS Hooks.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ background: 'var(--bg)', color: 'var(--ink)' }}>
        <header style={{ borderBottom: '3px solid var(--ink)', background: 'var(--bg)', position: 'sticky', top: 0, zIndex: 50 }}>
          <div style={{ maxWidth: 1400, margin: '0 auto', padding: '0 24px', height: 52, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Link href="/" style={{ display: 'flex', alignItems: 'baseline', gap: 10, textDecoration: 'none' }}>
              <span style={{ fontFamily: 'Bebas Neue, Impact, sans-serif', fontSize: '1.4rem', color: 'var(--ink)', letterSpacing: '0.05em' }}>ProteinSynergyDock</span>
              <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 9, color: 'var(--red)', letterSpacing: '0.15em', textTransform: 'uppercase' }}>v3.0</span>
            </Link>
            <nav style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              {['Platform', 'Predictor', 'GitHub'].map((label, i) => (
                <Link key={label}
                  href={i === 1 ? '/predict' : i === 2 ? 'https://github.com/Aprameya05/ProteinSynergyDock-App' : '/'}
                  style={{ fontFamily: 'Space Mono, monospace', fontSize: 10, color: 'var(--muted)', textDecoration: 'none', padding: '6px 12px', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                  {label}
                </Link>
              ))}
              <Link href="/predict"
                style={{ fontFamily: 'Bebas Neue, Impact, sans-serif', fontSize: '1rem', background: 'var(--red)', color: '#fff', padding: '6px 18px', textDecoration: 'none', letterSpacing: '0.08em' }}>
                Launch
              </Link>
            </nav>
          </div>
        </header>

        <main style={{ flex: 1 }}>{children}</main>

        <footer style={{ borderTop: '3px solid var(--ink)', background: 'var(--surface)', marginTop: 80, padding: '24px' }}>
          <div style={{ maxWidth: 1400, margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
            <span style={{ fontFamily: 'Space Mono, monospace', fontSize: 9, color: 'var(--muted)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
              ProteinSynergyDock · Research Edition · Not for clinical use
            </span>
            <div style={{ display: 'flex', gap: 8 }}>
              {['GATv2 GNN', 'NCI ALMANAC', 'RDKit ADMET', 'FHIR R4'].map(t => (
                <span key={t} className="pill" style={{ color: 'var(--muted)' }}>{t}</span>
              ))}
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
