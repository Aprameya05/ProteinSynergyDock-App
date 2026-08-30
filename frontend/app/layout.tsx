import type { Metadata } from 'next';
import Link from 'next/link';
import './globals.css';

export const metadata: Metadata = {
  title: 'ProteinSynergyDock — GATv2 Drug Synergy Engine',
  description: 'GATv2 GNN drug synergy prediction. AutoDock Vina · ADMET · FHIR R4 · CDS Hooks.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ background: 'var(--bg)', color: 'var(--text)', minHeight: '100vh' }}>
        <header style={{
          position: 'fixed', top: 0, left: 0, right: 0, zIndex: 100,
          borderBottom: '1px solid var(--border)',
          backdropFilter: 'blur(20px)', WebkitBackdropFilter: 'blur(20px)',
          background: 'rgba(7,11,20,0.8)',
        }}>
          <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 24px', height: 56, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Link href="/" style={{ display: 'flex', alignItems: 'center', gap: 10, textDecoration: 'none' }}>
              <div style={{
                width: 30, height: 30, borderRadius: 8,
                background: 'linear-gradient(135deg, #7c3aed, #06d6a0)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 14, fontWeight: 800, color: '#fff',
              }}>Ψ</div>
              <span style={{ fontWeight: 700, fontSize: 15, color: 'var(--text)', letterSpacing: '-0.01em' }}>ProteinSynergyDock</span>
              <span style={{ fontSize: 10, color: 'var(--muted)', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.08em' }}>v3.0</span>
            </Link>
            <nav style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              {[['Platform', '/'], ['Predictor', '/predict'], ['GitHub', 'https://github.com/Aprameya05/ProteinSynergyDock-App']].map(([label, href]) => (
                <Link key={label} href={href} className="nav-link">{label}</Link>
              ))}
              <Link href="/predict" style={{
                background: 'linear-gradient(135deg, #7c3aed, #06d6a0)',
                color: '#fff', textDecoration: 'none',
                fontSize: 13, fontWeight: 600,
                padding: '7px 18px', borderRadius: 8,
                marginLeft: 8,
              }}>Launch →</Link>
            </nav>
          </div>
        </header>
        <main style={{ paddingTop: 56 }}>{children}</main>
      </body>
    </html>
  );
}
