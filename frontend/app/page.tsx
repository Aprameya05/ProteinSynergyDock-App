'use client';
import Link from 'next/link';

export default function LandingPage() {
  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', position: 'relative', overflow: 'hidden' }}>
      {/* Glow orbs */}
      <div style={{ position: 'absolute', top: '10%', left: '15%', width: 400, height: 400, borderRadius: '50%', background: 'radial-gradient(circle, rgba(124,58,237,0.12) 0%, transparent 70%)', pointerEvents: 'none' }} />
      <div style={{ position: 'absolute', top: '30%', right: '10%', width: 300, height: 300, borderRadius: '50%', background: 'radial-gradient(circle, rgba(6,214,160,0.10) 0%, transparent 70%)', pointerEvents: 'none' }} />

      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '100px 24px 80px', position: 'relative', zIndex: 1 }}>
        {/* Hero */}
        <div style={{ textAlign: 'center', marginBottom: 72 }}>
          <div className="badge" style={{ display: 'inline-flex', marginBottom: 28 }}>
            <div className="badge-dot" />
            GATv2 Architecture · 107,103 NCI ALMANAC Triplets
          </div>
          <h1 style={{ fontSize: 'clamp(2.8rem, 6vw, 5rem)', fontWeight: 900, lineHeight: 1.06, letterSpacing: '-0.04em', marginBottom: 24 }}>
            Precision Drug<br />
            <span className="grad-text">Synergy Engine.</span>
          </h1>
          <p style={{ fontSize: 17, color: 'var(--muted)', maxWidth: 580, margin: '0 auto 40px', lineHeight: 1.65 }}>
            Predict dual-drug synergy, molecular docking affinity, and 6-axis ADMET pharmacokinetics in real-time powered by Graph Attention Networks.
          </p>
          <div style={{ display: 'flex', gap: 14, justifyContent: 'center', flexWrap: 'wrap' }}>
            <Link href="/predict" className="btn-primary" style={{ textDecoration: 'none', display: 'inline-block', fontSize: 15 }}>
              Launch Synergy Predictor →
            </Link>
            <a href="https://github.com/Aprameya05/ProteinSynergyDock-App" target="_blank" rel="noreferrer" className="btn-ghost" style={{ textDecoration: 'none', display: 'inline-block', fontSize: 14, padding: '11px 24px' }}>
              GitHub ↗
            </a>
          </div>
        </div>

        {/* Stats */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 1, borderRadius: 12, overflow: 'hidden', border: '1px solid var(--border)', marginBottom: 56 }}>
          {[['800,000+', 'Drug Combinations'], ['0.842', 'AUC-ROC Score'], ['60', 'NCI-60 Cell Lines'], ['< 180ms', 'Inference Time']].map(([v, l]) => (
            <div key={l} style={{ padding: '28px 24px', textAlign: 'center', background: 'var(--surface)' }}>
              <div className="grad-text" style={{ fontSize: 28, fontWeight: 800, letterSpacing: '-0.02em', marginBottom: 6 }}>{v}</div>
              <div style={{ fontSize: 12, color: 'var(--muted)', fontFamily: 'JetBrains Mono, monospace', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{l}</div>
            </div>
          ))}
        </div>

        {/* Feature grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 56 }}>
          {[
            { icon: '⬡', title: 'GATv2 Graph Neural Network', desc: 'Multi-head graph attention over molecular graphs. Trained on 107,103 NCI ALMANAC triplets.', color: '#a78bfa' },
            { icon: '⚛', title: 'AutoDock Vina Docking', desc: 'Molecular docking affinity scores with 5-pose binding mode analysis and RMSD metrics.', color: '#5eead4' },
            { icon: '◈', title: '6-Axis ADMET Profiling', desc: 'Absorption, Distribution, Metabolism, Excretion, Toxicity, Bioavailability radar.', color: '#22d3ee' },
            { icon: '⬡', title: 'FHIR R4 Integration', desc: 'DiagnosticReport generation, hash-chained audit log, SMART on FHIR OAuth2.', color: '#a78bfa' },
            { icon: '⊕', title: 'CDS Hooks', desc: 'Real-time clinical decision support via medication-prescribe webhook integration.', color: '#5eead4' },
            { icon: '∿', title: 'Bliss · CI · Dose-Response', desc: 'Chou-Talalay CI, Bliss independence model, Hill equation dose-response tables.', color: '#22d3ee' },
          ].map(({ icon, title, desc, color }) => (
            <div key={title} className="glass" style={{ borderRadius: 12, padding: 24, transition: 'border-color 0.2s' }}>
              <div style={{ fontSize: 22, marginBottom: 12, color }}>{icon}</div>
              <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 8, color: 'var(--text)' }}>{title}</div>
              <div style={{ fontSize: 12, color: 'var(--muted)', lineHeight: 1.6 }}>{desc}</div>
            </div>
          ))}
        </div>

        {/* CTA */}
        <div className="glass-strong" style={{ borderRadius: 16, padding: '48px 40px', textAlign: 'center' }}>
          <h2 style={{ fontSize: 28, fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 12 }}>
            Ready to predict <span className="grad-text">synergy?</span>
          </h2>
          <p style={{ color: 'var(--muted)', fontSize: 14, marginBottom: 28 }}>Enter SMILES notation, UniProt ID, and cell line — results in under 180ms.</p>
          <Link href="/predict" className="btn-primary" style={{ textDecoration: 'none', display: 'inline-block', fontSize: 15 }}>
            Open Predictor →
          </Link>
        </div>
      </div>
    </div>
  );
}
