'use client';

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import {
  Sparkles,
  Dna,
  Zap,
  Activity,
  ShieldCheck,
  ArrowRight,
  Database,
  Layers,
  Cpu,
  Award,
} from 'lucide-react';

export default function LandingPage() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
  };

  return (
    <div className="relative overflow-hidden">
      {/* Background Hero Grid */}
      <div className="absolute inset-0 bg-grid-pattern opacity-15 pointer-events-none -z-10" />

      {/* Hero Section */}
      <section className="relative pt-24 pb-20 md:pt-32 md:pb-28 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col items-center text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
          className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-violet-950/70 border border-violet-700/50 text-violet-300 text-xs font-mono font-medium mb-8 shadow-violet-glow"
        >
          <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
          <span>GATv2 Architecture &bull; 800,000 Combination Triplets</span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.1 }}
          className="text-4xl sm:text-6xl md:text-7xl font-extrabold tracking-tight text-white max-w-4xl leading-[1.1]"
        >
          Precision Drug Combination Synergy &{' '}
          <span className="gradient-text-violet-emerald">
            ADMET AI System
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2 }}
          className="mt-6 text-lg sm:text-xl text-slate-300 max-w-2xl font-normal leading-relaxed"
        >
          Predict dual-drug synergy, molecular docking affinity, and 6-axis ADMET pharmacokinetics in real-time powered by Graph Attention Networks.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.3 }}
          className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4 w-full max-w-md"
        >
          <Link
            href="/predict"
            className="w-full sm:w-auto px-8 py-4 rounded-xl bg-gradient-to-r from-violet-600 to-emerald-500 hover:from-violet-500 hover:to-emerald-400 text-white font-semibold text-base shadow-violet-glow flex items-center justify-center space-x-2 transition-all transform hover:-translate-y-0.5"
          >
            <span>Launch Synergy Predictor</span>
            <ArrowRight className="w-5 h-5" />
          </Link>
          <a
            href="#architecture"
            className="w-full sm:w-auto px-8 py-4 rounded-xl glass-card hover:bg-white/10 text-slate-200 font-medium text-base border border-white/10 flex items-center justify-center transition-all"
          >
            Explore Architecture
          </a>
        </motion.div>

        {/* Live Metrics Highlights */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-4 w-full max-w-4xl"
        >
          {[
            { label: 'Combined Pair Samples', val: '800,000+', icon: Database, color: 'text-violet-400' },
            { label: 'Pearson Correlation r', val: '0.842', icon: Activity, color: 'text-emerald-400' },
            { label: 'NCI-60 Cell Line Models', val: '60 Lines', icon: Dna, color: 'text-cyan-400' },
            { label: 'Inference Latency', val: '< 180 ms', icon: Zap, color: 'text-amber-400' },
          ].map((stat, i) => (
            <motion.div
              key={i}
              variants={itemVariants}
              className="glass-card rounded-2xl p-5 border border-white/10 flex flex-col items-center"
            >
              <stat.icon className={`w-6 h-6 ${stat.color} mb-2`} />
              <span className="text-2xl font-extrabold font-mono text-white tracking-tight">
                {stat.val}
              </span>
              <span className="text-xs text-slate-400 mt-1">{stat.label}</span>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* Feature Grid */}
      <section id="architecture" className="py-20 relative border-t border-white/10 bg-[#07070c]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-xs uppercase tracking-widest font-mono text-emerald-400 font-semibold">
              Deep Learning Engine
            </h2>
            <p className="mt-2 text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
              Multimodal Graph Neural Network & Docking Pipeline
            </p>
            <p className="mt-4 text-slate-400">
              End-to-end synergy prediction combining molecular graphs, gene ontology target embeddings, and RDKit ADMET profiles.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="glass-card glass-card-hover rounded-2xl p-8 border border-white/10 flex flex-col">
              <div className="w-12 h-12 rounded-xl bg-violet-950/80 border border-violet-700/50 flex items-center justify-center text-violet-400 mb-6 shadow-violet-glow">
                <Cpu className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">GATv2 Graph Encoder</h3>
              <p className="text-sm text-slate-400 leading-relaxed">
                4-head dynamic graph attention network operating directly on 2D chemical bond matrices and atom features to construct invariant drug embeddings.
              </p>
            </div>

            <div className="glass-card glass-card-hover rounded-2xl p-8 border border-white/10 flex flex-col">
              <div className="w-12 h-12 rounded-xl bg-emerald-950/80 border border-emerald-700/50 flex items-center justify-center text-emerald-400 mb-6 shadow-emerald-glow">
                <Layers className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Cross-Drug Attention</h3>
              <p className="text-sm text-slate-400 leading-relaxed">
                Bidirectional Multi-Head Attention layer captures polypharmacological cross-talk between Drug A and Drug B at active binding sites.
              </p>
            </div>

            <div className="glass-card glass-card-hover rounded-2xl p-8 border border-white/10 flex flex-col">
              <div className="w-12 h-12 rounded-xl bg-cyan-950/80 border border-cyan-700/50 flex items-center justify-center text-cyan-400 mb-6 shadow-glass">
                <ShieldCheck className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">6-Axis ADMET Radar</h3>
              <p className="text-sm text-slate-400 leading-relaxed">
                RDKit-driven evaluation of Delaney solubility, Clark BBB penetration, QED drug-likeness, TPSA, and Lipinski compliance.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Symposium CTA Banner */}
      <section className="py-16 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="glass-card rounded-3xl p-8 sm:p-12 border border-violet-500/30 relative overflow-hidden flex flex-col md:flex-row items-center justify-between gap-8 shadow-violet-glow">
          <div className="absolute top-0 right-0 w-96 h-96 glow-orb-emerald -z-10" />
          
          <div className="max-w-2xl">
            <div className="flex items-center space-x-2 text-xs font-mono text-emerald-400 uppercase font-semibold mb-2">
              <Award className="w-4 h-4" />
              <span>National Industry Symposium Edition</span>
            </div>
            <h3 className="text-2xl sm:text-3xl font-extrabold text-white">
              Ready to analyze drug pairs in real-time?
            </h3>
            <p className="mt-2 text-slate-300 text-sm">
              Input SMILES strings or choose from pre-curated oncological pairs to compute synergy scores and full ADMET pharmacokinetic profiles.
            </p>
          </div>

          <Link
            href="/predict"
            className="px-8 py-4 rounded-xl bg-gradient-to-r from-violet-600 to-emerald-500 hover:from-violet-500 hover:to-emerald-400 text-white font-bold text-sm shadow-violet-glow flex-shrink-0 transition-transform hover:scale-105"
          >
            Launch Predictor App &rarr;
          </Link>
        </div>
      </section>
    </div>
  );
}
