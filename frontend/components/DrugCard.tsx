'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Copy, Check, FlaskConical, ShieldCheck, ShieldAlert } from 'lucide-react';

export interface DrugCardProps {
  role: 'Drug A' | 'Drug B';
  name: string;
  smiles: string;
  mw?: number;
  logp?: number;
  tpsa?: number;
  hbd?: number;
  hba?: number;
  lipinskiPass?: boolean;
  dockingScore?: number;
}

export const DrugCard: React.FC<DrugCardProps> = ({
  role = 'Drug A',
  name = 'Olaparib',
  smiles = 'O=C1N(Cc2ccc(C(=O)N3CCN(C(=O)c4cc(F)ccc4)CC3)cc2)NC(=O)C1=O',
  mw = 434.47,
  logp = 1.85,
  tpsa = 89.2,
  hbd = 1,
  hba = 6,
  lipinskiPass = true,
  dockingScore = -9.4,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(smiles);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isRoleA = role === 'Drug A';
  const accentGradient = isRoleA
    ? 'from-violet-600 to-purple-500'
    : 'from-emerald-600 to-teal-500';
  const badgeBorder = isRoleA ? 'border-violet-500/40 text-violet-300' : 'border-emerald-500/40 text-emerald-300';

  return (
    <motion.div
      className="glass-card glass-card-hover rounded-2xl p-5 flex flex-col justify-between border border-white/10 w-full relative overflow-hidden"
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      {/* Top Accent Strip */}
      <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${accentGradient}`} />

      {/* Header */}
      <div className="flex items-center justify-between mt-1">
        <div className="flex items-center space-x-2">
          <span className={`px-2.5 py-0.5 rounded-md text-[10px] uppercase font-mono font-bold border bg-slate-900/60 ${badgeBorder}`}>
            {role}
          </span>
          <h4 className="text-lg font-bold text-white tracking-tight">{name}</h4>
        </div>
        <div className="flex items-center space-x-1 text-[11px] font-mono">
          {lipinskiPass ? (
            <span className="flex items-center text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded-full border border-emerald-800/40">
              <ShieldCheck className="w-3 h-3 mr-1" /> Lipinski Pass
            </span>
          ) : (
            <span className="flex items-center text-amber-400 bg-amber-950/40 px-2 py-0.5 rounded-full border border-amber-800/40">
              <ShieldAlert className="w-3 h-3 mr-1" /> Lipinski Viol.
            </span>
          )}
        </div>
      </div>

      {/* Structure Preview Placeholder Box */}
      <div className="my-4 relative h-28 rounded-xl bg-slate-950/70 border border-white/5 flex items-center justify-center overflow-hidden group">
        <div className="absolute inset-0 bg-grid-pattern opacity-20" />
        <div className="flex flex-col items-center justify-center text-slate-500 z-10">
          <FlaskConical className={`w-8 h-8 mb-1 ${isRoleA ? 'text-violet-400' : 'text-emerald-400'} opacity-70 group-hover:scale-110 transition-transform`} />
          <span className="text-[11px] font-mono text-slate-400">
            2D Structure Graph
          </span>
          <span className="text-[9px] font-mono text-slate-600">
            RDKit Chem.MolFromSmiles
          </span>
        </div>
      </div>

      {/* SMILES section */}
      <div className="mb-4 bg-slate-900/80 rounded-lg p-2.5 border border-white/5 flex items-center justify-between">
        <div className="flex flex-col overflow-hidden mr-2">
          <span className="text-[9px] uppercase font-mono text-slate-500">SMILES Notation</span>
          <span className="text-xs font-mono text-slate-300 truncate" title={smiles}>
            {smiles}
          </span>
        </div>
        <button
          onClick={handleCopy}
          className="p-1.5 rounded-md hover:bg-white/10 text-slate-400 hover:text-white transition-colors flex-shrink-0"
          title="Copy SMILES"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
        </button>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-3 gap-2 text-center pt-2 border-t border-white/5">
        <div className="bg-white/[0.02] p-2 rounded-lg border border-white/5">
          <div className="text-[10px] text-slate-400 font-mono">MW (g/mol)</div>
          <div className="text-xs font-bold font-mono text-slate-200 mt-0.5">{mw.toFixed(1)}</div>
        </div>
        <div className="bg-white/[0.02] p-2 rounded-lg border border-white/5">
          <div className="text-[10px] text-slate-400 font-mono">cLogP</div>
          <div className="text-xs font-bold font-mono text-slate-200 mt-0.5">{logp.toFixed(2)}</div>
        </div>
        <div className="bg-white/[0.02] p-2 rounded-lg border border-white/5">
          <div className="text-[10px] text-slate-400 font-mono">TPSA (Å²)</div>
          <div className="text-xs font-bold font-mono text-slate-200 mt-0.5">{tpsa.toFixed(1)}</div>
        </div>
      </div>

      {dockingScore !== undefined && (
        <div className="mt-3 flex items-center justify-between text-xs font-mono px-3 py-1.5 rounded-lg bg-slate-900/60 border border-white/5">
          <span className="text-slate-400">Target Docking Affinity:</span>
          <span className="font-bold text-violet-300">{dockingScore.toFixed(1)} kcal/mol</span>
        </div>
      )}
    </motion.div>
  );
};

export default DrugCard;
