'use client';

import React from 'react';
import { motion } from 'framer-motion';

interface SynergyGaugeProps {
  score: number; // Scale from -1.0 to +1.0 (or -100 to +100 normalized)
  confidence?: number;
  label?: string;
}

export const SynergyGauge: React.FC<SynergyGaugeProps> = ({
  score = 0.65,
  confidence = 0.92,
  label,
}) => {
  // Clamp score to [-1, 1]
  const clampedScore = Math.max(-1, Math.min(1, score));
  
  // Normalize score to percentage [0, 1] for angle calculation
  const percentage = (clampedScore + 1) / 2;
  
  // Angle calculations for SVG arc (180 degree semi-circle, from 180deg to 0deg)
  const angle = 180 * percentage; // 0 to 180 degrees
  
  // Interpretation text & styling
  const getCategory = (val: number) => {
    if (val >= 0.4) return { text: 'Strong Synergy', color: 'text-emerald-400', border: 'border-emerald-500/40' };
    if (val >= 0.1) return { text: 'Moderate Synergy', color: 'text-violet-400', border: 'border-violet-500/40' };
    if (val >= -0.1) return { text: 'Additive Effect', color: 'text-cyan-400', border: 'border-cyan-500/40' };
    if (val >= -0.4) return { text: 'Slight Antagonism', color: 'text-amber-400', border: 'border-amber-500/40' };
    return { text: 'Strong Antagonism', color: 'text-rose-500', border: 'border-rose-500/40' };
  };

  const category = getCategory(clampedScore);

  return (
    <div className="relative flex flex-col items-center justify-center p-6 glass-card rounded-2xl w-full max-w-sm border border-white/10 shadow-glass">
      <div className="text-xs uppercase tracking-wider font-mono text-slate-400 mb-2">
        Synergy Index (Loewe / GATv2)
      </div>

      <div className="relative w-64 h-36 flex items-center justify-center overflow-hidden">
        {/* SVG Arc Gauge */}
        <svg className="w-64 h-64 -rotate-90 transform" viewBox="0 0 120 120">
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#f43f5e" />     {/* Antagonistic Red */}
              <stop offset="35%" stopColor="#fbbf24" />    {/* Yellow */}
              <stop offset="65%" stopColor="#8b5cf6" />    {/* Violet */}
              <stop offset="100%" stopColor="#10b981" />   {/* Emerald Synergy */}
            </linearGradient>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Background Arc Track */}
          <path
            d="M 20 60 A 40 40 0 0 1 100 60"
            fill="none"
            stroke="rgba(255, 255, 255, 0.1)"
            strokeWidth="10"
            strokeLinecap="round"
          />

          {/* Active Gradient Arc */}
          <motion.path
            d="M 20 60 A 40 40 0 0 1 100 60"
            fill="none"
            stroke="url(#gaugeGradient)"
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray="125.6"
            strokeDashoffset={125.6 * (1 - percentage)}
            initial={{ strokeDashoffset: 125.6 }}
            animate={{ strokeDashoffset: 125.6 * (1 - percentage) }}
            transition={{ duration: 1.2, ease: 'easeOut' }}
            filter="url(#glow)"
          />
        </svg>

        {/* Needle */}
        <motion.div
          className="absolute bottom-2 left-1/2 w-1 h-24 bg-gradient-to-t from-white via-violet-300 to-emerald-400 origin-bottom rounded-full shadow-lg"
          style={{ x: '-50%' }}
          initial={{ rotate: -90 }}
          animate={{ rotate: angle - 90 }}
          transition={{ type: 'spring', stiffness: 60, damping: 15 }}
        />

        {/* Center Cap */}
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 w-6 h-6 rounded-full bg-slate-900 border-2 border-emerald-400 shadow-violet-glow" />
      </div>

      {/* Numerical Display */}
      <div className="mt-4 flex flex-col items-center">
        <motion.span
          className="text-4xl font-extrabold font-mono tracking-tight text-white"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
        >
          {clampedScore >= 0 ? `+${clampedScore.toFixed(3)}` : clampedScore.toFixed(3)}
        </motion.span>

        <div className={`mt-2 px-3 py-1 rounded-full text-xs font-semibold border bg-slate-900/60 ${category.color} ${category.border}`}>
          {label || category.text}
        </div>

        <div className="mt-3 flex items-center space-x-2 text-[11px] text-slate-400 font-mono">
          <span>Confidence: {(confidence * 100).toFixed(0)}%</span>
          <span>&bull;</span>
          <span>Scale: -1.0 to +1.0</span>
        </div>
      </div>
    </div>
  );
};

export default SynergyGauge;
