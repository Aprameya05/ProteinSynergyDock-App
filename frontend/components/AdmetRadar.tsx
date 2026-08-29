'use client';

import React from 'react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from 'recharts';

export interface AdmetMetric {
  property: string;
  drugA: number; // 0 to 100
  drugB: number; // 0 to 100
  synergyOpt?: number; // Optional reference optimum
}

interface AdmetRadarProps {
  data?: AdmetMetric[];
  drugAName?: string;
  drugBName?: string;
}

const defaultAdmetData: AdmetMetric[] = [
  { property: 'Absorption', drugA: 85, drugB: 72, synergyOpt: 90 },
  { property: 'Distribution', drugA: 78, drugB: 88, synergyOpt: 85 },
  { property: 'Metabolism', drugA: 65, drugB: 80, synergyOpt: 75 },
  { property: 'Excretion', drugA: 90, drugB: 60, synergyOpt: 80 },
  { property: 'Toxicity Safety', drugA: 70, drugB: 75, synergyOpt: 85 },
  { property: 'Bioavailability', drugA: 92, drugB: 84, synergyOpt: 95 },
];

export const AdmetRadar: React.FC<AdmetRadarProps> = ({
  data = defaultAdmetData,
  drugAName = 'Drug A',
  drugBName = 'Drug B',
}) => {
  return (
    <div className="glass-card rounded-2xl p-6 w-full flex flex-col items-center border border-white/10 shadow-glass">
      <div className="w-full flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold tracking-wide text-white uppercase font-mono">
            Hexagonal ADMET Radar Profile
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Comparative RDKit score across 6 key pharmacokinetic axes
          </p>
        </div>
        <div className="text-[11px] font-mono px-2 py-1 rounded bg-violet-950/60 border border-violet-800/40 text-violet-300">
          6-Axis Normalized (0-100)
        </div>
      </div>

      <div className="w-full h-72">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart cx="50%" cy="50%" outerRadius="75%" data={data}>
            <PolarGrid stroke="rgba(255, 255, 255, 0.15)" strokeDasharray="3 3" />
            <PolarAngleAxis
              dataKey="property"
              stroke="#94a3b8"
              tick={{ fill: '#cbd5e1', fontSize: 11, fontWeight: 500 }}
            />
            <PolarRadiusAxis
              angle={30}
              domain={[0, 100]}
              stroke="#64748b"
              tick={{ fill: '#64748b', fontSize: 9 }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                borderColor: 'rgba(255, 255, 255, 0.15)',
                borderRadius: '0.75rem',
                color: '#f8fafc',
                fontSize: '12px',
                boxShadow: '0 10px 25px rgba(0,0,0,0.5)',
              }}
            />
            <Radar
              name={drugAName}
              dataKey="drugA"
              stroke="#a855f7"
              fill="#8b5cf6"
              fillOpacity={0.35}
            />
            <Radar
              name={drugBName}
              dataKey="drugB"
              stroke="#34d399"
              fill="#10b981"
              fillOpacity={0.35}
            />
            <Legend
              wrapperStyle={{
                paddingTop: '10px',
                fontSize: '12px',
                color: '#94a3b8',
              }}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default AdmetRadar;
