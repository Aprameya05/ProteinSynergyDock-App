'use client';

import React, { useEffect, useRef, useState } from 'react';

interface MolViewer3DProps {
  sdf: string;
  name: string;
  height?: number;
  style?: 'stick' | 'sphere' | 'cartoon' | 'surface';
  colorScheme?: 'default' | 'ssPyMol' | 'Jmol' | 'rasmol';
  backgroundColor?: string;
}

declare global {
  interface Window {
    $3Dmol: any;
  }
}

export default function MolViewer3D({
  sdf,
  name,
  height = 300,
  style = 'stick',
  colorScheme = 'default',
  backgroundColor = '0x0a0a0f',
}: MolViewer3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load 3Dmol.js from CDN
  useEffect(() => {
    if (window.$3Dmol) { setLoaded(true); return; }
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js';
    script.async = true;
    script.onload = () => setLoaded(true);
    script.onerror = () => setError('Failed to load 3Dmol.js');
    document.head.appendChild(script);
    return () => { document.head.removeChild(script); };
  }, []);

  // Initialize viewer when 3Dmol is loaded and SDF changes
  useEffect(() => {
    if (!loaded || !containerRef.current || !sdf) return;
    try {
      if (viewerRef.current) {
        viewerRef.current.clear();
      } else {
        viewerRef.current = window.$3Dmol.createViewer(containerRef.current, {
          backgroundColor: backgroundColor,
          antialias: true,
        });
      }

      const viewer = viewerRef.current;
      viewer.addModel(sdf, 'sdf');

      if (style === 'stick') {
        viewer.setStyle({}, { stick: { colorscheme: colorScheme === 'default' ? 'elementColors' : colorScheme, radius: 0.15 }, sphere: { colorscheme: colorScheme === 'default' ? 'elementColors' : colorScheme, radius: 0.35 } });
      } else if (style === 'sphere') {
        viewer.setStyle({}, { sphere: { colorscheme: 'elementColors' } });
      } else {
        viewer.setStyle({}, { stick: { colorscheme: 'elementColors', radius: 0.15 } });
      }

      // Add surface with transparency
      viewer.addSurface(window.$3Dmol.SurfaceType.VDW, {
        opacity: 0.12,
        colorscheme: 'whiteCarbon',
      });

      viewer.zoomTo();
      viewer.zoom(0.85);
      viewer.render();
    } catch (e: any) {
      setError('Viewer error: ' + e.message);
    }
  }, [loaded, sdf, style, colorScheme, backgroundColor]);

  if (error) {
    return (
      <div className="flex items-center justify-center rounded-xl bg-black/30 border border-white/10 text-xs text-rose-400 font-mono" style={{ height }}>
        {error}
      </div>
    );
  }

  return (
    <div className="relative rounded-xl overflow-hidden border border-white/10" style={{ height }}>
      {!loaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
          <div className="flex items-center gap-2 text-xs text-slate-400 font-mono">
            <div className="w-4 h-4 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" />
            Loading 3D viewer...
          </div>
        </div>
      )}
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      <div className="absolute bottom-2 left-2 text-[10px] font-mono text-slate-500 bg-black/60 px-2 py-0.5 rounded">
        {name} · 3Dmol.js
      </div>
    </div>
  );
}
