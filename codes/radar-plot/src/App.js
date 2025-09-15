import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

const OptimizerRadarPlot = () => {
  const svgRef = useRef();
  const [selectedOptimizers, setSelectedOptimizers] = useState({
    PLOJF: true,
    ETO: true,
    MGO: true,
    SBO: true,
    MSAO: true,
    CLPSO: true,
    LSHADE: true,
    RMRIME: true,
    MDPLO: true,
    PSMADE: true
  });
  const [colorScheme, setColorScheme] = useState('default'); // Change to 'default'

  // REPLACE COLOR SCHEMES WITH THE COMPREHENSIVE VERSION
  const colorSchemes = {
    default: {
      name: 'Default',
      rank: d3.interpolateRdYlGn,
      arv: d3.interpolateViridis
    },
    blues: {
      name: 'Blues',
      rank: d3.interpolateBlues,
      arv: d3.interpolateGreens
    },
    plasma: {
      name: 'Plasma',
      rank: d3.interpolatePlasma,
      arv: d3.interpolateInferno
    },
    colorblind: {
      name: 'Colorblind Safe',
      rank: d3.interpolateCividis,
      arv: d3.interpolateCividis
    },
    monochrome: {
      name: 'Monochrome',
      rank: d3.interpolateGreys,
      arv: d3.interpolateBlues
    },
    warm: {
      name: 'Warm',
      rank: d3.interpolateWarm,
      arv: d3.interpolateOrRd
    },
    cool: {
      name: 'Cool',
      rank: d3.interpolateCool,
      arv: d3.interpolateBuGn
    },
    sunset: {
      name: 'Sunset',
      rank: d3.interpolateYlOrRd,
      arv: d3.interpolateOrRd
    },
    ocean: {
      name: 'Ocean',
      rank: d3.interpolateBuPu,
      arv: d3.interpolateGnBu
    },
    forest: {
      name: 'Forest',
      rank: d3.interpolateYlGn,
      arv: d3.interpolateGreens
    },
    purple: {
      name: 'Purple',
      rank: d3.interpolatePurples,
      arv: d3.interpolateBuPu
    },
    rainbow: {
      name: 'Rainbow',
      rank: d3.interpolateRainbow,
      arv: d3.interpolateTurbo
    },
    earth: {
      name: 'Earth',
      rank: d3.interpolateBrBG,
      arv: d3.interpolateRdYlBu
    },
    magma: {
      name: 'Magma',
      rank: d3.interpolateMagma,
      arv: d3.interpolateInferno
    },
    spring: {
      name: 'Spring',
      rank: d3.interpolateYlGnBu,
      arv: d3.interpolatePiYG
    },
    autumn: {
      name: 'Autumn',
      rank: d3.interpolateRdYlBu,
      arv: d3.interpolateYlOrBr
    },
    neon: {
      name: 'Neon',
      rank: d3.interpolateTurbo,
      arv: d3.interpolateRainbow
    },
    grayscale: {
      name: 'Grayscale',
      rank: d3.interpolateGreys,
      arv: d3.interpolateGreys
    },
    diverging: {
      name: 'Diverging',
      rank: d3.interpolateRdBu,
      arv: d3.interpolatePuOr
    },
    spectral: {
      name: 'Spectral',
      rank: d3.interpolateSpectral,
      arv: d3.interpolateRdYlGn
    },
    pastel: {
      name: 'Pastel',
      rank: d3.interpolateRgb("#FFB6C1", "#87CEEB"),
      arv: d3.interpolateRgb("#98FB98", "#DDA0DD")
    },
    mint: {
      name: 'Mint',
      rank: d3.interpolateRgb("#F0FFF0", "#00CED1"),
      arv: d3.interpolateRgb("#E0FFFF", "#20B2AA")
    },
    coral: {
      name: 'Coral',
      rank: d3.interpolateRgb("#FFF8DC", "#FF7F50"),
      arv: d3.interpolateRgb("#FFEFD5", "#CD5C5C")
    },
    lavender: {
      name: 'Lavender',
      rank: d3.interpolateRgb("#F8F8FF", "#9370DB"),
      arv: d3.interpolateRgb("#E6E6FA", "#8A2BE2")
    }
  };

  const downloadPNG = () => {
    const svg = svgRef.current;
    if (!svg) return;

    // Set high DPI (300 DPI equivalent)
    const scaleFactor = 4; // 4x scale for high resolution
    const originalWidth = 800;
    const originalHeight = 900;
    const highResWidth = originalWidth * scaleFactor;
    const highResHeight = originalHeight * scaleFactor;
    
    // Clone and prepare SVG
    const svgClone = svg.cloneNode(true);
    svgClone.setAttribute('width', highResWidth);
    svgClone.setAttribute('height', highResHeight);
    
    // Scale all elements
    const g = svgClone.querySelector('g');
    if (g) {
      const currentTransform = g.getAttribute('transform') || '';
      g.setAttribute('transform', `scale(${scaleFactor}) ${currentTransform}`);
    }
    
    const svgData = new XMLSerializer().serializeToString(svgClone);
    const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = highResWidth;
    canvas.height = highResHeight;
    
    const img = new Image();
    
    img.onload = () => {
      try {
        // Fill white background
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw the high-res image
        ctx.drawImage(img, 0, 0);
        
        // Force download using a different approach
        canvas.toBlob((blob) => {
          if (blob) {
            // Create download using multiple fallback methods
            const url = URL.createObjectURL(blob);
            
            // Method 1: Try direct download
            const link = document.createElement('a');
            link.href = url;
            link.download = 'optimizer-radar-chart-300dpi.png';
            link.style.display = 'none';
            
            // Add to DOM, click, and remove
            document.body.appendChild(link);
            link.click();
            
            // Clean up after a delay
            setTimeout(() => {
              document.body.removeChild(link);
              URL.revokeObjectURL(url);
            }, 100);
            
            // Method 2: Fallback - open in new tab if direct download fails
            setTimeout(() => {
            // Add this comment to disable the ESlint rule for the next line
            // eslint-disable-next-line no-restricted-globals
            if (confirm('If download didn\'t start, click OK to open image in new tab (then right-click to save)')) {                window.open(url, '_blank');
              }
            }, 1000);
          }
        }, 'image/png', 1.0); // Maximum quality
        
      } catch (error) {
        console.error('Canvas drawing failed:', error);
        alert('Download failed. Please try again.');
      }
    };
    
    img.onerror = (error) => {
      console.error('Image loading failed:', error);
      alert('Failed to generate image. Please try again.');
    };
    
    // Use object URL instead of data URL for better compatibility
    const url = URL.createObjectURL(svgBlob);
    img.src = url;
  };

const data = [
  { func: 'F1', PLOJF: 1, PLOJ: 4, PLOF: 2, PLO: 3 },
  { func: 'F2', PLOJF: 3, PLOJ: 2, PLOF: 1, PLO: 4 },
  { func: 'F3', PLOJF: 3, PLOJ: 4, PLOF: 2, PLO: 1 },
  { func: 'F4', PLOJF: 2, PLOJ: 1, PLOF: 3, PLO: 4 },
  { func: 'F5', PLOJF: 3, PLOJ: 2, PLOF: 1, PLO: 4 },
  { func: 'F6', PLOJF: 2, PLOJ: 3, PLOF: 1, PLO: 4 },
  { func: 'F7', PLOJF: 3, PLOJ: 2, PLOF: 1, PLO: 4 },
  { func: 'F8', PLOJF: 3, PLOJ: 1, PLOF: 2, PLO: 4 },
  { func: 'F9', PLOJF: 3, PLOJ: 4, PLOF: 2, PLO: 1 },
  { func: 'F10', PLOJF: 1, PLOJ: 2, PLOF: 3, PLO: 4 },
  { func: 'F11', PLOJF: 1, PLOJ: 3, PLOF: 2, PLO: 4 },
  { func: 'F12', PLOJF: 1, PLOJ: 2, PLOF: 3, PLO: 4 },
  { func: 'F13', PLOJF: 3, PLOJ: 1, PLOF: 2, PLO: 4 },
  { func: 'F14', PLOJF: 2, PLOJ: 1, PLOF: 3, PLO: 4 },
  { func: 'F15', PLOJF: 2, PLOJ: 4, PLOF: 3, PLO: 1 },
  { func: 'F16', PLOJF: 2, PLOJ: 1, PLOF: 3, PLO: 4 },
  { func: 'F17', PLOJF: 2, PLOJ: 3, PLOF: 1, PLO: 4 },
  { func: 'F18', PLOJF: 1, PLOJ: 2, PLOF: 3, PLO: 4 },
  { func: 'F19', PLOJF: 3, PLOJ: 2, PLOF: 1, PLO: 4 },
  { func: 'F20', PLOJF: 1, PLOJ: 4, PLOF: 2, PLO: 3 },
  { func: 'F21', PLOJF: 2, PLOJ: 4, PLOF: 1, PLO: 3 },
  { func: 'F22', PLOJF: 2, PLOJ: 1, PLOF: 3, PLO: 4 },
  { func: 'F23', PLOJF: 3, PLOJ: 2, PLOF: 1, PLO: 4 },
  { func: 'F24', PLOJF: 3, PLOJ: 2, PLOF: 4, PLO: 1 },
  { func: 'F25', PLOJF: 4, PLOJ: 2, PLOF: 3, PLO: 1 },
  { func: 'F26', PLOJF: 2, PLOJ: 3, PLOF: 4, PLO: 1 },
  { func: 'F27', PLOJF: 1, PLOJ: 3, PLOF: 2, PLO: 4 },
  { func: 'F28', PLOJF: 2, PLOJ: 3, PLOF: 4, PLO: 1 },
  { func: 'F29', PLOJF: 3, PLOJ: 1, PLOF: 2, PLO: 4 }
];

// const data = [
//   { func: 'F1', PLOJF: 6, ETO: 10, MGO: 8, SBO: 3, MSAO: 7, CLPSO: 5, LSHADE: 1, RMRIME: 4, MDPLO: 2, PSMADE: 9 },
//   { func: 'F2', PLOJF: 5, ETO: 10, MGO: 6, SBO: 2, MSAO: 3, CLPSO: 7, LSHADE: 8, RMRIME: 4, MDPLO: 1, PSMADE: 9 },
//   { func: 'F3', PLOJF: 8, ETO: 10, MGO: 7, SBO: 3, MSAO: 6, CLPSO: 4, LSHADE: 2, RMRIME: 5, MDPLO: 1, PSMADE: 9 },
//   { func: 'F4', PLOJF: 3, ETO: 10, MGO: 5, SBO: 8, MSAO: 2, CLPSO: 4, LSHADE: 1, RMRIME: 6, MDPLO: 7, PSMADE: 9 },
//   { func: 'F5', PLOJF: 3, ETO: 10, MGO: 2, SBO: 9, MSAO: 1, CLPSO: 7, LSHADE: 6, RMRIME: 5, MDPLO: 4, PSMADE: 8 },
//   { func: 'F6', PLOJF: 1, ETO: 10, MGO: 3, SBO: 8, MSAO: 2, CLPSO: 4, LSHADE: 5, RMRIME: 6, MDPLO: 7, PSMADE: 9 },
//   { func: 'F7', PLOJF: 3, ETO: 9, MGO: 6, SBO: 8, MSAO: 4, CLPSO: 2, LSHADE: 1, RMRIME: 5, MDPLO: 7, PSMADE: 10 },
//   { func: 'F8', PLOJF: 4, ETO: 10, MGO: 2, SBO: 9, MSAO: 1, CLPSO: 3, LSHADE: 8, RMRIME: 5, MDPLO: 6, PSMADE: 7 },
//   { func: 'F9', PLOJF: 4, ETO: 10, MGO: 6, SBO: 7, MSAO: 2, CLPSO: 3, LSHADE: 1, RMRIME: 5, MDPLO: 8, PSMADE: 9 },
//   { func: 'F10', PLOJF: 2, ETO: 10, MGO: 4, SBO: 7, MSAO: 1, CLPSO: 3, LSHADE: 8, RMRIME: 5, MDPLO: 6, PSMADE: 9 },
//   { func: 'F11', PLOJF: 3, ETO: 10, MGO: 9, SBO: 4, MSAO: 6, CLPSO: 8, LSHADE: 1, RMRIME: 5, MDPLO: 2, PSMADE: 7 },
//   { func: 'F12', PLOJF: 4, ETO: 10, MGO: 8, SBO: 6, MSAO: 5, CLPSO: 1, LSHADE: 2, RMRIME: 7, MDPLO: 3, PSMADE: 9 },
//   { func: 'F13', PLOJF: 3, ETO: 10, MGO: 8, SBO: 6, MSAO: 7, CLPSO: 9, LSHADE: 1, RMRIME: 4, MDPLO: 2, PSMADE: 5 },
//   { func: 'F14', PLOJF: 4, ETO: 10, MGO: 8, SBO: 5, MSAO: 7, CLPSO: 1, LSHADE: 2, RMRIME: 6, MDPLO: 3, PSMADE: 9 },
//   { func: 'F15', PLOJF: 2, ETO: 10, MGO: 5, SBO: 8, MSAO: 1, CLPSO: 4, LSHADE: 6, RMRIME: 3, MDPLO: 7, PSMADE: 9 },
//   { func: 'F16', PLOJF: 1, ETO: 10, MGO: 3, SBO: 9, MSAO: 6, CLPSO: 2, LSHADE: 5, RMRIME: 4, MDPLO: 7, PSMADE: 8 },
//   { func: 'F17', PLOJF: 3, ETO: 10, MGO: 8, SBO: 4, MSAO: 6, CLPSO: 7, LSHADE: 1, RMRIME: 5, MDPLO: 2, PSMADE: 9 },
//   { func: 'F18', PLOJF: 2, ETO: 10, MGO: 7, SBO: 5, MSAO: 8, CLPSO: 1, LSHADE: 3, RMRIME: 6, MDPLO: 4, PSMADE: 9 },
//   { func: 'F19', PLOJF: 1, ETO: 10, MGO: 4, SBO: 8, MSAO: 7, CLPSO: 3, LSHADE: 6, RMRIME: 2, MDPLO: 5, PSMADE: 9 },
//   { func: 'F20', PLOJF: 2, ETO: 10, MGO: 5, SBO: 8, MSAO: 4, CLPSO: 3, LSHADE: 1, RMRIME: 6, MDPLO: 7, PSMADE: 9 },
//   { func: 'F21', PLOJF: 1, ETO: 10, MGO: 6, SBO: 2, MSAO: 7, CLPSO: 3, LSHADE: 4, RMRIME: 5, MDPLO: 8, PSMADE: 9 },
//   { func: 'F22', PLOJF: 2, ETO: 10, MGO: 6, SBO: 8, MSAO: 1, CLPSO: 4, LSHADE: 3, RMRIME: 7, MDPLO: 5, PSMADE: 9 },
//   { func: 'F23', PLOJF: 1, ETO: 10, MGO: 6, SBO: 8, MSAO: 3, CLPSO: 5, LSHADE: 2, RMRIME: 7, MDPLO: 4, PSMADE: 9 },
//   { func: 'F24', PLOJF: 4, ETO: 10, MGO: 3, SBO: 8, MSAO: 2, CLPSO: 1, LSHADE: 6, RMRIME: 9, MDPLO: 5, PSMADE: 7 },
//   { func: 'F25', PLOJF: 4, ETO: 10, MGO: 6, SBO: 5, MSAO: 3, CLPSO: 1, LSHADE: 7, RMRIME: 8, MDPLO: 2, PSMADE: 9 },
//   { func: 'F26', PLOJF: 3, ETO: 10, MGO: 4, SBO: 9, MSAO: 2, CLPSO: 5, LSHADE: 8, RMRIME: 6, MDPLO: 7, PSMADE: 1 },
//   { func: 'F27', PLOJF: 5, ETO: 10, MGO: 8, SBO: 3, MSAO: 4, CLPSO: 6, LSHADE: 2, RMRIME: 7, MDPLO: 1, PSMADE: 9 },
//   { func: 'F28', PLOJF: 5, ETO: 10, MGO: 7, SBO: 9, MSAO: 1, CLPSO: 2, LSHADE: 3, RMRIME: 4, MDPLO: 6, PSMADE: 8 },
//   { func: 'F29', PLOJF: 8, ETO: 10, MGO: 9, SBO: 6, MSAO: 4, CLPSO: 5, LSHADE: 1, RMRIME: 3, MDPLO: 2, PSMADE: 7 }
// ];
  const optimizers = Object.keys(data[0]).filter(k => k !== 'func');

  // MODIFY COLOR ASSIGNMENT to use the rank interpolator for optimizer colors
  const colorScale = d3.scaleSequential()
    .domain([0, optimizers.length - 1])
    .interpolator(colorSchemes[colorScheme].rank);

  const colors = {};
  optimizers.forEach((opt, i) => {
    colors[opt] = colorScale(i);
  });

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 800;
    const height = 900;
    const margin = 120;
    const radius = Math.min(width, height - 200) / 2 - margin;

    svg.attr('width', width).attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${width/2}, ${height/2})`);

    const angleScale = d3.scaleLinear()
      .domain([0, data.length])
      .range([0, 2 * Math.PI]);

    const radiusScale = d3.scaleLinear()
      .domain([1, 10])
      .range([radius, radius * 0.2]);

    const levels = [1, 3, 5, 7, 10];
    levels.forEach(level => {
      g.append('circle')
        .attr('r', radiusScale(level))
        .attr('fill', 'none')
        .attr('stroke', '#e0e0e0')
        .attr('stroke-width', 1)
        .attr('opacity', 0.5);
    });

    data.forEach((d, i) => {
      const angle = angleScale(i) - Math.PI / 2;
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;
      
      g.append('line')
        .attr('x1', 0)
        .attr('y1', 0)
        .attr('x2', x)
        .attr('y2', y)
        .attr('stroke', '#e0e0e0')
        .attr('stroke-width', 1)
        .attr('opacity', 0.5);
      
      const labelRadius = radius + 20;
      const labelX = Math.cos(angle) * labelRadius;
      const labelY = Math.sin(angle) * labelRadius;
      
      g.append('text')
        .attr('x', labelX)
        .attr('y', labelY)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '11px')
        .attr('font-weight', 'bold')
        .attr('font-family', 'Times New Roman, serif') // Add Times New Roman
        .attr('fill', '#333')
        .text(d.func);
    });

    optimizers.forEach(optimizer => {
      if (!selectedOptimizers[optimizer]) return;

      const points = data.map((d, i) => {
        const angle = angleScale(i) - Math.PI / 2;
        const r = radiusScale(d[optimizer]);
        return [Math.cos(angle) * r, Math.sin(angle) * r];
      });

      points.push(points[0]);

      const line = d3.line()
        .x(d => d[0])
        .y(d => d[1])
        .curve(d3.curveCardinalClosed.tension(0.5));

      g.append('path')
        .datum(points)
        .attr('d', line)
        .attr('fill', colors[optimizer])
        .attr('fill-opacity', 0.1)
        .attr('stroke', colors[optimizer])
        .attr('stroke-width', 2.5)
        .attr('stroke-opacity', 0.8);

      data.forEach((d, i) => {
        const angle = angleScale(i) - Math.PI / 2;
        const r = radiusScale(d[optimizer]);
        const x = Math.cos(angle) * r;
        const y = Math.sin(angle) * r;
        
        g.append('circle')
          .attr('cx', x)
          .attr('cy', y)
          .attr('r', 4)
          .attr('fill', colors[optimizer])
          .attr('stroke', 'white')
          .attr('stroke-width', 2)
          .style('cursor', 'pointer')
          .on('mouseover', function(event) {
            const tooltip = d3.select('body').append('div')
              .attr('class', 'tooltip')
              .style('position', 'absolute')
              .style('background', 'rgba(0,0,0,0.8)')
              .style('color', 'white')
              .style('padding', '8px')
              .style('border-radius', '4px')
              .style('font-size', '12px')
              .style('font-family', 'Times New Roman, serif') // Add Times New Roman
              .style('pointer-events', 'none')
              .style('z-index', 1000);
            
            tooltip.html(`${optimizer}<br/>${d.func}: Rank ${d[optimizer]}`)
              .style('left', (event.pageX + 10) + 'px')
              .style('top', (event.pageY - 10) + 'px');
          })
          .on('mouseout', function() {
            d3.selectAll('.tooltip').remove();
          });
      });
    });

    g.append('text')
      .attr('x', 0)
      .attr('y', -(radius + 80))
      .attr('text-anchor', 'middle')
      .attr('font-size', '24px')
      .attr('font-weight', 'bold')
      .attr('font-family', 'Times New Roman, serif') // Add Times New Roman
      .attr('fill', '#333')
      .text('Optimizer Performance Radar Chart');

    g.append('text')
      .attr('x', 0)
      .attr('y', -(radius + 55))
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-family', 'Times New Roman, serif') // Add Times New Roman
      .attr('fill', '#666')
    //  .text('Better performance (lower rank) appears further from center');

    levels.forEach(level => {
      g.append('text')
        .attr('x', 5)
        .attr('y', -radiusScale(level))
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .attr('font-family', 'Times New Roman, serif') // Add Times New Roman
        .attr('fill', '#333')
        .attr('stroke', 'white')
        .attr('stroke-width', 2)
        .attr('paint-order', 'stroke')
        .text(`Rank ${level}`);
    });

    // Add legend
    const legendRows = 2;
    const legendPerRow = 5;
    const legendGroups = [];
    for (let i = 0; i < legendRows; i++) {
      legendGroups.push(optimizers.slice(i * legendPerRow, (i + 1) * legendPerRow));
    }
    const legendYStart = radius + 50;
    const legendXSpacing = 120;
    const legendYSpacing = 32;
    const legend = g.append('g')
      .attr('transform', `translate(0, ${legendYStart})`);

    legendGroups.forEach((group, rowIdx) => {
      const rowWidth = group.length * legendXSpacing;
      const rowXStart = -rowWidth / 1.85 + legendXSpacing / 2;
      group.forEach((optimizer, i) => {
        const legendItem = legend.append('g')
          .attr('transform', `translate(${rowXStart + i * legendXSpacing}, ${rowIdx * legendYSpacing})`);

        legendItem.append('circle')
          .attr('cx', 0)
          .attr('cy', 0)
          .attr('r', 6)
          .attr('fill', colors[optimizer])
          .attr('stroke', 'white')
          .attr('stroke-width', 2);

        legendItem.append('text')
          .attr('x', 15)
          .attr('y', 0)
          .attr('dominant-baseline', 'middle')
          .attr('font-size', '14px')
          .attr('font-weight', 'bold')
          .attr('font-family', 'Times New Roman, serif') // Add Times New Roman
          .attr('fill', '#333')
          .text(optimizer);
      });
    });

  }, [selectedOptimizers, data, colors, optimizers, colorScheme]); // Add colorScheme to dependencies
  
  const toggleOptimizer = (optimizer) => {
    setSelectedOptimizers(prev => ({
      ...prev,
      [optimizer]: !prev[optimizer]
    }));
  };

  return (
    <div className="flex flex-col items-center p-6 bg-gradient-to-br from-slate-50 to-blue-50 min-h-screen" style={{fontFamily: 'Times New Roman, serif'}}>
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 text-center">
          Optimizer Performance Analysis
        </h2>
        
        <div className="mb-4">
          <h3 className="text-sm font-medium text-gray-700 mb-2 text-center">Color Schemes:</h3>
          <div className="flex flex-wrap gap-2 justify-center">
            {Object.entries(colorSchemes).map(([key, scheme]) => (
              <button
                key={key}
                onClick={() => setColorScheme(key)}
                className={`px-3 py-1 text-xs rounded-md font-medium transition-all duration-200 ${
                  colorScheme === key
                    ? 'bg-indigo-500 text-white shadow-md transform scale-105'
                    : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                }`}
              >
                {scheme.name}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap gap-3 justify-center">
          {optimizers.map(optimizer => (
            <button
              key={optimizer}
              onClick={() => toggleOptimizer(optimizer)}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                selectedOptimizers[optimizer]
                  ? 'shadow-lg transform scale-105'
                  : 'opacity-50 hover:opacity-75'
              }`}
              style={{
                backgroundColor: selectedOptimizers[optimizer] ? colors[optimizer] : '#e5e7eb',
                color: selectedOptimizers[optimizer] ? 'white' : '#6b7280'
              }}
            >
              {optimizer}
            </button>
          ))}
        </div>
        <button
          onClick={downloadPNG}
          className="mt-4 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium rounded-lg shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
        >
          Download as PNG
        </button>
      </div>
      
      <div className="bg-white rounded-xl shadow-xl p-6">
        <svg ref={svgRef}></svg>
      </div>

      <div className="mt-6 max-w-4xl text-center">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <div className="bg-white rounded-lg p-4 shadow-md">
            <h3 className="font-bold text-gray-800 mb-2">How to Read</h3>
            <p className="text-sm text-gray-600">
              Each optimizer is represented by a colored line. Points closer to the center indicate worse performance (higher rank), 
              while points further from center indicate better performance (lower rank).
            </p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-md">
            <h3 className="font-bold text-gray-800 mb-2">Interactive Features</h3>
            <p className="text-sm text-gray-600">
              Click optimizer buttons to show/hide them. Choose different color schemes from the buttons above.
              Hover over data points to see detailed rank information for each function-optimizer combination.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OptimizerRadarPlot;