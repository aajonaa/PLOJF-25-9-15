import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

const FriedmanRankChart = () => {
  const svgRef = useRef();
  const [sortBy, setSortBy] = useState('rank'); // 'rank' or 'arv'
  const [showValues, setShowValues] = useState(true);
  const [showLegend, setShowLegend] = useState(true);
  const [colorScheme, setColorScheme] = useState('default'); // Add this line

  // Define color scheme options - REPLACE THE PASTEL SCHEME
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
    // REPLACE PASTEL WITH VALID INTERPOLATORS
    pastel: {
      name: 'Pastel',
      rank: d3.interpolateRgb("#FFB6C1", "#87CEEB"), // Light pink to sky blue
      arv: d3.interpolateRgb("#98FB98", "#DDA0DD")   // Pale green to plum
    },
    // ADD MORE VALID COLOR SCHEMES
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

// const data = [
//   { optimizer: 'PLOJF', arv: 3.3448, rank: 1 },
//   { optimizer: 'ETO', arv: 9.9655, rank: 10 },
//   { optimizer: 'MGO', arv: 5.8276, rank: 7 },
//   { optimizer: 'SBO', arv: 6.3793, rank: 8 },
//   { optimizer: 'MSAO', arv: 3.8966, rank: 3 },
//   { optimizer: 'CLPSO', arv: 3.8966, rank: 4 },
//   { optimizer: 'LSHADE', arv: 3.6207, rank: 2 },
//   { optimizer: 'RMRIME', arv: 5.3103, rank: 6 },
//   { optimizer: 'MDPLO', arv: 4.5172, rank: 5 },
//   { optimizer: 'PSMADE', arv: 8.2414, rank: 9 }
// ];

const data = [
  { optimizer: 'PLOJF', arv: 2.2069, rank: 1 },
  { optimizer: 'PLOJ', arv: 2.3793, rank: 3 },
  { optimizer: 'PLOF', arv: 2.2414, rank: 2 },
  { optimizer: 'PLO', arv: 3.1724, rank: 4 }
];

  const downloadPNG = () => {
    const svg = svgRef.current;
    if (!svg) return;

    const scaleFactor = 4;
    const originalWidth = 1000;
    const originalHeight = 700;
    const highResWidth = originalWidth * scaleFactor;
    const highResHeight = originalHeight * scaleFactor;
    
    const svgClone = svg.cloneNode(true);
    svgClone.setAttribute('width', highResWidth);
    svgClone.setAttribute('height', highResHeight);
    
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
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
        
        canvas.toBlob((blob) => {
          if (blob) {
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'friedman-rank-chart-300dpi.png';
            link.style.display = 'none';
            
            document.body.appendChild(link);
            link.click();
            
            setTimeout(() => {
              document.body.removeChild(link);
              URL.revokeObjectURL(url);
            }, 100);
            
            setTimeout(() => {
            // Add this comment to disable the ESlint rule for the next line
            // eslint-disable-next-line no-restricted-globals
            if (confirm('If download didn\'t start, click OK to open image in new tab (then right-click to save)')) {                window.open(url, '_blank');
                window.open(url, '_blank');
              }
            }, 1000);
          }
        }, 'image/png', 1.0);
        
      } catch (error) {
        console.error('Canvas drawing failed:', error);
        alert('Download failed. Please try again.');
      }
    };
    
    img.onerror = (error) => {
      console.error('Image loading failed:', error);
      alert('Failed to generate image. Please try again.');
    };
    
    const url = URL.createObjectURL(svgBlob);
    img.src = url;
  };

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 1000;
    const height = 700;
    const margin = { top: 80, right: 60, bottom: 80, left: 120 };

    svg.attr('width', width).attr('height', height);

    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Sort data
    const sortedData = [...data].sort((a, b) => {
      if (sortBy === 'rank') {
        return a.rank - b.rank;
      } else {
        return a.arv - b.arv;
      }
    });

    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => Math.max(d.arv, d.rank * 1.2))])
      .range([0, chartWidth]);

    const yScale = d3.scaleBand()
      .domain(sortedData.map(d => d.optimizer))
      .range([0, chartHeight])
      .padding(0.2);

    // Create color scales using selected scheme
    const rankColorScale = d3.scaleSequential()
      .domain([1, 10])
      .interpolator(colorSchemes[colorScheme].rank);

    const arvColorScale = d3.scaleSequential()
      .domain([d3.min(data, d => d.arv), d3.max(data, d => d.arv)])
      .interpolator(colorSchemes[colorScheme].arv);

    // Add background grid
    const xTicks = xScale.ticks(8);
    g.selectAll('.grid-line')
      .data(xTicks)
      .enter()
      .append('line')
      .attr('class', 'grid-line')
      .attr('x1', d => xScale(d))
      .attr('x2', d => xScale(d))
      .attr('y1', 0)
      .attr('y2', chartHeight)
      .attr('stroke', '#f0f0f0')
      .attr('stroke-width', 1);

    // Add ARV bars
    const arvBars = g.selectAll('.arv-bar')
      .data(sortedData)
      .enter()
      .append('rect')
      .attr('class', 'arv-bar')
      .attr('x', 0)
      .attr('y', d => yScale(d.optimizer))
      .attr('width', d => xScale(d.arv))
      .attr('height', yScale.bandwidth() * 0.35)
      .attr('fill', d => arvColorScale(d.arv))
      .attr('opacity', 0.8)
      .attr('rx', 4);

    // Add Rank bars (offset)
    const rankBars = g.selectAll('.rank-bar')
      .data(sortedData)
      .enter()
      .append('rect')
      .attr('class', 'rank-bar')
      .attr('x', 0)
      .attr('y', d => yScale(d.optimizer) + yScale.bandwidth() * 0.4)
      .attr('width', d => xScale(d.rank))
      .attr('height', yScale.bandwidth() * 0.35)
      .attr('fill', d => rankColorScale(d.rank))
      .attr('opacity', 0.8)
      .attr('rx', 4);

    // Add value labels if enabled
    if (showValues) {
      // ARV labels
      g.selectAll('.arv-label')
        .data(sortedData)
        .enter()
        .append('text')
        .attr('class', 'arv-label')
        .attr('x', d => xScale(d.arv) + 5)
        .attr('y', d => yScale(d.optimizer) + yScale.bandwidth() * 0.175)
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .attr('font-family', 'Times New Roman, serif') // Add this line
        .attr('fill', '#333')
        .text(d => `ARV: ${d.arv.toFixed(4)}`);

      // Rank labels
      g.selectAll('.rank-label')
        .data(sortedData)
        .enter()
        .append('text')
        .attr('class', 'rank-label')
        .attr('x', d => xScale(d.rank) + 5)
        .attr('y', d => yScale(d.optimizer) + yScale.bandwidth() * 0.575)
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .attr('font-family', 'Times New Roman, serif') // Add this line
        .attr('fill', '#333')
        .text(d => `Rank: ${d.rank}`);
    }

    // Add Y-axis (optimizer names)
    g.append('g')
      .selectAll('.y-label')
      .data(sortedData)
      .enter()
      .append('text')
      .attr('class', 'y-label')
      .attr('x', -10)
      .attr('y', d => yScale(d.optimizer) + yScale.bandwidth() / 2)
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'middle')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .attr('font-family', 'Times New Roman, serif') // Add this line
      .attr('fill', '#333')
      .text(d => d.optimizer);

    // Add X-axis
    const xAxis = d3.axisBottom(xScale)
      .ticks(8)
      .tickFormat(d3.format('.1f'));

    g.append('g')
      .attr('transform', `translate(0, ${chartHeight})`)
      .call(xAxis)
      .selectAll('text')
      .attr('font-size', '12px')
      .attr('font-family', 'Times New Roman, serif') // Add this line
      .attr('fill', '#666');

    // Add X-axis label
    g.append('text')
      .attr('x', chartWidth / 2)
      .attr('y', chartHeight + 50)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .attr('font-family', 'Times New Roman, serif') // Add this line
      .attr('fill', '#333')
      .text('Value');

    // Add title
    g.append('text')
      .attr('x', chartWidth / 2)
      .attr('y', -40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '24px')
      .attr('font-weight', 'bold')
      .attr('font-family', 'Times New Roman, serif') // Add this line
      .attr('fill', '#333')
      .text('Friedman Rank Analysis');

    g.append('text')
      .attr('x', chartWidth / 2)
      .attr('y', -15)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-family', 'Times New Roman, serif') // Add this line
      .attr('fill', '#666')
    //  .text(`Sorted by ${sortBy === 'rank' ? 'Rank' : 'ARV'} • Lower values indicate better performance`);

    // Add legend (wrap in condition)
    if (showLegend) {
      const legend = g.append('g')
        .attr('transform', `translate(${chartWidth - 200}, 20)`);

      // ARV legend
      legend.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', 20)
        .attr('height', 12)
        .attr('fill', arvColorScale(d3.mean(data, d => d.arv)))
        .attr('rx', 2);

      legend.append('text')
        .attr('x', 25)
        .attr('y', 6)
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .attr('font-family', 'Times New Roman, serif') // Add this line
        .attr('fill', '#333')
        .text('Average Rank Value (ARV)');

      // Rank legend
      legend.append('rect')
        .attr('x', 0)
        .attr('y', 20)
        .attr('width', 20)
        .attr('height', 12)
        .attr('fill', rankColorScale(5))
        .attr('rx', 2);

      legend.append('text')
        .attr('x', 25)
        .attr('y', 26)
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .attr('font-family', 'Times New Roman, serif') // Add this line
        .attr('fill', '#333')
        .text('Friedman Rank');
    }

    // Add hover effects
    arvBars.on('mouseover', function(event, d) {
      d3.select(this).attr('opacity', 1);
      
      const tooltip = d3.select('body').append('div')
        .attr('class', 'tooltip')
        .style('position', 'absolute')
        .style('background', 'rgba(0,0,0,0.8)')
        .style('color', 'white')
        .style('padding', '10px')
        .style('border-radius', '5px')
        .style('font-size', '12px')
        .style('pointer-events', 'none')
        .style('z-index', 1000);
      
      tooltip.html(`<strong>${d.optimizer}</strong><br/>ARV: ${d.arv.toFixed(4)}<br/>Rank: ${d.rank}`)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px');
    })
    .on('mouseout', function() {
      d3.select(this).attr('opacity', 0.8);
      d3.selectAll('.tooltip').remove();
    });

    rankBars.on('mouseover', function(event, d) {
      d3.select(this).attr('opacity', 1);
      
      const tooltip = d3.select('body').append('div')
        .attr('class', 'tooltip')
        .style('position', 'absolute')
        .style('background', 'rgba(0,0,0,0.8)')
        .style('color', 'white')
        .style('padding', '10px')
        .style('border-radius', '5px')
        .style('font-size', '12px')
        .style('pointer-events', 'none')
        .style('z-index', 1000);
      
      tooltip.html(`<strong>${d.optimizer}</strong><br/>ARV: ${d.arv.toFixed(4)}<br/>Rank: ${d.rank}`)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px');
    })
    .on('mouseout', function() {
      d3.select(this).attr('opacity', 0.8);
      d3.selectAll('.tooltip').remove();
    });

  }, [sortBy, showValues, showLegend, colorScheme]); // Add colorScheme to dependencies

  return (
    <div className="flex flex-col items-center p-6 bg-gradient-to-br from-slate-50 to-indigo-50 min-h-screen">
      <div className="mb-6">
        <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">
          Friedman Rank Visualization
        </h2>
        
        <div className="flex flex-wrap gap-4 justify-center items-center mb-4">
          <div className="flex gap-2">
            <button
              onClick={() => setSortBy('rank')}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                sortBy === 'rank'
                  ? 'bg-blue-500 text-white shadow-lg transform scale-105'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              Sort by Rank
            </button>
            <button
              onClick={() => setSortBy('arv')}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                sortBy === 'arv'
                  ? 'bg-blue-500 text-white shadow-lg transform scale-105'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              Sort by ARV
            </button>
          </div>
          
          <button
            onClick={() => setShowValues(!showValues)}
            className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
              showValues
                ? 'bg-green-500 text-white shadow-lg'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            {showValues ? 'Hide Values' : 'Show Values'}
          </button>
          
          <button
            onClick={() => setShowLegend(!showLegend)}
            className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
              showLegend
                ? 'bg-orange-500 text-white shadow-lg'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            {showLegend ? 'Hide Legend' : 'Show Legend'}
          </button>
          
          <button
            onClick={downloadPNG}
            className="px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-600 text-white font-medium rounded-lg shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
          >
            Download PNG
          </button>
        </div>

        {/* Color Scheme Selection */}
        <div className="flex flex-wrap gap-2 justify-center items-center">
          <span className="text-sm font-medium text-gray-700 mr-2">Color Schemes:</span>
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
      
      <div className="bg-white rounded-xl shadow-2xl p-6">
        <svg ref={svgRef}></svg>
      </div>

      <div className="mt-6 max-w-4xl">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-lg p-4 shadow-md">
            <h3 className="font-bold text-gray-800 mb-2">Understanding the Chart</h3>
            <p className="text-sm text-gray-600">
              Each optimizer shows two bars: ARV (Average Rank Value) and Friedman Rank. 
              Lower values indicate better performance.
            </p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-md">
            <h3 className="font-bold text-gray-800 mb-2">Color Coding</h3>
            <p className="text-sm text-gray-600">
              ARV bars and Rank bars use different color scales. 
              You can change the color scheme using the buttons above.
            </p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-md">
            <h3 className="font-bold text-gray-800 mb-2">Interactive Features</h3>
            <p className="text-sm text-gray-600">
              Hover over bars for detailed information. Toggle sorting options and value display. 
              Download as high-resolution PNG.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FriedmanRankChart;