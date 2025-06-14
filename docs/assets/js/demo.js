// Demo-specific JavaScript functionality

let scalingChart = null;
let comparisonChart = null;
let scalingComparisonChart = null;
let currentNetworkSize = 16;
let currentLayerSize = 100;
let currentCheckpoints = [];

document.addEventListener('DOMContentLoaded', function() {
    initializeMemoryVisualizer();
    initializeCheckpointExplorer();
    initializeModelCalculator();
    initializeScalingComparison();
});

// Demo 1: Memory Usage Visualizer
function initializeMemoryVisualizer() {
    const networkSizeSlider = document.getElementById('network-size');
    const layerSizeSlider = document.getElementById('layer-size');
    const networkSizeDisplay = document.getElementById('network-size-display');
    const layerSizeDisplay = document.getElementById('layer-size-display');
    
    if (!networkSizeSlider) return;
    
    function updateMemoryVisualization() {
        const numLayers = parseInt(networkSizeSlider.value);
        const layerSize = parseInt(layerSizeSlider.value);
        
        currentNetworkSize = numLayers;
        currentLayerSize = layerSize;
        
        networkSizeDisplay.textContent = numLayers;
        layerSizeDisplay.textContent = layerSize;
        
        // Calculate memory requirements
        const totalParams = calculateTotalParams(numLayers, layerSize);
        const standardMemoryMB = (totalParams * 4) / (1024 * 1024); // 4 bytes per param, convert to MB
        
        // Calculate optimal checkpoints
        const optimalCheckpoints = getOptimalCheckpoints(numLayers);
        const checkpointParams = calculateCheckpointParams(numLayers, layerSize, optimalCheckpoints);
        const checkpointMemoryMB = (checkpointParams * 4) / (1024 * 1024);
        const reduction = standardMemoryMB / checkpointMemoryMB;
        
        // Update displays
        document.getElementById('standard-memory').textContent = standardMemoryMB.toFixed(1);
        document.getElementById('checkpoint-memory').textContent = checkpointMemoryMB.toFixed(1);
        document.getElementById('memory-reduction').textContent = reduction.toFixed(1);
        document.getElementById('checkpoint-count').textContent = optimalCheckpoints.length;
        
        // Update chart
        updateScalingChart();
    }
    
    networkSizeSlider.addEventListener('input', updateMemoryVisualization);
    layerSizeSlider.addEventListener('input', updateMemoryVisualization);
    
    updateMemoryVisualization();
}

function updateScalingChart() {
    const ctx = document.getElementById('scaling-chart');
    if (!ctx) return;
    
    // Generate data for different network sizes
    const sizes = [];
    const standardMem = [];
    const checkpointMem = [];
    
    for (let layers = 3; layers <= Math.max(currentNetworkSize + 10, 30); layers += 2) {
        const totalParams = calculateTotalParams(layers, currentLayerSize);
        const standardMemoryMB = (totalParams * 4) / (1024 * 1024);
        
        const optimalCheckpoints = getOptimalCheckpoints(layers);
        const checkpointParams = calculateCheckpointParams(layers, currentLayerSize, optimalCheckpoints);
        const checkpointMemoryMB = (checkpointParams * 4) / (1024 * 1024);
        
        sizes.push(layers);
        standardMem.push(standardMemoryMB);
        checkpointMem.push(checkpointMemoryMB);
    }
    
    if (scalingChart) {
        scalingChart.destroy();
    }
    
    scalingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: sizes,
            datasets: [{
                label: 'Standard O(n)',
                data: standardMem,
                borderColor: 'rgba(220, 38, 38, 1)',
                backgroundColor: 'rgba(220, 38, 38, 0.1)',
                borderWidth: 3,
                tension: 0.1
            }, {
                label: 'Checkpoint O(√n)',
                data: checkpointMem,
                borderColor: 'rgba(16, 185, 129, 1)',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 3,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Memory Usage Scaling Comparison'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Number of Layers'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Memory Usage (MB)'
                    }
                }
            },
            elements: {
                point: {
                    radius: function(context) {
                        return context.parsed.x === currentNetworkSize ? 8 : 3;
                    },
                    backgroundColor: function(context) {
                        return context.parsed.x === currentNetworkSize ? '#f59e0b' : context.dataset.borderColor;
                    }
                }
            }
        }
    });
}

// Demo 2: Checkpoint Strategy Explorer
function initializeCheckpointExplorer() {
    const addCheckpointBtn = document.getElementById('add-checkpoint');
    const removeCheckpointBtn = document.getElementById('remove-checkpoint');
    const optimalStrategyBtn = document.getElementById('optimal-strategy');
    const clearCheckpointsBtn = document.getElementById('clear-checkpoints');
    
    if (!addCheckpointBtn) return;
    
    currentCheckpoints = [0]; // Start with input layer
    
    addCheckpointBtn.addEventListener('click', () => {
        const nextLayer = Math.min(currentCheckpoints[currentCheckpoints.length - 1] + 2, 15);
        if (nextLayer < 16 && !currentCheckpoints.includes(nextLayer)) {
            currentCheckpoints.push(nextLayer);
            currentCheckpoints.sort((a, b) => a - b);
            updateCheckpointStrategy();
        }
    });
    
    removeCheckpointBtn.addEventListener('click', () => {
        if (currentCheckpoints.length > 1) { // Keep at least input layer
            currentCheckpoints.pop();
            updateCheckpointStrategy();
        }
    });
    
    optimalStrategyBtn.addEventListener('click', () => {
        currentCheckpoints = getOptimalCheckpoints(16);
        updateCheckpointStrategy();
    });
    
    clearCheckpointsBtn.addEventListener('click', () => {
        currentCheckpoints = [0];
        updateCheckpointStrategy();
    });
    
    updateCheckpointStrategy();
}

function updateCheckpointStrategy() {
    const networkLayers = 16;
    const layerSize = 100;
    
    // Update network diagram
    drawNetworkDiagram();
    
    // Calculate metrics
    const totalParams = calculateTotalParams(networkLayers, layerSize);
    const checkpointParams = calculateCheckpointParams(networkLayers, layerSize, currentCheckpoints);
    const memoryMB = (checkpointParams * 4) / (1024 * 1024);
    const reduction = totalParams / checkpointParams;
    const recomputation = calculateRecomputationCost(networkLayers, currentCheckpoints);
    
    // Update displays
    document.getElementById('strategy-checkpoints').textContent = currentCheckpoints.length;
    document.getElementById('strategy-memory').textContent = memoryMB.toFixed(1) + ' MB';
    document.getElementById('strategy-reduction').textContent = reduction.toFixed(1) + '×';
    document.getElementById('strategy-recomp').textContent = recomputation;
    
    // Update checkpoint list
    const checkpointList = document.getElementById('checkpoint-list');
    checkpointList.innerHTML = currentCheckpoints.map(idx => 
        `<span class="status-badge" style="margin: 0.25rem;">Layer ${idx}</span>`
    ).join('');
}

function drawNetworkDiagram() {
    const container = document.getElementById('network-diagram');
    if (!container) return;
    
    const layers = 16;
    const layerHeight = 20;
    const layerSpacing = 15;
    const totalHeight = layers * (layerHeight + layerSpacing);
    
    let html = '<svg width="100%" height="320px" viewBox="0 0 200 320">';
    
    // Draw layers
    for (let i = 0; i < layers; i++) {
        const y = 10 + i * (layerHeight + layerSpacing);
        const isCheckpoint = currentCheckpoints.includes(i);
        const isInput = i === 0;
        const isOutput = i === layers - 1;
        
        let color = '#e5e7eb'; // Default gray
        if (isInput) color = '#3b82f6'; // Blue for input
        else if (isOutput) color = '#ef4444'; // Red for output
        else if (isCheckpoint) color = '#10b981'; // Green for checkpoints
        
        html += `<rect x="20" y="${y}" width="160" height="${layerHeight}" 
                 fill="${color}" stroke="#374151" stroke-width="1" rx="4"/>`;
        
        // Layer label
        let label = `Layer ${i}`;
        if (isInput) label += ' (Input)';
        else if (isOutput) label += ' (Output)';
        else if (isCheckpoint) label += ' (Checkpoint)';
        
        html += `<text x="100" y="${y + layerHeight/2 + 4}" text-anchor="middle" 
                 fill="white" font-size="10" font-family="Inter">${label}</text>`;
    }
    
    html += '</svg>';
    container.innerHTML = html;
}

// Demo 3: Real-World Model Calculator
function initializeModelCalculator() {
    const modelRadios = document.querySelectorAll('input[name="model"]');
    const customInputs = document.getElementById('custom-inputs');
    const customParams = document.getElementById('custom-params');
    const availableMemory = document.getElementById('available-memory');
    
    if (!modelRadios.length) return;
    
    function updateModelCalculator() {
        const selectedModel = document.querySelector('input[name="model"]:checked');
        if (!selectedModel) return;
        
        let params;
        if (selectedModel.value === 'custom') {
            customInputs.style.display = 'block';
            params = parseFloat(customParams.value) * 1000000 || 175000000000;
        } else {
            customInputs.style.display = 'none';
            params = parseInt(selectedModel.value);
        }
        
        const memoryGB = parseFloat(availableMemory.value) || 16;
        
        updateCompatibilityResults(params, memoryGB);
        updateComparisonChart(params, memoryGB);
    }
    
    modelRadios.forEach(radio => {
        radio.addEventListener('change', updateModelCalculator);
    });
    
    if (customParams) customParams.addEventListener('input', updateModelCalculator);
    if (availableMemory) availableMemory.addEventListener('input', updateModelCalculator);
    
    updateModelCalculator();
}

function updateCompatibilityResults(params, availableMemoryGB) {
    const standardMemoryGB = (params * 4) / (1024**3);
    const reduction = calculateMemoryReduction(params);
    const checkpointMemoryGB = standardMemoryGB / reduction;
    
    const standardFits = standardMemoryGB <= availableMemoryGB;
    const checkpointFits = checkpointMemoryGB <= availableMemoryGB;
    
    const resultsContainer = document.getElementById('compatibility-results');
    if (!resultsContainer) return;
    
    resultsContainer.innerHTML = `
        <div class="card-grid">
            <div class="card text-center">
                <h5>Standard Inference</h5>
                <div style="font-size: 1.5rem; margin: 1rem 0; color: ${standardFits ? 'var(--success-color)' : 'var(--error-color)'};">
                    ${standardMemoryGB.toFixed(1)} GB
                </div>
                <span class="status-badge ${standardFits ? 'status-success' : 'status-error'}">
                    ${standardFits ? '✓ FITS' : '✗ TOO LARGE'}
                </span>
            </div>
            
            <div class="card text-center">
                <h5>Our Method</h5>
                <div style="font-size: 1.5rem; margin: 1rem 0; color: ${checkpointFits ? 'var(--success-color)' : 'var(--error-color)'};">
                    ${checkpointMemoryGB.toFixed(2)} GB
                </div>
                <span class="status-badge ${checkpointFits ? 'status-success' : 'status-error'}">
                    ${checkpointFits ? '✓ FITS' : '✗ TOO LARGE'}
                </span>
            </div>
        </div>
        
        <div class="card text-center" style="margin-top: 1rem;">
            <h5>Memory Reduction</h5>
            <div style="font-size: 2rem; color: var(--accent-color); margin: 1rem 0;">
                ${reduction.toFixed(0)}× smaller
            </div>
            <p>${checkpointFits && !standardFits ? 'Our method enables this model on your hardware!' : 
                 checkpointFits && standardFits ? 'Both methods work, but ours is much more efficient!' :
                 'This model requires more memory than available.'}</p>
        </div>
    `;
}

function updateComparisonChart(params, availableMemoryGB) {
    const ctx = document.getElementById('comparison-chart');
    if (!ctx) return;
    
    const standardMemoryGB = (params * 4) / (1024**3);
    const reduction = calculateMemoryReduction(params);
    const checkpointMemoryGB = standardMemoryGB / reduction;
    
    if (comparisonChart) {
        comparisonChart.destroy();
    }
    
    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Standard', 'Our Method', 'Available'],
            datasets: [{
                label: 'Memory (GB)',
                data: [standardMemoryGB, checkpointMemoryGB, availableMemoryGB],
                backgroundColor: [
                    'rgba(220, 38, 38, 0.7)',
                    'rgba(16, 185, 129, 0.7)',
                    'rgba(59, 130, 246, 0.7)'
                ],
                borderColor: [
                    'rgba(220, 38, 38, 1)',
                    'rgba(16, 185, 129, 1)',
                    'rgba(59, 130, 246, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Memory Requirements Comparison'
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Memory (GB)'
                    },
                    type: 'logarithmic'
                }
            }
        }
    });
}

// Demo 4: Scaling Comparison
function initializeScalingComparison() {
    const checkboxes = [
        'show-linear',
        'show-sqrt', 
        'show-log',
        'show-data'
    ];
    
    checkboxes.forEach(id => {
        const checkbox = document.getElementById(id);
        if (checkbox) {
            checkbox.addEventListener('change', updateScalingComparisonChart);
        }
    });
    
    updateScalingComparisonChart();
}

function updateScalingComparisonChart() {
    const ctx = document.getElementById('scaling-comparison-chart');
    if (!ctx) return;
    
    const showLinear = document.getElementById('show-linear')?.checked ?? true;
    const showSqrt = document.getElementById('show-sqrt')?.checked ?? true;
    const showLog = document.getElementById('show-log')?.checked ?? false;
    const showData = document.getElementById('show-data')?.checked ?? true;
    
    // Generate parameter range (logarithmic)
    const paramSizes = [];
    const linearMem = [];
    const sqrtMem = [];
    const logMem = [];
    
    for (let exp = 6; exp <= 12; exp += 0.2) {
        const params = Math.pow(10, exp);
        paramSizes.push(params);
        
        // Linear scaling O(n)
        linearMem.push((params * 4) / (1024**3)); // GB
        
        // Square root scaling O(√n) 
        const reduction = calculateMemoryReduction(params);
        sqrtMem.push(((params * 4) / (1024**3)) / reduction);
        
        // Logarithmic scaling O(log n) - theoretical best
        logMem.push(Math.log10(params) * 0.1); // Scaled for visualization
    }
    
    const datasets = [];
    
    if (showLinear) {
        datasets.push({
            label: 'O(n) Linear',
            data: paramSizes.map((params, i) => ({ x: params, y: linearMem[i] })),
            borderColor: 'rgba(220, 38, 38, 1)',
            backgroundColor: 'rgba(220, 38, 38, 0.1)',
            borderWidth: 3,
            fill: false,
            tension: 0.1
        });
    }
    
    if (showSqrt) {
        datasets.push({
            label: 'O(√n) Our Method',
            data: paramSizes.map((params, i) => ({ x: params, y: sqrtMem[i] })),
            borderColor: 'rgba(16, 185, 129, 1)',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 3,
            fill: false,
            tension: 0.1
        });
    }
    
    if (showLog) {
        datasets.push({
            label: 'O(log n) Theoretical Best',
            data: paramSizes.map((params, i) => ({ x: params, y: logMem[i] })),
            borderColor: 'rgba(59, 130, 246, 1)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 3,
            fill: false,
            tension: 0.1
        });
    }
    
    if (showData) {
        // Add our validation data points
        const validationPoints = validationData.memoryScaling.theoreticalParams.map((params, i) => ({
            x: params,
            y: (validationData.memoryScaling.checkpointParams[i] * 4) / (1024**3) // Convert to GB
        }));
        
        datasets.push({
            label: 'Our Validation Data',
            data: validationPoints,
            type: 'scatter',
            backgroundColor: 'rgba(245, 158, 11, 0.8)',
            borderColor: 'rgba(245, 158, 11, 1)',
            pointRadius: 6,
            pointHoverRadius: 8
        });
    }
    
    if (scalingComparisonChart) {
        scalingComparisonChart.destroy();
    }
    
    scalingComparisonChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Memory Scaling Laws Comparison'
                }
            },
            scales: {
                x: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Number of Parameters'
                    },
                    ticks: {
                        callback: function(value) {
                            return formatNumber(value);
                        }
                    }
                },
                y: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Memory Usage (GB)'
                    }
                }
            }
        }
    });
}

// Utility functions
function calculateTotalParams(numLayers, layerSize) {
    if (numLayers === 1) return layerSize * 10; // Single layer to output
    
    let totalParams = 0;
    
    // First layer (input to hidden)
    totalParams += layerSize * layerSize + layerSize;
    
    // Hidden layers
    for (let i = 1; i < numLayers - 1; i++) {
        totalParams += layerSize * layerSize + layerSize;
    }
    
    // Output layer (hidden to output)
    totalParams += layerSize * 10 + 10;
    
    return totalParams;
}

function getOptimalCheckpoints(numLayers) {
    if (numLayers <= 2) return [0];
    
    const interval = Math.max(1, Math.floor(Math.sqrt(numLayers)));
    const checkpoints = [];
    
    for (let i = 0; i < numLayers - 1; i += interval) {
        checkpoints.push(i);
    }
    
    return checkpoints;
}

function calculateCheckpointParams(numLayers, layerSize, checkpoints) {
    let totalParams = 0;
    
    checkpoints.forEach(layerIdx => {
        if (layerIdx === 0) {
            // Input layer
            totalParams += layerSize * layerSize + layerSize;
        } else if (layerIdx === numLayers - 1) {
            // Output layer
            totalParams += layerSize * 10 + 10;
        } else {
            // Hidden layer
            totalParams += layerSize * layerSize + layerSize;
        }
    });
    
    return totalParams;
}

function calculateRecomputationCost(numLayers, checkpoints) {
    let maxCost = 0;
    
    for (let i = 0; i < checkpoints.length - 1; i++) {
        const gap = checkpoints[i + 1] - checkpoints[i];
        maxCost = Math.max(maxCost, gap);
    }
    
    // Cost from last checkpoint to end
    const lastGap = numLayers - checkpoints[checkpoints.length - 1];
    maxCost = Math.max(maxCost, lastGap);
    
    return maxCost;
}