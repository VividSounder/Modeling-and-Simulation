// Global variables for state management
let isProcessing = false;
let currentDataTab = 'synthetic-data';
let features = [];
let classes = [];

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function showLoader() {
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');
    if (btnText && loader) {
        btnText.style.display = 'none';
        loader.style.display = 'inline-block';
    }
    isProcessing = true;
}

function hideLoader() {
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');
    if (btnText && loader) {
        btnText.style.display = 'inline-block';
        loader.style.display = 'none';
    }
    isProcessing = false;
}

function showError(message) {
    console.error('Error:', message);
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    
    const container = document.querySelector('.container');
    container.insertBefore(errorDiv, container.firstChild);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
    
    hideLoader();
}

function formatPercent(value) {
    return (value * 100).toFixed(1) + '%';
}

function updateSplitInfo(splitInfo) {
    if (!splitInfo) {
        console.error('No split info provided');
        return;
    }

    try {
        const elements = {
            train: document.querySelector('#train-split'),
            val: document.querySelector('#val-split'),
            test: document.querySelector('#test-split')
        };

        if (elements.train && elements.val && elements.test) {
            // Update training set info
            elements.train.querySelector('.split-size').textContent = 
                `${splitInfo.train_size} samples`;
            elements.train.querySelector('.split-ratio').textContent = 
                `${formatPercent(splitInfo.train_ratio)} of total`;

            // Update validation set info
            elements.val.querySelector('.split-size').textContent = 
                `${splitInfo.val_size} samples`;
            elements.val.querySelector('.split-ratio').textContent = 
                `${formatPercent(splitInfo.val_ratio)} of total`;

            // Update test set info
            elements.test.querySelector('.split-size').textContent = 
                `${splitInfo.test_size} samples`;
            elements.test.querySelector('.split-ratio').textContent = 
                `${formatPercent(splitInfo.test_ratio)} of total`;
        }
    } catch (error) {
        console.error('Error updating split info:', error);
    }
}

// Feature Configuration Management
function updateFeatureConfiguration() {
    const featureInput = document.getElementById('feature-names');
    const classInput = document.getElementById('class-names');
    
    if (!featureInput || !classInput) {
        console.error('Required input elements not found');
        return;
    }
    
    features = featureInput.value.split(',').map(f => f.trim()).filter(f => f);
    classes = classInput.value.split(',').map(c => c.trim()).filter(c => c);
    
    if (features.length > 0 && classes.length > 0) {
        generateClassConfigs();
        updateTargetFeatureOptions();
    }
}

function generateClassConfigs() {
    const container = document.getElementById('class-configs');
    if (!container) {
        console.error('Class configs container not found');
        return;
    }
    
    container.innerHTML = '';
    
    classes.forEach(className => {
        const classDiv = document.createElement('div');
        classDiv.className = 'class-config';
        
        classDiv.innerHTML = `
            <div class="class-header">
                <h5>${className} Settings</h5>
                <label class="toggle">
                    <input type="checkbox" checked>
                    <span>Set specific values for ${className}</span>
                </label>
            </div>
            <div class="feature-settings-container">
                ${features.map(feature => `
                    <div class="feature-settings">
                        <div class="input-group">
                            <label>Mean for ${feature}</label>
                            <input type="number" step="0.01" 
                                   class="feature-mean" 
                                   data-class="${className}" 
                                   data-feature="${feature}"
                                   value="0">
                            <div class="number-controls">
                                <button type="button" class="decrease">-</button>
                                <button type="button" class="increase">+</button>
                            </div>
                        </div>
                        <div class="input-group">
                            <label>Std Dev for ${feature}</label>
                            <input type="number" step="0.01" min="0" 
                                   class="feature-std" 
                                   data-class="${className}" 
                                   data-feature="${feature}"
                                   value="1">
                            <div class="number-controls">
                                <button type="button" class="decrease">-</button>
                                <button type="button" class="increase">+</button>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>`;
        
        container.appendChild(classDiv);
    });

    // Add event listeners for number controls
    document.querySelectorAll('.number-controls button').forEach(button => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            const input = button.parentElement.previousElementSibling;
            const step = parseFloat(input.step) || 1;
            const min = parseFloat(input.min) || -Infinity;
            if (button.classList.contains('increase')) {
                input.value = (parseFloat(input.value) || 0) + step;
            } else {
                input.value = Math.max((parseFloat(input.value) || 0) - step, min);
            }
            input.dispatchEvent(new Event('change'));
        });
    });

    // Add event listeners for toggle switches
    document.querySelectorAll('.toggle input[type="checkbox"]').forEach(toggle => {
        toggle.addEventListener('change', (e) => {
            const container = e.target.closest('.class-config').querySelector('.feature-settings-container');
            if (container) {
                container.style.display = e.target.checked ? 'grid' : 'none';
            }
        });
    });
}

function updateTargetFeatureOptions() {
    const targetSelect = document.getElementById('target_column');
    if (!targetSelect) {
        console.error('Target column select not found');
        return;
    }
    
    targetSelect.innerHTML = '<option value="">Select target feature...</option>';
    
    features.forEach(feature => {
        const option = document.createElement('option');
        option.value = feature;
        option.textContent = feature;
        targetSelect.appendChild(option);
    });
}

function collectFeatureConfig() {
    const config = {};
    
    classes.forEach(className => {
        config[className] = {};
        features.forEach(feature => {
            const meanInput = document.querySelector(
                `.feature-mean[data-class="${className}"][data-feature="${feature}"]`
            );
            const stdInput = document.querySelector(
                `.feature-std[data-class="${className}"][data-feature="${feature}"]`
            );
            
            if (meanInput && stdInput) {
                config[className][feature] = {
                    mean: parseFloat(meanInput.value) || 0,
                    std: parseFloat(stdInput.value) || 1
                };
            }
        });
    });
    
    return config;
}

function showTab(tabId) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab content
    const selectedTab = document.getElementById(tabId);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }
    
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-tab') === tabId) {
            btn.classList.add('active');
        }
    });
}

function showTab(tabId) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab content
    const selectedTab = document.getElementById(tabId);
    if (selectedTab) {
        selectedTab.classList.add('active');
        
        // Trigger resize for plots in the newly shown tab
        if (tabId === 'data-distribution') {
            selectedTab.querySelectorAll('.visualization-plot').forEach(plot => {
                if (plot.data) {
                    Plotly.relayout(plot, {
                        'autosize': true
                    });
                }
            });
        }
    }
    
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('onclick').includes(tabId)) {
            btn.classList.add('active');
        }
    });
}

window.addEventListener('resize', debounce(() => {
    document.querySelectorAll('.visualization-plot').forEach(plot => {
        if (plot.data) {
            Plotly.relayout(plot, {
                'autosize': true
            });
        }
    });
}, 250));

function resizePlots() {
    const plots = document.querySelectorAll('.visualization-plot');
    plots.forEach(plot => {
        if (plot.data) {
            Plotly.Plots.resize(plot);
        }
    });
}

function displayModelResults(results) {
    if (!results) {
        console.error('No model results provided');
        return;
    }

    try {
        const table = document.getElementById('model-metrics');
        if (!table) {
            console.error('Model metrics table not found');
            return;
        }

        // Clear existing content
        table.innerHTML = '';

        // Create table header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        headerRow.innerHTML = `
            <th>Model</th>
            <th>Training</th>
            <th>Validation</th>
            <th>Test</th>
            <th>CV Score</th>
        `;
        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Create table body
        const tbody = document.createElement('tbody');
        for (const [modelName, metrics] of Object.entries(results)) {
            if (!metrics.error) {
                const row = document.createElement('tr');
                const isClassification = 'train_accuracy' in metrics;
                
                row.innerHTML = `
                    <td>${modelName}</td>
                    <td>${isClassification ? 
                        formatPercent(metrics.train_accuracy) : 
                        `MSE: ${metrics.train_mse.toFixed(4)}<br>R²: ${metrics.train_r2.toFixed(4)}`
                    }</td>
                    <td>${isClassification ? 
                        formatPercent(metrics.val_accuracy) : 
                        `MSE: ${metrics.val_mse.toFixed(4)}<br>R²: ${metrics.val_r2.toFixed(4)}`
                    }</td>
                    <td>${isClassification ? 
                        formatPercent(metrics.test_accuracy) : 
                        `MSE: ${metrics.test_mse.toFixed(4)}<br>R²: ${metrics.test_r2.toFixed(4)}`
                    }</td>
                    <td>${formatPercent(metrics.cv_scores)}</td>
                `;
                tbody.appendChild(row);
            }
        }
        table.appendChild(tbody);
    } catch (error) {
        console.error('Error displaying model results:', error);
    }
}

function displayTable(elementId, data) {
    if (!data) {
        console.error(`No data provided for table ${elementId}`);
        return;
    }

    try {
        const container = document.getElementById(elementId);
        if (!container) {
            console.error(`Container not found: ${elementId}`);
            return;
        }

        // Create table HTML directly instead of DOM manipulation
        const tableHTML = `
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            ${Object.keys(data[0]).map(key => `<th>${key}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${data.map(row => `
                            <tr>
                                ${Object.values(row).map(value => 
                                    `<td>${typeof value === 'number' ? value.toFixed(4) : value}</td>`
                                ).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;

        // Replace entire content
        container.innerHTML = tableHTML;
    } catch (error) {
        console.error('Error displaying table:', error);
        document.getElementById(elementId).innerHTML = '<p class="error">Error displaying data</p>';
    }
}

function displayVisualizations(visualizations) {
    if (!visualizations) {
        console.error('No visualizations provided');
        return;
    }

    try {
        const config = {
            responsive: true,
            displayModeBar: 'hover',
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            toImageButtonOptions: {
                format: 'png',
                filename: 'plot',
                scale: 2
            }
        };

        // Common layout settings
        const baseLayout = {
            autosize: true,
            margin: { l: 50, r: 30, t: 50, b: 50, pad: 4 },
            paper_bgcolor: '#1e1e1e',
            plot_bgcolor: '#1e1e1e',
            font: {
                color: '#e0e0e0',
                size: 12
            },
            showlegend: true,
            legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: 1.02,
                xanchor: 'right',
                x: 1,
                bgcolor: 'rgba(30,30,30,0.8)'
            },
            xaxis: {
                gridcolor: '#333',
                linecolor: '#999',
                tickfont: { size: 10 }
            },
            yaxis: {
                gridcolor: '#333',
                linecolor: '#999',
                tickfont: { size: 10 }
            }
        };

        // Display model comparison visualizations
        const modelVisuals = document.getElementById('model-visualizations');
        if (modelVisuals) {
            modelVisuals.innerHTML = '';
            
            ['model_mse', 'model_r2', 'model_accuracy', 'cv_scores'].forEach(key => {
                if (visualizations[key]) {
                    const container = document.createElement('div');
                    container.className = 'visualization-container';
                    
                    const plot = document.createElement('div');
                    plot.className = 'visualization-plot';
                    container.appendChild(plot);
                    
                    const data = JSON.parse(visualizations[key]);
                    const layout = {
                        ...baseLayout,
                        ...data.layout,
                        height: 400,
                        width: undefined // Let Plotly handle the width responsively
                    };
                    
                    modelVisuals.appendChild(container);
                    Plotly.newPlot(plot, data.data, layout, config);
                    
                    // Add resize handler
                    const resizePlot = () => {
                        Plotly.Plots.resize(plot);
                    };
                    
                    // Debounced resize observer
                    const resizeObserver = new ResizeObserver(debounce(() => {
                        resizePlot();
                    }, 100));
                    
                    resizeObserver.observe(container);
                }
            });
        }
        
        // Display distribution plots
        const distributionPlots = document.getElementById('distribution-plots');
        if (distributionPlots) {
            distributionPlots.innerHTML = '';
            
            Object.entries(visualizations).forEach(([key, value]) => {
                if (key.startsWith('dist_') || key === 'correlation') {
                    const container = document.createElement('div');
                    container.className = 'visualization-container';
                    
                    const plot = document.createElement('div');
                    plot.className = 'visualization-plot';
                    container.appendChild(plot);
                    
                    const data = JSON.parse(value);
                    const layout = {
                        ...baseLayout,
                        ...data.layout,
                        height: key === 'correlation' ? 450 : 400,
                        width: key === 'correlation' ? 450 : undefined
                    };
                    
                    distributionPlots.appendChild(container);
                    Plotly.newPlot(plot, data.data, layout, config);
                    
                    // Add resize handler
                    const resizePlot = () => {
                        if (key !== 'correlation') { // Don't resize correlation matrix
                            Plotly.Plots.resize(plot);
                        }
                    };
                    
                    // Debounced resize observer
                    const resizeObserver = new ResizeObserver(debounce(() => {
                        resizePlot();
                    }, 100));
                    
                    resizeObserver.observe(container);
                }
            });
        }
    } catch (error) {
        console.error('Error displaying visualizations:', error);
        showError('Error displaying visualizations');
    }
}

// Improved resize handling
function resizePlots() {
    document.querySelectorAll('.visualization-plot').forEach(plot => {
        if (plot._fullLayout && plot._fullLayout.height > 0) {
            Plotly.Plots.resize(plot);
        }
    });
}

function downloadDataset(type) {
    try {
        window.location.href = `/download/${type}`;
    } catch (error) {
        console.error('Error downloading dataset:', error);
        showError(`Failed to download ${type} dataset`);
    }
}

// Event Listeners
document.getElementById('feature-names')?.addEventListener('change', updateFeatureConfiguration);
document.getElementById('class-names')?.addEventListener('change', updateFeatureConfiguration);

document.getElementById('dataForm')?.addEventListener('submit', function(e) {
    e.preventDefault();
    if (isProcessing) return;
    
    const featureConfig = collectFeatureConfig();
    const targetColumn = document.getElementById('target_column')?.value;
    
    if (!targetColumn) {
        showError('Please select a target feature');
        return;
    }
    
    const params = {
        num_samples: parseInt(document.getElementById('num_samples')?.value || '1000'),
        feature_config: featureConfig,
        target_column: targetColumn,
        test_size: parseFloat(document.getElementById('test_size')?.value || '0.2'),
        val_size: parseFloat(document.getElementById('val_size')?.value || '0.25'),
        random_state: parseInt(document.getElementById('random_state')?.value || '42')
    };
    
    console.log('Sending params:', params);
    showLoader();
    
    fetch('/generate_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(params)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Check if data is valid before proceeding
        if (!data || typeof data !== 'object') {
            throw new Error('Invalid response data');
        }
        
        const resultsSection = document.querySelector('.results-section');
        if (resultsSection) {
            resultsSection.style.display = 'block';
        }
        
        // Add null checks for each data property
        if (data.model_results) displayModelResults(data.model_results);
        if (data.visualizations) displayVisualizations(data.visualizations);
        if (data.synthetic_data) displayTable('synthetic-data', data.synthetic_data);
        if (data.train_data) displayTable('train-data', data.train_data);
        if (data.validation_data) displayTable('validation-data', data.validation_data);
        if (data.test_data) displayTable('test-data', data.test_data);
        if (data.split_info) updateSplitInfo(data.split_info);
        
        showTab('model-comparison');
        window.scrollTo({
            top: resultsSection?.offsetTop || 0,
            behavior: 'smooth'
        });
    })
    .catch(error => {
        console.error('Error:', error);
        showError(`Error: ${error.message || 'An error occurred while processing the data'}`);
    })
    .finally(hideLoader);
});

// Initialize
window.addEventListener('resize', debounce(resizePlots, 250));
document.addEventListener('DOMContentLoaded', updateFeatureConfiguration);