// Global variables for charts
let goChart, fgChart, puntChart;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeSliders();
    initializeCharts();
    setupEventListeners();
    initializeQuarterButtons();
    initializeBallSideButtons();
    // Load initial data
    calculateDecision();
});

// Setup slider value displays
function initializeSliders() {
    const kickerSlider = document.getElementById('kicker-range');
    const punterSlider = document.getElementById('punter-range');
    const kickerValue = document.getElementById('kicker-value');
    const punterValue = document.getElementById('punter-value');
    
    kickerSlider.addEventListener('input', function() {
        kickerValue.textContent = this.value;
    });
    
    punterSlider.addEventListener('input', function() {
        punterValue.textContent = this.value;
    });
}

// Initialize Chart.js charts
function initializeCharts() {
    // Go Chart
    const goCtx = document.getElementById('go-chart').getContext('2d');
    goChart = new Chart(goCtx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [75, 25],
                backgroundColor: ['#000000', '#e1e5e9'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: false,
            maintainAspectRatio: false,
            cutout: '60%',
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
    
    // FG Chart
    const fgCtx = document.getElementById('fg-chart').getContext('2d');
    fgChart = new Chart(fgCtx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [65, 35],
                backgroundColor: ['#000000', '#e1e5e9'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: false,
            maintainAspectRatio: false,
            cutout: '60%',
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
    
    // Punt Chart
    const puntCtx = document.getElementById('punt-chart').getContext('2d');
    puntChart = new Chart(puntCtx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [75, 25],
                backgroundColor: ['#000000', '#e1e5e9'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: false,
            maintainAspectRatio: false,
            cutout: '60%',
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Initialize quarter buttons
function initializeQuarterButtons() {
    const quarterButtons = document.querySelectorAll('.quarter-btn');
    quarterButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            quarterButtons.forEach(btn => btn.classList.remove('active'));
            // Add active class to clicked button
            this.classList.add('active');
            // Trigger calculation
            calculateDecision();
        });
    });
}

// Initialize ball side buttons
function initializeBallSideButtons() {
    const ballSideButtons = document.querySelectorAll('.ball-side-btn');
    ballSideButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            ballSideButtons.forEach(btn => btn.classList.remove('active'));
            // Add active class to clicked button
            this.classList.add('active');
            // Trigger calculation
            calculateDecision();
        });
    });
}

// Setup event listeners
function setupEventListeners() {
    document.getElementById('calculate-btn').addEventListener('click', calculateDecision);
    
    // Auto-calculate when inputs change
    const inputs = document.querySelectorAll('input');
    inputs.forEach(input => {
        input.addEventListener('change', calculateDecision);
        input.addEventListener('input', debounce(calculateDecision, 500));
    });
}

// Debounce function to limit API calls
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

// Get current input values
function getInputValues() {
    const activeQuarterBtn = document.querySelector('.quarter-btn.active');
    const activeBallSideBtn = document.querySelector('.ball-side-btn.active');
    return {
        currentYardline: parseInt(document.getElementById('current-yardline').value),
        ballSide: activeBallSideBtn.getAttribute('data-side'),
        yardsToGo: parseInt(document.getElementById('yards-to-go').value),
        quarter: parseInt(activeQuarterBtn.getAttribute('data-quarter')),
        timeRemaining: document.getElementById('time-remaining').value,
        scoreDifferential: parseInt(document.getElementById('score-differential').value),
        kickerRange: parseInt(document.getElementById('kicker-range').value),
        punterRange: parseInt(document.getElementById('punter-range').value)
    };
}

// Calculate decision (main function)
async function calculateDecision() {
    const inputs = getInputValues();
    
    try {
        // Show loading state
        document.getElementById('calculate-btn').classList.add('loading');
        
        // Use the backend API for calculations
        const results = await calculateDecisionWithAPI(inputs);
        
        // Update the UI with results
        updateResults(results);
        
    } catch (error) {
        console.error('Error calculating decision:', error);
        // Show error state
        updateResults({
            error: 'Failed to calculate decision. Please try again.'
        });
    } finally {
        // Remove loading state
        document.getElementById('calculate-btn').classList.remove('loading');
    }
}

// Calculate decision using the backend API
async function calculateDecisionWithAPI(inputs) {
    const response = await fetch('/api/calculate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputs)
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const results = await response.json();
    return results;
}


// Update the UI with calculation results
function updateResults(results) {
    if (results.error) {
        // Handle error state
        document.getElementById('recommendation-decision').textContent = 'Error';
        return;
    }
    
    // Update Go section
    document.getElementById('go-td-prob').textContent = `+${results.go.tdProb}%`;
    document.getElementById('go-fg-prob').textContent = `${results.go.fgProb}%`;
    document.getElementById('go-noscore-prob').textContent = `${results.go.noScoreProb}%`;
    document.getElementById('go-value').textContent = `+${results.go.wpa}%`;
    goChart.data.datasets[0].data = results.go.chartData;
    goChart.update();
    
    // Update FG section
    document.getElementById('fg-td-prob').textContent = `${results.fg.tdProb}%`;
    document.getElementById('fg-fg-prob').textContent = `+${results.fg.fgProb}%`;
    document.getElementById('fg-noscore-prob').textContent = `${results.fg.noScoreProb}%`;
    document.getElementById('fg-value').textContent = `+${results.fg.wpa}%`;
    fgChart.data.datasets[0].data = results.fg.chartData;
    fgChart.update();
    
    // Update Punt section
    document.getElementById('punt-net-td').textContent = `${results.punt.netTd}%`;
    document.getElementById('punt-score').textContent = `+${results.punt.score}%`;
    document.getElementById('punt-win').textContent = `+${results.punt.win}%`;
    document.getElementById('punt-value').textContent = `+${results.punt.wpa}%`;
    puntChart.data.datasets[0].data = results.punt.chartData;
    puntChart.update();
    
    // Update Recommendation
    document.getElementById('recommendation-decision').textContent = results.recommendation.decision;
    document.getElementById('recommendation-wpa').textContent = `+${results.recommendation.wpa}% WPA`;
    document.getElementById('recommendation-win').textContent = `+${results.recommendation.win}% Δ Win%`;
}

