// Main JavaScript for Airbnb Bali ML Dashboard

document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Airbnb Bali ML Dashboard loaded successfully');
    
    // Initialize tooltips
    initializeTooltips();
    
    // Form validation
    initializeFormValidation();
    
    // Smooth scrolling
    initializeSmoothScrolling();
    
    // Animation on scroll
    initializeScrollAnimation();
    
    // Currency selector change handler
    initializeCurrencyHandler();
});

// Initialize Bootstrap tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Form validation
function initializeFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
}

// Smooth scrolling for anchor links
function initializeSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href !== '#' && href !== '#!') {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });
}

// Animate elements on scroll
function initializeScrollAnimation() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, {
        threshold: 0.1
    });
    
    document.querySelectorAll('.card, .result-card').forEach(el => {
        observer.observe(el);
    });
}

// Currency selector handler
function initializeCurrencyHandler() {
    const currencySelect = document.getElementById('display_currency');
    if (currencySelect) {
        currencySelect.addEventListener('change', function() {
            const selectedCurrency = this.value;
            console.log('Currency changed to:', selectedCurrency);
            
            // You can add visual feedback here
            const rateInfo = document.querySelector('.rate-info');
            if (rateInfo) {
                if (selectedCurrency === 'IDR') {
                    rateInfo.classList.add('text-success');
                    rateInfo.classList.remove('text-muted');
                } else {
                    rateInfo.classList.add('text-muted');
                    rateInfo.classList.remove('text-success');
                }
            }
        });
    }
}

// Format number with thousand separators
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Show loading spinner
function showLoading(buttonId) {
    const button = document.getElementById(buttonId);
    if (button) {
        button.disabled = true;
        button.innerHTML = '<span class="loading"></span> Processing...';
    }
}

// Hide loading spinner
function hideLoading(buttonId, originalText) {
    const button = document.getElementById(buttonId);
    if (button) {
        button.disabled = false;
        button.innerHTML = originalText;
    }
}

// Show notification
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed top-0 end-0 m-3`;
    alertDiv.style.zIndex = '9999';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.appendChild(alertDiv);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// Copy to clipboard function
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard!', 'success');
    }).catch(err => {
        console.error('Failed to copy:', err);
        showNotification('Failed to copy to clipboard', 'danger');
    });
}

// Print prediction results
function printResults() {
    window.print();
}

// Export data as JSON
function exportDataAsJSON(data, filename = 'prediction_data.json') {
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
}

// Console log for debugging
console.log('✅ Dashboard JavaScript initialized');
