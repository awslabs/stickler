/**
 * @fileoverview Interactive functionality for HTML evaluation reports
 * Handles document drill-down, search, filtering, and navigation between aggregate and individual document views
 * 
 * Global State Variables:
 * @var {Array<Object>} documentData - Array of individual document results from JSONL file
 * @var {Object} fieldThresholds - Field name -> threshold value mapping extracted from model schema  
 * @var {Object} aggregateData - Original aggregate metrics captured from DOM on page load
 * @var {string|null} currentFilterDoc - Current filtered document ID (null = showing aggregate view)
 * 
 * Data Structure Notes:
 * - Individual document structure: doc.comparison_result.confusion_matrix.fields.fieldName.overall.derived.cm_*
 * - Aggregate data structure: captured directly from DOM elements on page load
 */

let documentData = [];
let fieldThresholds = {};
let aggregateData = null;
let currentFilterDoc = null; // null means showing aggregate data

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Check if we have individual documents section
    if (document.getElementById('individual-documents')) {
        setupDocumentInteraction();
        setupSearch();
        setupFilter();
    }
});

/**
 * Initialize document data from HTML (called by inline script in HTML template)
 * @param {Array<Object>} docs - Array of individual document evaluation results
 * @param {Object} thresholds - Field name to threshold value mapping
 */
function initializeDocumentData(docs, thresholds) {
    documentData = docs;
    fieldThresholds = thresholds;
    
    // Store original aggregate data on first load
    if (!aggregateData) {
        captureAggregateData();
        addReportFilterControls();
    }
}

// ============================================================================
// DATA INITIALIZATION & CAPTURE
// ============================================================================

/**
 * Capture the original aggregate data from the DOM elements
 * This preserves the aggregate view so we can switch back from individual document views
 */
function captureAggregateData() {
    aggregateData = {
        executiveSummary: captureExecutiveSummaryData(),
        fieldAnalysis: captureFieldAnalysisData(),
        confusionMatrix: captureConfusionMatrixData(),
        nonMatches: captureNonMatchesData(),
    };

    const imageGallery = document.querySelector('.image-gallery');
    if (imageGallery) {
        aggregateData.documentImages = captureDocumentImageData();
    }
}

function captureExecutiveSummaryData() {
    const data = {};
    
    // Capture performance gauge
    const gaugeValue = document.querySelector('.gauge-value')?.textContent;
    if (gaugeValue) {
        data.gaugeValue = parseFloat(gaugeValue.replace('%', '')) / 100;
    }
    
    // Capture metric cards
    data.metrics = {};
    document.querySelectorAll('.metric-card').forEach(card => {
        const label = card.querySelector('.metric-label')?.textContent.toLowerCase().replace(' ', '_');
        const value = card.querySelector('.metric-value')?.textContent;
        if (label && value && label !== 'documents') {
            data.metrics[label] = parseFloat(value) || value;
        }
    });
    
    return data;
}

function captureDocumentImageData() {
    const data = {};

    // Capture image data
    document.querySelectorAll('.image-item').forEach(image => {
        const imgElement = image.querySelector('img')?.src;
        const docId = image.querySelector('p strong')?.textContent;
        if (imgElement && docId) {
            data[docId] = imgElement;
        }
    });
    return data;
}

function captureFieldAnalysisData() {
    const data = {
        chart: [],
        table: []
    };
    
    // Capture field chart data
    document.querySelectorAll('.field-bar').forEach(bar => {
        const label = bar.querySelector('.field-label')?.textContent;
        const value = bar.querySelector('.bar-value')?.textContent;
        const width = bar.querySelector('.bar-fill')?.style.width;
        const color = bar.querySelector('.bar-fill')?.style.backgroundColor;
        
        if (label && value) {
            data.chart.push({
                field: label,
                value: parseFloat(value),
                width: width,
                color: color
            });
        }
    });
    
    // Capture performance table data
    const tableRows = document.querySelectorAll('.performance-table tbody tr');
    tableRows.forEach(row => {
        const cells = row.querySelectorAll('td');
        if (cells.length >= 8) {
            data.table.push({
                field: cells[0].textContent,
                precision: parseFloat(cells[1].textContent),
                recall: parseFloat(cells[2].textContent),
                f1: parseFloat(cells[3].textContent),
                tp: parseInt(cells[4].textContent),
                fd: parseInt(cells[5].textContent),
                fa: parseInt(cells[6].textContent),
                fn: parseInt(cells[7].textContent)
            });
        }
    });
    
    return data;
}

function captureConfusionMatrixData() {
    const data = {};
    
    document.querySelectorAll('.cm-cell').forEach(cell => {
        const label = cell.querySelector('.cm-label')?.textContent;
        const value = cell.querySelector('.cm-value')?.textContent;
        const percentage = cell.querySelector('.cm-percentage')?.textContent;
        
        if (label && value) {
            data[label.toLowerCase()] = {
                value: parseInt(value),
                percentage: percentage
            };
        }
    });
    
    return data;
}

function captureNonMatchesData() {
    const data = []

    document.querySelectorAll('.non-matches-table tbody tr').forEach(row => {
        const cells = row.querySelectorAll('td');
        if (cells.length >= 5) {
            data.push({
                doc_id: cells[0].textContent,
                field_path: cells[1].textContent,
                non_match_type: cells[2].textContent,
                ground_truth_value: cells[3].textContent,
                prediction_value: cells[4].textContent
            })
        }
    });

    return data;
}

function addReportFilterControls() {
    // Add a filter control above the first section
    const firstSection = document.querySelector('.section');
    if (firstSection) {
        const filterControl = document.createElement('div');
        filterControl.className = 'report-filter-control';
        filterControl.innerHTML = `<div style="background: #f8f9fa; padding: 15px; border-radius: 6px; margin-bottom: 20px; border-left: 4px solid #007bff;">
                <strong>Report View:</strong>
                <select id="report-doc-filter" style="margin-left: 10px; padding: 5px 10px; border: 1px solid #dee2e6; border-radius: 4px;">
                    <option value="">All Documents (Aggregate)</option>
                </select>
                <button id="reset-report-filter" style="margin-left: 10px; padding: 5px 10px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer;">Reset to Aggregate</button>
            </div>`;
        
        firstSection.parentNode.insertBefore(filterControl, firstSection);
        
        // Populate document options
        const select = document.getElementById('report-doc-filter');
        if (select) {
            documentData.forEach(doc => {
                const option = document.createElement('option');
                option.value = doc.doc_id;
                option.textContent = doc.doc_id;
                select.appendChild(option);
            });
        }
        
        // Add event listeners with error handling
        const selectElement = document.getElementById('report-doc-filter');
        if (selectElement) {
            selectElement.addEventListener('change', function() {
                filterReportToDocument(this.value || null);
            });
        }
        
        const resetButton = document.getElementById('reset-report-filter');
        if (resetButton) {
            resetButton.addEventListener('click', function() {
                const select = document.getElementById('report-doc-filter');
                if (select) {
                    select.value = '';
                    filterReportToDocument(null);
                }
            });
        }
    }
}

// ============================================================================
// DOCUMENT FILTERING & MAIN REPORT UPDATES
// ============================================================================

/**
 * Main function to filter the entire report to show metrics for a specific document
 * @param {string|null} docId - Document ID to filter to, or null for aggregate view
 */
function filterReportToDocument(docId) {
    currentFilterDoc = docId;
    
    if (!docId) {
        // Show aggregate data - restore original captured metrics
        updateExecutiveSummary(aggregateData.executiveSummary);
        updateFieldAnalysis(aggregateData.fieldAnalysis);
        updateConfusionMatrix(aggregateData.confusionMatrix);
        updateNonMatchesTable(aggregateData.nonMatches);
        updateDocumentImages(aggregateData.documentImages);
        updateReportTitle("All Documents");
    } else {
        // Find the document and show its individual metrics
        const doc = documentData.find(d => d.doc_id === docId);
        if (doc) {
            const docMetrics = extractDocumentMetrics(doc);
            updateExecutiveSummary(docMetrics.executiveSummary);
            updateFieldAnalysis(docMetrics.fieldAnalysis);
            updateConfusionMatrix(docMetrics.confusionMatrix);
            updateNonMatchesTable(docMetrics.nonMatches);
            updateDocumentImages(docMetrics.documentImages);
            updateReportTitle(`Document: ${docId}`);
        }
    }
}

/**
 * Extract metrics from individual document data for display in the main report sections
 * @param {Object} doc - Individual document data containing comparison results
 * @returns {Object} Structured metrics data matching the aggregate data format
 * 
 * Data Structure:
 * - Individual docs store field metrics at: doc.comparison_result.confusion_matrix.fields.fieldName.overall.derived.cm_*
 * - Overall metrics are at: doc.comparison_result.confusion_matrix.overall.derived.cm_*
 */
function extractDocumentMetrics(doc) {
    const comparison = doc.comparison_result;
    const confusionMatrix = comparison.confusion_matrix || {};
    const overallMetrics = confusionMatrix.overall || {};
    const fieldsData = confusionMatrix.fields || {};
    const nonMatches = comparison.non_matches || []
    const documentImages = {};
    if (aggregateData.documentImages && aggregateData.documentImages[doc.doc_id]) {
        documentImages[doc.doc_id] = aggregateData.documentImages[doc.doc_id];
    }

    
    return {
        executiveSummary: {
            gaugeValue: overallMetrics.derived?.cm_f1 || 0,
            metrics: {
                precision: overallMetrics.derived?.cm_precision || 0,
                recall: overallMetrics.derived?.cm_recall || 0,
                f1: overallMetrics.derived?.cm_f1 || 0,
                accuracy: overallMetrics.derived?.cm_accuracy || 0
            }
        },
        fieldAnalysis: {
            // Fix: Chart data needs to use the same path as table data for consistency
            chart: Object.entries(fieldsData).map(([field, data]) => ({
                field: field,
                value: data.overall?.derived?.cm_f1 || 0,
                width: `${Math.round((data.overall?.derived?.cm_f1 || 0) * 100)}%`,
                color: getPerformanceColor(data.overall?.derived?.cm_f1 || 0)
            })),
            // Table data: Extract field-level precision, recall, F1 from correct nested structure
            table: Object.entries(fieldsData).map(([field, data]) => ({
                field: field,
                precision: data.overall?.derived?.cm_precision || 0,
                recall: data.overall?.derived?.cm_recall || 0,
                f1: data.overall?.derived?.cm_f1 || 0,
                tp: data.overall?.tp || 0,
                fd: data.overall?.fd || 0,
                fa: data.overall?.fa || 0,
                fn: data.overall?.fn || 0
            }))
        },
        confusionMatrix: {
            tp: { value: overallMetrics.tp || 0, percentage: `${Math.round(((overallMetrics.tp || 0) / Math.max(1, (overallMetrics.tp || 0) + (overallMetrics.fd || 0) + (overallMetrics.fa || 0) + (overallMetrics.fn || 0))) * 100)}%` },
            tn: { value: overallMetrics.tn || 0, percentage: `${Math.round(((overallMetrics.tn || 0) / Math.max(1, (overallMetrics.tp || 0) + (overallMetrics.fd || 0) + (overallMetrics.fa || 0) + (overallMetrics.fn || 0))) * 100)}%` },
            fd: { value: overallMetrics.fd || 0, percentage: `${Math.round(((overallMetrics.fd || 0) / Math.max(1, (overallMetrics.tp || 0) + (overallMetrics.fd || 0) + (overallMetrics.fa || 0) + (overallMetrics.fn || 0))) * 100)}%` },
            fa: { value: overallMetrics.fa || 0, percentage: `${Math.round(((overallMetrics.fa || 0) / Math.max(1, (overallMetrics.tp || 0) + (overallMetrics.fd || 0) + (overallMetrics.fa || 0) + (overallMetrics.fn || 0))) * 100)}%` },
            fn: { value: overallMetrics.fn || 0, percentage: `${Math.round(((overallMetrics.fn || 0) / Math.max(1, (overallMetrics.tp || 0) + (overallMetrics.fd || 0) + (overallMetrics.fa || 0) + (overallMetrics.fn || 0))) * 100)}%` }
        },
        nonMatches: nonMatches.map(nonMatch => ({
            doc_id: doc.doc_id,
            field_path: nonMatch.field_path,
            non_match_type: nonMatch.non_match_type,
            ground_truth_value: nonMatch.ground_truth_value,
            prediction_value: nonMatch.prediction_value
        })),
        documentImages: documentImages
    };
}

function getPerformanceColor(value) {
    if (value >= 0.8) return '#28a745'; // Green
    if (value >= 0.6) return '#ffc107'; // Yellow
    return '#dc3545'; // Red
}

function updateReportTitle(title) {
    // Update any section headers or add a visual indicator
    const header = document.querySelector('header h1');
    if (header) {
        const originalTitle = header.textContent.split(' - ')[0];
        header.textContent = `${originalTitle} - ${title}`;
    }
}

function updateExecutiveSummary(data) {
    // Update performance gauge
    const gaugeCircle = document.querySelector('.gauge-circle');
    const gaugeValue = document.querySelector('.gauge-value');
    if (gaugeCircle && gaugeValue && data.gaugeValue !== undefined) {
        const percentage = Math.round(data.gaugeValue * 100);
        const color = getPerformanceColor(data.gaugeValue);
        
        gaugeCircle.style.background = `conic-gradient(${color} ${percentage}%, #e9ecef ${percentage}%)`;
        gaugeValue.textContent = `${percentage}%`;
    }
    
    // Update metric cards
    if (data.metrics) {
        Object.entries(data.metrics).forEach(([metric, value]) => {
            const metricCards = document.querySelectorAll('.metric-card');
            metricCards.forEach(card => {
                const label = card.querySelector('.metric-label')?.textContent.toLowerCase().replace(' ', '_');
                if (label === metric) {
                    const valueElement = card.querySelector('.metric-value');
                    if (valueElement) {
                        valueElement.textContent = typeof value === 'number' ? value.toFixed(3) : value;
                        valueElement.style.color = getPerformanceColor(parseFloat(value) || 0);
                    }
                }
            });
        });
    }
}

// ============================================================================
// UI UPDATE FUNCTIONS
// ============================================================================

/**
 * Update field analysis section (chart and table) with new data
 * @param {Object} data - Field analysis data
 * @param {Array} data.chart - Chart data: [{field, value, width, color}, ...]
 * @param {Array} data.table - Table data: [{field, precision, recall, f1, tp, fd, fa, fn}, ...]
 */
function updateFieldAnalysis(data) {
    // Update field chart - match by field name and update visual bars
    if (data.chart) {
        const fieldBars = document.querySelectorAll('.field-bar');
        fieldBars.forEach(bar => {
            const fieldLabel = bar.querySelector('.field-label')?.textContent;
            if (fieldLabel) {
                const fieldData = data.chart.find(item => item.field === fieldLabel);
                if (fieldData) {
                    const barFill = bar.querySelector('.bar-fill');
                    const barValue = bar.querySelector('.bar-value');
                    
                    if (barFill) {
                        barFill.style.width = fieldData.width;
                        barFill.style.backgroundColor = fieldData.color;
                    }
                    if (barValue) {
                        barValue.textContent = fieldData.value.toFixed(3);
                    }
                }
            }
        });
    }
    
    // Update performance table - match by field name and update all metric columns
    if (data.table) {
        const tableRows = document.querySelectorAll('.performance-table tbody tr');
        tableRows.forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length >= 8) {
                const fieldName = cells[0].textContent;
                const rowData = data.table.find(item => item.field === fieldName);
                if (rowData) {
                    // Update precision, recall, F1 (the main metrics that were broken)
                    cells[1].textContent = rowData.precision.toFixed(3);
                    cells[2].textContent = rowData.recall.toFixed(3); 
                    cells[3].textContent = rowData.f1.toFixed(3);
                    cells[3].style.backgroundColor = getPerformanceColor(rowData.f1);
                    // Update confusion matrix values
                    cells[4].textContent = rowData.tp;
                    cells[5].textContent = rowData.fd;
                    cells[6].textContent = rowData.fa;
                    cells[7].textContent = rowData.fn;
                }
            }
        });
    }
}

function updateConfusionMatrix(data) {
    document.querySelectorAll('.cm-cell').forEach(cell => {
        const label = cell.querySelector('.cm-label')?.textContent.toLowerCase();
        if (data[label]) {
            const valueElement = cell.querySelector('.cm-value');
            const percentageElement = cell.querySelector('.cm-percentage');
            
            if (valueElement) valueElement.textContent = data[label].value;
            if (percentageElement) percentageElement.textContent = data[label].percentage;
        }
    });
}

function updateNonMatchesTable(data) {
    const tableBody = document.querySelector('.non-matches-table tbody');
    if (tableBody && data) {
        tableBody.innerHTML = '';
        data.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.doc_id}</td>
                <td>${row.field_path}</td>
                <td>${row.non_match_type}</td>
                <td>${row.ground_truth_value}</td>
                <td>${row.prediction_value}</td>
            `;
            tableBody.appendChild(tr);
        });
    }
}

function updateDocumentImages(data) {
    const imageGallery = document.querySelector('.image-gallery');
    if (!imageGallery || !data) return;

    imageGallery.innerHTML = '';
    Object.entries(data).forEach(([docId, imagePath]) => {
        const imageItem = document.createElement('div');
        imageItem.className = 'image-item';
        imageItem.innerHTML = `
            <img src="${imagePath}" alt="${docId}" style="max-width: 200px;">
            <p><strong>${docId}</strong></p>
        `;
        imageGallery.appendChild(imageItem);
    });
    }
