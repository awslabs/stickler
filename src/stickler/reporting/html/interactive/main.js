/**
 * Interactive functionality for HTML evaluation reports
 * Handles document drill-down, search, filtering, and navigation
 */

let documentData = [];
let fieldThresholds = {};
let currentDocIndex = -1;
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

// Initialize data from HTML (called by inline script)
function initializeDocumentData(docs, thresholds) {
    documentData = docs;
    fieldThresholds = thresholds;
    
    // Store original aggregate data on first load
    if (!aggregateData) {
        captureAggregateData();
        addReportFilterControls();
    }
}

// Capture the original aggregate data from the DOM
function captureAggregateData() {
    aggregateData = {
        executiveSummary: captureExecutiveSummaryData(),
        fieldAnalysis: captureFieldAnalysisData(),
        confusionMatrix: captureConfusionMatrixData(),
        nonMatches: captureNonMatchesData()
    };
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
        const filterControl = DOMUtils.createElement('div', 
            { className: 'report-filter-control' },
            '',
            `<div style="background: #f8f9fa; padding: 15px; border-radius: 6px; margin-bottom: 20px; border-left: 4px solid #007bff;">
                <strong>Report View:</strong>
                <select id="report-doc-filter" style="margin-left: 10px; padding: 5px 10px; border: 1px solid #dee2e6; border-radius: 4px;">
                    <option value="">All Documents (Aggregate)</option>
                </select>
                <button id="reset-report-filter" style="margin-left: 10px; padding: 5px 10px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer;">Reset to Aggregate</button>
            </div>`
        );
        
        firstSection.parentNode.insertBefore(filterControl, firstSection);
        
        // Populate document options
        const select = document.getElementById('report-doc-filter');
        if (select) {
            documentData.forEach(doc => {
                const option = DOMUtils.createElement('option', 
                    { value: doc.doc_id },
                    doc.doc_id
                );
                select.appendChild(option);
            });
        }
        
        // Add event listeners with error handling
        DOMUtils.addEventListenerSafe('#report-doc-filter', 'change', function() {
            filterReportToDocument(this.value || null);
        });
        
        DOMUtils.addEventListenerSafe('#reset-report-filter', 'click', function() {
            const select = document.getElementById('report-doc-filter');
            if (select) {
                select.value = '';
                filterReportToDocument(null);
            }
        });
    }
}


function setupSearch() {
    const searchInput = document.getElementById('doc-search');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            filterDocuments();
        });
    }
}

function setupFilter() {
    const filterSelect = document.getElementById('threshold-filter');
    if (filterSelect) {
        filterSelect.addEventListener('change', function() {
            filterDocuments();
        });
    }
}

function filterDocuments() {
    const searchInput = document.getElementById('doc-search');
    const filterSelect = document.getElementById('threshold-filter');
    
    if (!searchInput || !filterSelect) return;
    
    const searchTerm = searchInput.value.toLowerCase();
    const filterValue = filterSelect.value;
    const rows = document.querySelectorAll('.doc-row');
    
    rows.forEach(row => {
        const docId = row.getAttribute('data-doc-id').toLowerCase();
        const matchesSearch = docId.includes(searchTerm);
        
        let matchesFilter = true;
        if (filterValue === 'pass') {
            matchesFilter = row.classList.contains('threshold-pass');
        } else if (filterValue === 'fail') {
            matchesFilter = row.classList.contains('threshold-fail');
        }
        
        row.style.display = (matchesSearch && matchesFilter) ? '' : 'none';
    });
}

function showDocumentDetails(docId) {
    // Find document index
    currentDocIndex = documentData.findIndex(doc => doc.doc_id === docId);
    if (currentDocIndex === -1) return;
    
    const doc = documentData[currentDocIndex];
    const comparisonResult = doc.comparison_result;
    
    // Hide document list and show details
    const tableParent = document.querySelector('.document-table').parentElement;
    const controls = document.querySelector('.document-controls');
    const detailView = document.getElementById('document-detail');
    
    if (tableParent) tableParent.style.display = 'none';
    if (controls) controls.style.display = 'none';
    if (detailView) detailView.style.display = 'block';
    
    // Update title and navigation
    const titleElement = document.getElementById('detail-title');
    if (titleElement) {
        titleElement.textContent = `Document: ${docId}`;
    }
    
    updateNavigationButtons();
    
    // Generate detailed content
    const detailContent = generateDocumentDetailContent(comparisonResult, docId);
    const contentElement = document.getElementById('detail-content');
    if (contentElement) {
        contentElement.innerHTML = detailContent;
    }
}

function hideDocumentDetails() {
    const detailView = document.getElementById('document-detail');
    const tableParent = document.querySelector('.document-table')?.parentElement;
    const controls = document.querySelector('.document-controls');
    
    if (detailView) detailView.style.display = 'none';
    if (tableParent) tableParent.style.display = 'block';
    if (controls) controls.style.display = 'flex';
    
    currentDocIndex = -1;
}

function navigateDocument(direction) {
    const newIndex = currentDocIndex + direction;
    if (newIndex >= 0 && newIndex < documentData.length) {
        const newDoc = documentData[newIndex];
        showDocumentDetails(newDoc.doc_id);
    }
}

function updateNavigationButtons() {
    const prevButton = document.getElementById('prev-doc');
    const nextButton = document.getElementById('next-doc');
    
    if (prevButton) {
        prevButton.disabled = currentDocIndex <= 0;
    }
    
    if (nextButton) {
        nextButton.disabled = currentDocIndex >= documentData.length - 1;
    }
}

function generateDocumentDetailContent(comparisonResult, docId) {
    let html = `
        <div class="document-metrics">
            <h4>Overall Metrics</h4>
            <div class="metric-grid">
    `;
    
    // Overall similarity score
    const overallSimilarity = comparisonResult.similarity_score || 
                             comparisonResult.overall?.similarity_score || 0;
    html += `
        <div class="metric-item">
            <label>Overall Similarity:</label>
            <span class="metric-value">${overallSimilarity.toFixed(3)}</span>
        </div>
    `;
    
    html += `</div></div>`;
    
    // Field-level analysis - access fields from the correct nested structure
    const confusionMatrix = comparisonResult.confusion_matrix || {};
    const fieldsData = confusionMatrix.fields || {};
    if (Object.keys(fieldsData).length > 0) {
        html += `
            <div class="field-details">
                <h4>Field-by-Field Analysis</h4>
                <table class="field-detail-table">
                    <thead>
                        <tr>
                            <th>Field</th>
                            <th>Similarity Score</th>
                            <th>Threshold</th>
                            <th>Status</th>
                            <th>Ground Truth</th>
                            <th>Prediction</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        for (const [fieldName, fieldData] of Object.entries(fieldsData)) {
            const similarity = fieldData.raw_similarity_score || fieldData.similarity_score || 0;
            const threshold = fieldThresholds[fieldName] || 0.5;
            const status = similarity >= threshold ? '✓ Pass' : '✗ Fail';
            const statusClass = similarity >= threshold ? 'status-pass' : 'status-fail';
            
            // Extract ground truth and prediction values
            let gtValue = 'N/A';
            let predValue = 'N/A';
            
            // Try to get values from different possible locations
            if (fieldData.ground_truth_value !== undefined) {
                gtValue = String(fieldData.ground_truth_value).substring(0, 50);
            }
            if (fieldData.prediction_value !== undefined) {
                predValue = String(fieldData.prediction_value).substring(0, 50);
            }
            
            html += `
                <tr>
                    <td>${fieldName}</td>
                    <td>${similarity.toFixed(3)}</td>
                    <td>${threshold.toFixed(3)}</td>
                    <td class="${statusClass}">${status}</td>
                    <td class="value-cell" title="${gtValue}">${gtValue}</td>
                    <td class="value-cell" title="${predValue}">${predValue}</td>
                </tr>
            `;
        }
        
        html += `</tbody></table></div>`;
    }
    
    return html;
}

// Main function to filter the entire report to show metrics for a specific document
function filterReportToDocument(docId) {
    currentFilterDoc = docId;
    
    if (!docId) {
        // Show aggregate data
        updateExecutiveSummary(aggregateData.executiveSummary);
        updateFieldAnalysis(aggregateData.fieldAnalysis);
        updateConfusionMatrix(aggregateData.confusionMatrix);
        updateNonMatchesTable(aggregateData.nonMatches);
        updateReportTitle("All Documents");
    } else {
        // Find the document and show its data
        const doc = documentData.find(d => d.doc_id === docId);
        if (doc) {
            const docMetrics = extractDocumentMetrics(doc);
            updateExecutiveSummary(docMetrics.executiveSummary);
            updateFieldAnalysis(docMetrics.fieldAnalysis);
            updateConfusionMatrix(docMetrics.confusionMatrix);
            updateNonMatchesTable(docMetrics.nonMatches);
            updateReportTitle(`Document: ${docId}`);
        }
    }
}

function extractDocumentMetrics(doc) {
    const comparison = doc.comparison_result;
    const confusionMatrix = comparison.confusion_matrix || {};
    const overallMetrics = confusionMatrix.overall || {};
    const fieldsData = confusionMatrix.fields || {};
    const nonMatches = comparison.non_matches || []
    
    // Debug logging
    console.log('=== DEBUG: Document Metrics Extraction ===');
    console.log('Document ID:', doc.doc_id);
    console.log('doc.non_matches:', doc.non_matches);
    console.log('nonMatches array:', nonMatches);
    console.log('Full document structure:', doc);
    console.log('===========================================');
    
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
            chart: Object.entries(fieldsData).map(([field, data]) => ({
                field: field,
                value: data.derived?.cm_f1 || 0,
                width: `${Math.round((data.derived?.cm_f1 || 0) * 100)}%`,
                color: getPerformanceColor(data.derived?.cm_f1 || 0)
            })),
            table: Object.entries(fieldsData).map(([field, data]) => ({
                field: field,
                precision: data.derived?.cm_precision || 0,
                recall: data.derived?.cm_recall || 0,
                f1: data.derived?.cm_f1 || 0,
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
        }))
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

function updateFieldAnalysis(data) {
    // Update field chart
    if (data.chart) {
        const fieldBars = document.querySelectorAll('.field-bar');
        data.chart.forEach((fieldData, index) => {
            if (fieldBars[index]) {
                const barFill = fieldBars[index].querySelector('.bar-fill');
                const barValue = fieldBars[index].querySelector('.bar-value');
                
                if (barFill) {
                    barFill.style.width = fieldData.width;
                    barFill.style.backgroundColor = fieldData.color;
                }
                if (barValue) {
                    barValue.textContent = fieldData.value.toFixed(3);
                }
            }
        });
    }
    
    // Update performance table
    if (data.table) {
        const tableRows = document.querySelectorAll('.performance-table tbody tr');
        data.table.forEach((rowData, index) => {
            if (tableRows[index]) {
                const cells = tableRows[index].querySelectorAll('td');
                if (cells.length >= 8) {
                    cells[1].textContent = rowData.precision.toFixed(3);
                    cells[2].textContent = rowData.recall.toFixed(3);
                    cells[3].textContent = rowData.f1.toFixed(3);
                    cells[3].style.backgroundColor = getPerformanceColor(rowData.f1);
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

// Enhanced document interaction to also filter the main report
function setupDocumentInteraction() {
    // Handle document link clicks - now also filters the main report
    document.querySelectorAll('.doc-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const docId = this.getAttribute('data-doc-id');
            
            // Filter the main report to show this document's metrics
            filterReportToDocument(docId);
            
            // Update the report filter dropdown
            const reportFilter = document.getElementById('report-doc-filter');
            if (reportFilter) {
                reportFilter.value = docId;
            }
            
            // Also show the detailed view (existing functionality)
            showDocumentDetails(docId);
        });
    });
    
    // Handle "View Details" button clicks - show details without filtering main report
    document.querySelectorAll('.view-details-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            const docId = this.getAttribute('data-doc-id');
            showDocumentDetails(docId);
        });
    });
    
    // Handle back button
    const backButton = document.getElementById('back-to-list');
    if (backButton) {
        backButton.addEventListener('click', function() {
            hideDocumentDetails();
        });
    }
    
    // Handle navigation buttons
    const prevButton = document.getElementById('prev-doc');
    const nextButton = document.getElementById('next-doc');
    
    if (prevButton) {
        prevButton.addEventListener('click', function() {
            navigateDocument(-1);
        });
    }
    
    if (nextButton) {
        nextButton.addEventListener('click', function() {
            navigateDocument(1);
        });
    }
}
