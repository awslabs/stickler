/**
 * @fileoverview Interactive functionality for HTML evaluation reports
 * Optimized single-file version with helper functions and reduced duplication
 * 
 * Global State Variables:
 * @var {Array<Object>} documentData - Array of individual document results from JSONL file
 * @var {Object} fieldThresholds - Field name -> threshold value mapping extracted from model schema  
 * @var {Object} aggregateData - Original aggregate metrics captured from DOM on page load
 * @var {string|null} currentFilterDoc - Current filtered document ID (null = showing aggregate view)
 */

// ============================================================================
// GLOBAL STATE & UTILITIES
// ============================================================================

let documentData = [];
let fieldThresholds = {};
let aggregateData = null;
let currentFilterDoc = null;

// DOM helpers
const getElement = (selector) => document.querySelector(selector);
const getElements = (selector) => document.querySelectorAll(selector);
const setText = (selectorOrElement, text) => { 
    const el = typeof selectorOrElement === 'string' ? getElement(selectorOrElement) : selectorOrElement; 
    if (el) el.textContent = text; 
};
const setStyle = (selector, styles) => { const el = getElement(selector); if (el) Object.assign(el.style, styles); };
const createElement = (tag, className, innerHTML) => {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (innerHTML) el.innerHTML = innerHTML;
    return el;
};

// Utility functions
const getPerformanceColor = (value) => value >= 0.8 ? '#28a745' : value >= 0.6 ? '#ffc107' : '#dc3545';
const calculatePercentage = (value, total) => Math.round((value / Math.max(1, total)) * 100);
const formatMetricValue = (value) => typeof value === 'number' ? value.toFixed(3) : value;
const detectFileType = (filePath) => filePath.split('.').pop().toLowerCase() === 'pdf' ? 'pdf' : 'image';
const safeGet = (obj, path, defaultValue = null) => 
    path.split('.').reduce((current, key) => current && current[key] !== undefined ? current[key] : defaultValue, obj);
const ensureNumber = (value, defaultValue = 0) => typeof value === 'number' && !isNaN(value) ? value : defaultValue;

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    if (getElement('#individual-documents')) {
        console.log('Document interaction setup');
    }
});

function initializeDocumentData(docs, thresholds) {
    documentData = docs;
    fieldThresholds = thresholds;
    
    if (!aggregateData) {
        aggregateData = captureAggregateData();
        addReportFilterControls();
    }
}

// ============================================================================
// DATA CAPTURE (OPTIMIZED)
// ============================================================================

const captureAggregateData = () => ({
    executiveSummary: captureExecutiveSummaryData(),
    fieldAnalysis: captureFieldAnalysisData(),
    confusionMatrix: captureConfusionMatrixData(),
    nonMatches: captureNonMatchesData(),
    documentFiles: getElement('.document-gallery') ? captureDocumentFileData() : null
});

const captureExecutiveSummaryData = () => {
    const gaugeValue = getElement('.gauge-value')?.textContent;
    const metrics = {};
    
    getElements('.metric-card').forEach(card => {
        const label = card.querySelector('.metric-label')?.textContent.toLowerCase().replace(' ', '_');
        const value = card.querySelector('.metric-value')?.textContent;
        if (label && value && label !== 'documents') {
            metrics[label] = parseFloat(value) || value;
        }
    });
    
    return {
        gaugeValue: gaugeValue ? parseFloat(gaugeValue.replace('%', '')) / 100 : null,
        metrics
    };
};

const captureFieldAnalysisData = () => ({
    chart: Array.from(getElements('.field-bar')).map(bar => ({
        field: bar.querySelector('.field-label')?.textContent,
        value: parseFloat(bar.querySelector('.bar-value')?.textContent),
        width: bar.querySelector('.bar-fill')?.style.width,
        color: bar.querySelector('.bar-fill')?.style.backgroundColor
    })).filter(item => item.field && !isNaN(item.value)),
    
    table: Array.from(getElements('#performance-table tbody tr')).map(row => {
        const cells = row.querySelectorAll('td');
        return cells.length >= 8 ? {
            field: cells[0].textContent,
            precision: parseFloat(cells[1].textContent),
            recall: parseFloat(cells[2].textContent),
            f1: parseFloat(cells[3].textContent),
            tp: parseInt(cells[4].textContent),
            fd: parseInt(cells[5].textContent),
            fa: parseInt(cells[6].textContent),
            fn: parseInt(cells[7].textContent)
        } : null;
    }).filter(Boolean)
});

const captureConfusionMatrixData = () => {
    const data = {};
    getElements('.cm-cell').forEach(cell => {
        const label = cell.querySelector('.cm-label')?.textContent;
        const value = cell.querySelector('.cm-value')?.textContent;
        const percentage = cell.querySelector('.cm-percentage')?.textContent;
        
        if (label && value) {
            data[label.toLowerCase()] = { value: parseInt(value), percentage };
        }
    });
    return data;
};

const captureNonMatchesData = () => 
    Array.from(getElements('#non-matches-table tbody tr')).map(row => {
        const cells = row.querySelectorAll('td');
        return cells.length >= 5 ? {
            doc_id: cells[0].textContent,
            field_path: cells[1].textContent,
            non_match_type: cells[2].textContent,
            ground_truth_value: cells[3].textContent,
            prediction_value: cells[4].textContent
        } : null;
    }).filter(Boolean);

const captureDocumentFileData = () => {
    const data = {};
    getElements('.image-item').forEach(image => {
        const imgElement = image.querySelector('img')?.src;
        const docId = image.querySelector('p strong')?.textContent;
        if (imgElement && docId) data[docId] = imgElement;
    });
    getElements('.pdf-item').forEach(pdfItem => {
        const pdfPath = pdfItem.getAttribute('data-pdf-path');
        const docId = pdfItem.querySelector('p strong')?.textContent;
        if (pdfPath && docId) data[docId] = pdfPath;
    });
    return data;
};

// ============================================================================
// FILTER CONTROLS
// ============================================================================

const addReportFilterControls = () => {
    const firstSection = getElement('.section');
    if (!firstSection) return;
    
    // Check if document files exist before adding the document toggle button
    const hasDocumentFiles = aggregateData?.documentFiles && Object.keys(aggregateData.documentFiles).length > 0;
    
    const documentButtonHtml = hasDocumentFiles ? 
        '<button id="document-toggle-btn" class="btn btn-primary">Show Documents</button>' : '';
    
    const filterControl = createElement('div', 'report-filter-control', `
        <div style="background: #f8f9fa; padding: 15px; border-radius: 6px; margin-bottom: 20px; border-left: 4px solid #007bff;">
            <strong>Report View:</strong>
            <select id="report-doc-filter" style="margin-left: 10px; padding: 5px 10px; border: 1px solid #dee2e6; border-radius: 4px;">
                <option value="">All Documents (Aggregate)</option>
            </select>
            <button id="reset-report-filter" class="btn btn-secondary" style="margin-left: 10px;">Reset to Aggregate</button>
            ${documentButtonHtml}
        </div>
    `);
    
    firstSection.parentNode.insertBefore(filterControl, firstSection);
    
    // Populate options and setup events
    const select = getElement('#report-doc-filter');
    documentData.forEach(doc => {
        const option = createElement('option');
        option.value = doc.doc_id;
        option.textContent = doc.doc_id;
        select.appendChild(option);
    });
    
    select.addEventListener('change', function() { filterReportToDocument(this.value || null); });
    getElement('#reset-report-filter').addEventListener('click', function() {
        select.value = '';
        filterReportToDocument(null);
    });
    
    // Only add event listener if the button exists
    const documentToggleBtn = getElement('#document-toggle-btn');
    if (documentToggleBtn) {
        documentToggleBtn.addEventListener('click', toggleDocumentVisibility);
    }
    
    updateDocumentButtonState();
    setupMainContentLayout();
};

// ============================================================================
// MAIN FILTERING & UPDATES
// ============================================================================

const filterReportToDocument = (docId) => {
    currentFilterDoc = docId;
    
    if (!docId) {
        updateAllSections(aggregateData, "All Documents");
        const documentGallery = getElement('.document-gallery');
        const mainContent = getElement('.main-content');
        if (documentGallery && mainContent) {
            documentGallery.classList.remove('visible');
            mainContent.classList.remove('documents-visible');
        }
    } else {
        const doc = documentData.find(d => d.doc_id === docId);
        if (doc) {
            const docMetrics = extractDocumentMetrics(doc);
            updateAllSections(docMetrics, `Document: ${docId}`);
        }
    }
    updateDocumentButtonState();
};

const extractDocumentMetrics = (doc) => {
    const comparison = doc.comparison_result;
    const confusionMatrix = comparison.confusion_matrix || {};
    const overallMetrics = confusionMatrix.overall || {};
    const fieldsData = confusionMatrix.fields || {};
    const nonMatches = comparison.non_matches || [];
    
    const tp = ensureNumber(overallMetrics.tp);
    const tn = ensureNumber(overallMetrics.tn);
    const fd = ensureNumber(overallMetrics.fd);
    const fa = ensureNumber(overallMetrics.fa);
    const fn = ensureNumber(overallMetrics.fn);
    const total = tp + fd + fa + fn;
    
    return {
        executiveSummary: {
            gaugeValue: safeGet(overallMetrics, 'derived.cm_f1', 0),
            metrics: {
                precision: safeGet(overallMetrics, 'derived.cm_precision', 0),
                recall: safeGet(overallMetrics, 'derived.cm_recall', 0),
                f1: safeGet(overallMetrics, 'derived.cm_f1', 0),
                accuracy: safeGet(overallMetrics, 'derived.cm_accuracy', 0)
            }
        },
        fieldAnalysis: {
            chart: Object.entries(fieldsData).map(([field, data]) => {
                const f1Value = safeGet(data, 'overall.derived.cm_f1', 0);
                return { field, value: f1Value, width: `${Math.round(f1Value * 100)}%`, color: getPerformanceColor(f1Value) };
            }),
            table: Object.entries(fieldsData).map(([field, data]) => ({
                field,
                precision: safeGet(data, 'overall.derived.cm_precision', 0),
                recall: safeGet(data, 'overall.derived.cm_recall', 0),
                f1: safeGet(data, 'overall.derived.cm_f1', 0),
                tp: safeGet(data, 'overall.tp', 0),
                fd: safeGet(data, 'overall.fd', 0),
                fa: safeGet(data, 'overall.fa', 0),
                fn: safeGet(data, 'overall.fn', 0)
            }))
        },
        confusionMatrix: {
            tp: { value: tp, percentage: `${calculatePercentage(tp, total)}%` },
            tn: { value: tn, percentage: `${calculatePercentage(tn, total)}%` },
            fd: { value: fd, percentage: `${calculatePercentage(fd, total)}%` },
            fa: { value: fa, percentage: `${calculatePercentage(fa, total)}%` },
            fn: { value: fn, percentage: `${calculatePercentage(fn, total)}%` }
        },
        nonMatches: nonMatches.map(nonMatch => ({
            doc_id: doc.doc_id,
            field_path: nonMatch.field_path,
            non_match_type: nonMatch.non_match_type,
            ground_truth_value: nonMatch.ground_truth_value,
            prediction_value: nonMatch.prediction_value
        })),
        documentFiles: aggregateData?.documentFiles?.[doc.doc_id] ? {[doc.doc_id]: aggregateData.documentFiles[doc.doc_id]} : {}
    };
};

const updateAllSections = (data, title) => {
    updateExecutiveSummary(data.executiveSummary);
    updateFieldAnalysis(data.fieldAnalysis);
    updateConfusionMatrix(data.confusionMatrix);
    updateNonMatchesTable(data.nonMatches);
    updateDocumentFiles(data.documentFiles);
    updateReportTitle(title);
};

const updateExecutiveSummary = (data) => {
    if (data.gaugeValue !== undefined) {
        const percentage = Math.round(data.gaugeValue * 100);
        const color = getPerformanceColor(data.gaugeValue);
        setStyle('.gauge-circle', { background: `conic-gradient(${color} ${percentage}%, #e9ecef ${percentage}%)` });
        setText('.gauge-value', `${percentage}%`);
    }
    
    if (data.metrics) {
        getElements('.metric-card').forEach(card => {
            const label = card.querySelector('.metric-label')?.textContent.toLowerCase().replace(' ', '_');
            const valueElement = card.querySelector('.metric-value');
            if (label && data.metrics[label] !== undefined && valueElement) {
                const value = data.metrics[label];
                valueElement.textContent = formatMetricValue(value);
                valueElement.style.color = getPerformanceColor(parseFloat(value) || 0);
            }
        });
    }
};

const updateFieldAnalysis = (data) => {
    if (data.chart) {
        getElements('.field-bar').forEach(bar => {
            const fieldLabel = bar.querySelector('.field-label')?.textContent;
            const fieldData = data.chart.find(item => item.field === fieldLabel);
            if (fieldData) {
                const barFill = bar.querySelector('.bar-fill');
                const barValue = bar.querySelector('.bar-value');
                if (barFill) {
                    barFill.style.width = fieldData.width;
                    barFill.style.backgroundColor = fieldData.color;
                }
                if (barValue) barValue.textContent = formatMetricValue(fieldData.value);
            }
        });
    }
    
    if (data.table) {
        getElements('#performance-table tbody tr').forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length >= 8) {
                const fieldName = cells[0].textContent;
                const rowData = data.table.find(item => item.field === fieldName);
                if (rowData) {
                    cells[1].textContent = formatMetricValue(rowData.precision);
                    cells[2].textContent = formatMetricValue(rowData.recall);
                    cells[3].textContent = formatMetricValue(rowData.f1);
                    cells[3].style.backgroundColor = getPerformanceColor(rowData.f1);
                    cells[4].textContent = rowData.tp;
                    cells[5].textContent = rowData.fd;
                    cells[6].textContent = rowData.fa;
                    cells[7].textContent = rowData.fn;
                }
            }
        });
    }
};

const updateConfusionMatrix = (data) => {
    getElements('.cm-cell').forEach(cell => {
        const label = cell.querySelector('.cm-label')?.textContent.toLowerCase();
        const cellData = data[label];
        if (cellData) {
            setText(cell.querySelector('.cm-value'), cellData.value);
            setText(cell.querySelector('.cm-percentage'), cellData.percentage);
        }
    });
};

const updateNonMatchesTable = (data) => {
    const tableBody = getElement('#non-matches-table tbody');
    if (!tableBody || !data) return;
    
    tableBody.innerHTML = '';
    data.forEach(row => {
        const tr = createElement('tr', '', `
            <td>${row.doc_id}</td>
            <td>${row.field_path}</td>
            <td>${row.non_match_type}</td>
            <td>${row.ground_truth_value}</td>
            <td>${row.prediction_value}</td>
        `);
        tableBody.appendChild(tr);
    });
};

const updateDocumentFiles = (data) => {
    const documentGallery = getElement('.document-gallery');
    if (!documentGallery || !data) return;

    documentGallery.innerHTML = '';
    Object.entries(data).forEach(([docId, filePath]) => {
        const fileType = detectFileType(filePath);
        if (fileType === 'pdf') {
            const pdfItem = createElement('div', 'pdf-item');
            pdfItem.setAttribute('data-doc-id', docId);
            pdfItem.setAttribute('data-pdf-path', filePath);
            pdfItem.innerHTML = `
                <div class="pdf-container">
                    <canvas id="pdf-canvas-${docId}" class="pdf-canvas"></canvas>
                    <div class="pdf-loading" id="pdf-loading-${docId}">Loading PDF...</div>
                    <div class="pdf-error" id="pdf-error-${docId}" style="display: none;">Error loading PDF</div>
                </div>
                <p><strong>${docId}</strong></p>
            `;
            documentGallery.appendChild(pdfItem);
            setTimeout(() => loadPDF(filePath, `pdf-canvas-${docId}`, `pdf-loading-${docId}`, `pdf-error-${docId}`), 0);
        } else {
            const imageItem = createElement('div', 'image-item', `
                <img src="${filePath}" alt="${docId}">
                <p><strong>${docId}</strong></p>
            `);
            documentGallery.appendChild(imageItem);
        }
    });
};

const updateReportTitle = (title) => {
    const header = getElement('header h1');
    if (header) {
        const originalTitle = header.textContent.split(' - ')[0];
        header.textContent = `${originalTitle} - ${title}`;
    }
};

// ============================================================================
// DOCUMENT VIEWER & PDF HANDLING
// ============================================================================

const setupMainContentLayout = () => {
    const main = getElement('main');
    if (!main) return;
    
    const mainContent = createElement('div', 'main-content');
    const metricsColumn = createElement('div', 'metrics-column');
    const documentColumn = createElement('div', 'documents-column');
    
    Array.from(main.children).forEach(section => {
        const documentGallery = section.querySelector('.document-gallery');
        if (documentGallery) {
            documentColumn.appendChild(documentGallery);
            section.remove();
        } else {
            metricsColumn.appendChild(section);
        }
    });
    
    mainContent.appendChild(metricsColumn);
    mainContent.appendChild(documentColumn);
    main.appendChild(mainContent);
};

const toggleDocumentVisibility = () => {
    if (!currentFilterDoc) return;
    
    const documentGallery = getElement('.document-gallery');
    const mainContent = getElement('.main-content');
    const toggleBtn = getElement('#document-toggle-btn');
    
    if (!documentGallery || !toggleBtn || !mainContent) return;
    
    const isVisible = documentGallery.classList.contains('visible');
    
    if (isVisible) {
        documentGallery.classList.remove('visible');
        mainContent.classList.remove('documents-visible');
        toggleBtn.textContent = 'Show Documents';
        toggleBtn.className = 'btn btn-primary';
    } else {
        documentGallery.classList.add('visible');
        mainContent.classList.add('documents-visible');
        toggleBtn.textContent = 'Hide Documents';
        toggleBtn.className = 'btn btn-success';
    }
};

const updateDocumentButtonState = () => {
    const toggleBtn = getElement('#document-toggle-btn');
    // If button doesn't exist (no document files), just return
    if (!toggleBtn) return;
    
    if (!currentFilterDoc) {
        toggleBtn.disabled = true;
        toggleBtn.textContent = 'Documents (Select Individual Document)';
        toggleBtn.title = 'Document viewing is only available when filtering to a specific document';
        toggleBtn.classList.remove('active');
    } else {
        toggleBtn.disabled = false;
        toggleBtn.title = 'Toggle document viewer for the selected document';
        
        const documentGallery = getElement('.document-gallery');
        const isVisible = documentGallery && documentGallery.classList.contains('visible');
        
        toggleBtn.textContent = isVisible ? 'Hide Documents' : 'Show Documents';
        toggleBtn.className = isVisible ? 'btn btn-success' : 'btn btn-primary';
    }
};

const loadPDF = (url, canvasId, loadingId, errorId) => {
    const canvas = getElement(`#${canvasId}`);
    const loading = getElement(`#${loadingId}`);
    const error = getElement(`#${errorId}`);
    
    if (!canvas || !loading || !error) return;
    
    const context = canvas.getContext('2d');
    const docId = canvasId.replace('pdf-canvas-', '');
    
    pdfjsLib.getDocument(url).promise.then(pdf => {
        const pdfContainer = canvas.parentElement;
        const navControls = createElement('div', 'pdf-navigation', `
            <button class="btn btn-primary pdf-nav-btn" id="prev-${docId}" onclick="navigatePDF('${docId}', -1)">← Previous</button>
            <span class="pdf-page-info" id="page-info-${docId}">Page 1 of ${pdf.numPages}</span>
            <button class="btn btn-primary pdf-nav-btn" id="next-${docId}" onclick="navigatePDF('${docId}', 1)">Next →</button>
        `);
        pdfContainer.appendChild(navControls);
        
        window.pdfData = window.pdfData || {};
        window.pdfData[docId] = { doc: pdf, currentPage: 1, canvas: canvas, context: context };
        
        renderPDFPage(docId, 1);
        updateNavigationButtons(docId);
        
    }).catch(err => {
        loading.style.display = 'none';
        error.style.display = 'block';
        console.error('Error loading PDF:', err);
    });
};

const renderPDFPage = (docId, pageNum) => {
    const pdfInfo = window.pdfData?.[docId];
    if (!pdfInfo) return;
    
    pdfInfo.doc.getPage(pageNum).then(page => {
        const viewport = page.getViewport({scale: 1.0});
        const maxWidth = 800;
        const scale = Math.min(maxWidth / viewport.width, 2.5);
        const scaledViewport = page.getViewport({scale: scale});
        
        pdfInfo.canvas.height = scaledViewport.height;
        pdfInfo.canvas.width = scaledViewport.width;
        
        page.render({ canvasContext: pdfInfo.context, viewport: scaledViewport }).promise.then(() => {
            const loading = getElement(`#pdf-loading-${docId}`);
            if (loading) loading.style.display = 'none';
            pdfInfo.canvas.style.display = 'block';
            setText(`#page-info-${docId}`, `Page ${pageNum} of ${pdfInfo.doc.numPages}`);
        });
    });
};

const navigatePDF = (docId, direction) => {
    const pdfInfo = window.pdfData?.[docId];
    if (!pdfInfo) return;
    
    const newPage = pdfInfo.currentPage + direction;
    const totalPages = pdfInfo.doc.numPages;
    
    if (newPage >= 1 && newPage <= totalPages) {
        pdfInfo.currentPage = newPage;
        renderPDFPage(docId, newPage);
        updateNavigationButtons(docId);
    }
};

const updateNavigationButtons = (docId) => {
    const pdfInfo = window.pdfData?.[docId];
    if (!pdfInfo) return;
    
    const prevBtn = getElement(`#prev-${docId}`);
    const nextBtn = getElement(`#next-${docId}`);
    
    if (prevBtn) prevBtn.disabled = (pdfInfo.currentPage <= 1);
    if (nextBtn) nextBtn.disabled = (pdfInfo.currentPage >= pdfInfo.doc.numPages);
};

// Global exports
window.initializeDocumentData = initializeDocumentData;
window.navigatePDF = navigatePDF;
