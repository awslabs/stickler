// Offline PDF viewer contract test. This is a DOM adapter, not a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const [, , source, docId, failure] = process.argv;
const elements = new Map();
const makeElement = () => ({
    style: {}, children: [], listeners: {}, disabled: false,
    appendChild(child) {
        child.parentElement = this;
        this.children.push(child);
        if (child.id) elements.set(child.id, child);
    },
    addEventListener(name, handler) { this.listeners[name] = handler; },
    getContext() { return {}; }
});
const document = {
    addEventListener() {},
    createElement: makeElement,
    getElementById: id => elements.get(id) || null,
    // Only simple ID selectors are supported. Raw punctuation is not an ID lookup.
    querySelector: selector => /^#[\p{L}\p{N}_-]+$/u.test(selector) ? elements.get(selector.slice(1)) || null : null
};
const container = makeElement();
for (const prefix of ['pdf-canvas-', 'pdf-loading-', 'pdf-error-']) {
    const element = makeElement();
    element.id = prefix + docId;
    element.style.display = prefix === 'pdf-loading-' ? 'block' : 'none';
    container.appendChild(element);
}
const rendered = [];
const loads = [];
const pdf = {
    numPages: 2,
    getPage: async pageNumber => ({
        getViewport: ({scale}) => ({width: 400 * scale, height: 600 * scale}),
        render: () => ({promise: Promise.resolve().then(() => rendered.push(pageNumber))})
    })
};
const context = vm.createContext({
    document,
    window: {},
    console: {error() {}},
    pdfjsLib: {
        getDocument(options) {
            loads.push(options);
            return {promise: failure === 'True' ? Promise.reject(new Error('load failed')) : Promise.resolve(pdf)};
        }
    }
});
vm.runInContext(fs.readFileSync(source, 'utf8'), context);
context.args = ['fixture.pdf', 'pdf-canvas-' + docId, 'pdf-loading-' + docId, 'pdf-error-' + docId];
const settle = () => new Promise(resolve => setImmediate(resolve));

(async () => {
    vm.runInContext('loadPDF(...args)', context);
    await settle();
    assert.equal(loads.length, 1, 'the PDF loader should be reached for this document ID');
    assert.equal(loads[0].isEvalSupported, false);
    assert.equal(elements.get('pdf-loading-' + docId).style.display, 'none');
    const canvas = elements.get('pdf-canvas-' + docId);
    if (failure === 'True') {
        assert.equal(elements.get('pdf-error-' + docId).style.display, 'block');
        assert.equal(canvas.style.display, 'none');
        assert.deepEqual(rendered, []);
        return;
    }
    assert.equal(canvas.style.display, 'block');
    assert.deepEqual(rendered, [1]);
    const prev = elements.get('prev-' + docId);
    const next = elements.get('next-' + docId);
    const info = elements.get('page-info-' + docId);
    assert.equal(info.textContent, 'Page 1 of 2');
    assert.equal(prev.disabled, true);
    assert.equal(next.disabled, false);
    next.listeners.click();
    await settle();
    assert.deepEqual(rendered, [1, 2]);
    assert.equal(info.textContent, 'Page 2 of 2');
    assert.equal(prev.disabled, false);
    assert.equal(next.disabled, true);
    prev.listeners.click();
    await settle();
    assert.deepEqual(rendered, [1, 2, 1]);
    assert.equal(info.textContent, 'Page 1 of 2');
    assert.equal(prev.disabled, true);
    assert.equal(next.disabled, false);
})().catch(error => { console.error(error); process.exitCode = 1; });
