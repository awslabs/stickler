// Skip N/A checkboxes when tabbing between input fields
const observer = new MutationObserver(() => {
    document.querySelectorAll('[data-testid="stCheckbox"] input').forEach(
        el => el.setAttribute('tabindex', '-1')
    );
});
observer.observe(document.body, {childList: true, subtree: true});
