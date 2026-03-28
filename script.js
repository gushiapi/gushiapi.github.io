// This function is now in the global scope, accessible by research.html
function showTab(tabId) {
    const tabPanels = document.querySelectorAll('.tab-panel');
    tabPanels.forEach(panel => {
        panel.classList.remove('active');
    });

    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.classList.remove('active');
        button.setAttribute('aria-selected', 'false');
    });

    const selectedPanel = document.getElementById(tabId);
    if (selectedPanel) {
        selectedPanel.classList.add('active');
    }

    const selectedButton = document.querySelector(`.tab-button[onclick="showTab('${tabId}')"]`);
    if (selectedButton) {
        selectedButton.classList.add('active');
        selectedButton.setAttribute('aria-selected', 'true');
    }
}

// This runs after the entire page has loaded
document.addEventListener('DOMContentLoaded', () => {
    // --- Update footer year ---
    const yearSpan = document.querySelector('.current-year');
    if (yearSpan) {
        yearSpan.textContent = new Date().getFullYear();
    }

    // --- Logic for the Shares page (shares.html) ---
    const filterTabs = document.querySelectorAll('.tab-btn');
    const cards = document.querySelectorAll('.card');
    
    if (filterTabs.length > 0) { // Only run this code if filter tabs exist
        filterTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const filter = tab.getAttribute('data-filter');

                filterTabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                cards.forEach(card => {
                    const cardType = card.getAttribute('data-type');
                    
                    if (filter === 'all' || cardType === filter) {
                        card.classList.remove('hidden');
                    } else {
                        card.classList.add('hidden');
                    }
                });
            });
        });
    }

    // --- Logic to initialize the Research page tabs ---
    if (document.querySelector('.tab-button[onclick]')) {
        showTab('plasticity'); // Set the default tab on the research page
    }
});