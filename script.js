document.addEventListener('DOMContentLoaded', () => {
    const tabs = document.querySelectorAll('.tab-btn');
    const cards = document.querySelectorAll('.card');
    const grid = document.querySelector('.content-grid');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const filter = tab.getAttribute('data-filter');

            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            cards.forEach(card => {
                const cardType = card.getAttribute('data-type');
                
                if (filter === 'all' || cardType === filter) {
                    card.classList.remove('hidden');
                } else {
                    card.classList.add('hidden');
                }
            });

            if(grid) {
                grid.style.height = 'auto';
            }
        });
    });

    // Previous script functionality for other pages
    function showTab(tabId) {
        const tabPanels = document.querySelectorAll('.tab-panel');
        tabPanels.forEach(panel => {
            panel.classList.remove('active');
            panel.style.display = 'none';
        });

        const tabButtons = document.querySelectorAll('.tab-button');
        tabButtons.forEach(button => {
            button.classList.remove('active');
        });

        const selectedPanel = document.getElementById(tabId);
        if (selectedPanel) {
            selectedPanel.classList.add('active');
            selectedPanel.style.display = 'block';
        }

        const selectedButton = document.querySelector(`.tab-button[onclick="showTab('${tabId}')"]`);
        if (selectedButton) {
            selectedButton.classList.add('active');
        }
    }

    if (document.querySelector('.tab-button[onclick]')) {
        showTab('plasticity');
    }
});