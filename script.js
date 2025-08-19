function showTab(tabId) {
    // Hide all tab panels
    const tabPanels = document.querySelectorAll('.tab-panel');
    tabPanels.forEach(panel => {
        panel.classList.remove('active');
    });

    // Deactivate all tab buttons
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.classList.remove('active');
    });

    // Show the selected tab panel
    const selectedPanel = document.getElementById(tabId);
    if (selectedPanel) {
        selectedPanel.classList.add('active');
    }

    // Activate the clicked button
    const selectedButton = document.querySelector(`.tab-button[onclick="showTab('${tabId}')"]`);
    if (selectedButton) {
        selectedButton.classList.add('active');
    }
}

// Ensure the first tab is shown by default
document.addEventListener('DOMContentLoaded', () => {
    showTab('simulator');
});