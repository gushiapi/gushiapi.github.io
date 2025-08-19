function showTab(tabId) {
    // Hide all tab panels and remove active class
    const tabPanels = document.querySelectorAll('.tab-panel');
    tabPanels.forEach(panel => {
        panel.classList.remove('active');
        panel.style.display = 'none';
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
        selectedPanel.style.display = 'block';
    }

    // Activate the clicked button
    const selectedButton = document.querySelector(`.tab-button[onclick="showTab('${tabId}')"]`);
    if (selectedButton) {
        selectedButton.classList.add('active');
    }
}

// Ensure the 'plasticity' tab is shown by default
document.addEventListener('DOMContentLoaded', () => {
    showTab('plasticity');
});