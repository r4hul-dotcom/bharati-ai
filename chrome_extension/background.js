// background.js - Updated for Production

// Store API URL in Chrome storage (can be configured)
const API_BASE_URL = 'https://sales-dashboard.bharatifire.com';

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "classifyEmail") {
        classifyEmail(request.data, sendResponse);
        return true; // Keep channel open for async response
    }
});

async function classifyEmail(emailData, sendResponse) {
    try {
        // Get stored auth token from Chrome storage
        const result = await chrome.storage.sync.get(['authToken', 'userEmail']);
        
        if (!result.authToken) {
            console.error('Not authenticated. Please log in through the extension popup.');
            sendResponse({ 
                error: 'Not authenticated', 
                draft_html: '<p>Please log in to the Bharati AI extension first.</p>' 
            });
            return;
        }

        const response = await fetch(`${API_BASE_URL}/api/classify_and_draft`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${result.authToken}`
            },
            credentials: 'include',
            body: JSON.stringify(emailData)
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        console.log('Classification successful:', data.category);
        sendResponse(data);
        
    } catch (error) {
        console.error('Error calling API:', error);
        sendResponse({ 
            error: error.message,
            category: 'Error',
            draft_html: '<p>Could not connect to Bharati AI server. Please check your connection.</p>'
        });
    }
}
