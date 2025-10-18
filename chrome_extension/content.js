// START OF FILE: content.js (Automatic Draft-on-Open Version)

console.log("Bharati AI Content Script Loaded.");

// --- 1. The Brain: The Observer ---
const observer = new MutationObserver(mutations => {
    mutations.forEach(mutation => {
        if (mutation.addedNodes) {
            mutation.addedNodes.forEach(node => {
                if (node.nodeType === 1 && node.querySelector('.hP')) {
                    console.log("Email view detected. Triggering classification.");
                    handleEmailOpen(node);
                }
            });
        }
    });
});

// --- 2. The Configuration ---
observer.observe(document.body, {
    childList: true,
    subtree: true
});

// --- 3. The Action: What to do when an email is opened ---
function handleEmailOpen(emailContainer) {
    if (emailContainer.dataset.bharatiProcessed === 'true') {
        return;
    }
    emailContainer.dataset.bharatiProcessed = 'true';

    const subjectElement = emailContainer.querySelector('.hP');
    // Try multiple body selectors
    let emailBodyElement = emailContainer.querySelector('.a3s.aiL');
    if (!emailBodyElement) {
        emailBodyElement = emailContainer.querySelector('.adn.ads');
    }
    if (!emailBodyElement) {
        emailBodyElement = emailContainer.querySelector('[data-message-id]');
    }

    if (!subjectElement || !emailBodyElement) {
        console.log("Could not find subject or body element.");
        return;
    }

    const subject = subjectElement.innerText;
    const emailBody = emailBodyElement.innerText;

    console.log(`Sending to API - Subject: ${subject}`);

    chrome.runtime.sendMessage({
        action: "classifyEmail",
        data: { email_body: emailBody, subject: subject }
    }, response => {
        if (chrome.runtime.lastError) {
            console.error("Chrome runtime error:", chrome.runtime.lastError.message);
            return;
        }
        if (!response) {
            console.error("No response received from background script");
            return;
        }
        console.log("Received response from background script:", response);
        if (response.draft_html) {
            injectDraftReply(emailContainer, response.draft_html);
        } else {
            console.error("No draft_html in response");
        }
    });
}

// --- 4. The Result: Injecting the draft into the reply box ---
function injectDraftReply(emailContainer, draftHtml) {
    console.log("Starting draft injection process...");
    
    // Try multiple reply button selectors
    const replySelectors = [
        'span[role="link"][data-tooltip^="Reply"]',
        'div[role="button"][data-tooltip^="Reply"]',
        'span[role="button"][data-tooltip="Reply"]',
        '[aria-label="Reply"]',
        '.T-I.J-J5-Ji.T-I-Js-IF.aaq.T-I-ax7'
    ];
    
    let replyButton = null;
    for (const selector of replySelectors) {
        replyButton = emailContainer.querySelector(selector);
        if (replyButton) {
            console.log(`Found reply button with selector: ${selector}`);
            break;
        }
    }
    
    if (!replyButton) {
        console.error("Could not find reply button");
        return;
    }
    
    replyButton.click();
    console.log("Clicked reply button, waiting for editor...");

    // Poll for the editor
    let attempts = 0;
    const maxAttempts = 20;
    
    const findEditor = setInterval(() => {
        attempts++;
        
        const editorSelectors = [
            'div[aria-label="Message Body"]',
            'div[g_editable="true"]',
            'div.Am.Al.editable',
            'div[contenteditable="true"][role="textbox"]',
            'div[contenteditable="true"][aria-label*="Message"]'
        ];
        
        let replyEditor = null;
        for (const selector of editorSelectors) {
            replyEditor = document.querySelector(selector);
            if (replyEditor) {
                console.log(`Found editor with selector: ${selector}`);
                break;
            }
        }
        
        if (replyEditor) {
            clearInterval(findEditor);
            console.log("Injecting draft into reply editor.");
            
            // Clear existing content first
            replyEditor.innerHTML = '';
            
            // Insert new content
            replyEditor.innerHTML = draftHtml;
            
            // Trigger events to make Gmail recognize the content
            replyEditor.focus();
            const inputEvent = new Event('input', { bubbles: true });
            replyEditor.dispatchEvent(inputEvent);
            
            console.log("Draft injected successfully!");
        } else if (attempts >= maxAttempts) {
            clearInterval(findEditor);
            console.error("Could not find reply editor after", maxAttempts, "attempts");
        } else {
            console.log(`Attempt ${attempts}/${maxAttempts} - Editor not found yet...`);
        }
    }, 250);
}
