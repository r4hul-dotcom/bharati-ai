// popup.js - Handles authentication

const API_BASE_URL = 'https://sales-dashboard.bharatifire.com';

document.addEventListener('DOMContentLoaded', async function() {
    // Check if already logged in
    const result = await chrome.storage.sync.get(['authToken', 'userEmail']);
    
    if (result.authToken) {
        showLoggedInState(result.userEmail);
    } else {
        showLoginForm();
    }
    
    // Login button handler
    document.getElementById('loginBtn').addEventListener('click', login);
    
    // Logout button handler
    document.getElementById('logoutBtn').addEventListener('click', logout);
});

function showLoginForm() {
    document.getElementById('loginSection').style.display = 'block';
    document.getElementById('statusSection').style.display = 'none';
}

function showLoggedInState(email) {
    document.getElementById('loginSection').style.display = 'none';
    document.getElementById('statusSection').style.display = 'block';
    document.getElementById('userEmail').textContent = email;
}

async function login() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    
    if (!email || !password) {
        showMessage('Please enter email and password', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/extension_login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        
        if (response.ok && data.token) {
            // Store token in Chrome storage
            await chrome.storage.sync.set({
                authToken: data.token,
                userEmail: email
            });
            
            showMessage('Login successful!', 'success');
            showLoggedInState(email);
        } else {
            showMessage(data.error || 'Login failed', 'error');
        }
    } catch (error) {
        showMessage('Could not connect to server', 'error');
        console.error(error);
    }
}

async function logout() {
    await chrome.storage.sync.remove(['authToken', 'userEmail']);
    showMessage('Logged out', 'success');
    showLoginForm();
}

function showMessage(text, type) {
    const msgDiv = document.getElementById('message');
    msgDiv.textContent = text;
    msgDiv.style.color = type === 'error' ? '#dc3545' : '#28a745';
    msgDiv.style.padding = '10px';
    
    setTimeout(() => {
        msgDiv.textContent = '';
    }, 3000);
}
