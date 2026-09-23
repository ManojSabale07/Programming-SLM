(() => {
    'use strict';

    const workspace = document.getElementById('workspace');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const newChatButton = document.getElementById('newChatBtn');
    const searchInput = document.getElementById('chatSearch');
    const composer = document.getElementById('shellComposer');
    const composerInput = document.getElementById('composerInput');
    const composerSend = document.getElementById('composerSend');
    const composerStop = document.getElementById('composerStop');
    const banner = document.getElementById('workspaceBanner');
    const themeButton = document.querySelector('[title="Dark theme"]');

    if (!workspace || !sidebarToggle || !composer || !composerInput) {
        return;
    }

    const preferenceKey = 'programming-slm-shell';
    const preferences = JSON.parse(localStorage.getItem(preferenceKey) || '{}');
    let composing = false;

    function setSidebarCollapsed(collapsed) {
        workspace.classList.toggle('sidebar-collapsed', collapsed);
        sidebarToggle.textContent = collapsed ? '›' : '‹';
        sidebarToggle.title = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
        localStorage.setItem(preferenceKey, JSON.stringify({ ...preferences, sidebarCollapsed: collapsed }));
    }

    function setComposing(value) {
        composing = value;
        composerSend.disabled = value;
        composerStop.disabled = !value;
        composerInput.disabled = value;
        composer.classList.toggle('is-composing', value);
    }

    sidebarToggle.onclick = () => {
        setSidebarCollapsed(!workspace.classList.contains('sidebar-collapsed'));
    };

    if (preferences.sidebarCollapsed) {
        setSidebarCollapsed(true);
    }

    newChatButton?.addEventListener('click', () => {
        composerInput.value = '';
        composerInput.focus();
        document.querySelectorAll('.conversation-item').forEach(item => item.classList.remove('active'));
        document.querySelector('[data-conversation="current"]')?.classList.add('active');
        if (banner) {
            banner.querySelector('strong').textContent = 'Start a new coding session';
            banner.querySelector('.workspace-hint').textContent = 'Describe what you want to build';
        }
    });

    searchInput?.addEventListener('input', () => {
        const query = searchInput.value.trim().toLowerCase();
        document.querySelectorAll('.conversation-item').forEach(item => {
            item.hidden = query !== '' && !item.textContent.toLowerCase().includes(query);
        });
    });

    document.querySelectorAll('.conversation-item').forEach(item => {
        item.addEventListener('click', () => {
            document.querySelectorAll('.conversation-item').forEach(entry => entry.classList.remove('active'));
            item.classList.add('active');
        });
    });

    composer.addEventListener('submit', event => {
        event.preventDefault();
        if (!composerInput.value.trim() || composing) {
            composerInput.focus();
            return;
        }
        setComposing(true);
        if (banner) {
            banner.querySelector('strong').textContent = 'Prompt ready for Programming-SLM';
            banner.querySelector('.workspace-hint').textContent = 'Connect a backend to start generation';
        }
    });

    composerStop.addEventListener('click', () => {
        setComposing(false);
        composerInput.focus();
    });

    document.getElementById('composerSettings')?.addEventListener('click', () => {
        document.getElementById('paramsBtn')?.click();
    });

    themeButton?.addEventListener('click', () => {
        const light = document.body.classList.toggle('light-theme');
        themeButton.title = light ? 'Dark theme' : 'Light theme';
        localStorage.setItem(preferenceKey, JSON.stringify({ ...preferences, lightTheme: light }));
    });

    if (preferences.lightTheme) {
        document.body.classList.add('light-theme');
    }

    composerInput.addEventListener('keydown', event => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            composer.requestSubmit();
        }
    });
})();
