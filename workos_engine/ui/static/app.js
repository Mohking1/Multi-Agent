/**
 * Argus OS — Executive AI Operating System
 * Frontend Controller (Alpine.js) with SSE Live Telemetry,
 * Slide-Over Drawer, and Command Palette.
 */

function argusApp() {
  return {
    // --- System & Telemetry State ---
    status: {
      model_name: 'refinedneuro/refinedtoolcallv5-3b',
      autonomy_level: 'SUPERVISED',
      ollama_online: false,
      num_ctx: 6144,
      version: '1.0.0',
    },
    stats: {
      vault: { document_count: 0, parsed_count: 0, table_count: 0, total_size_formatted: '0 KB' },
      memory: { active_beliefs_count: 0, wings_count: 0, halls_count: 0 },
      mail: { connected: false, autonomy_level: 'SUPERVISED' },
      system: { system_name: 'Argus OS', model_name: '', ollama_online: false },
    },
    activeWorkspace: 'chat', // 'chat' | 'vault' | 'mail' | 'memory'
    sidebarCollapsed: false,

    // --- Session & Conversation State ---
    sessions: [],
    currentSessionId: 'session_default',
    currentSessionTitle: 'New Executive Workspace',
    chatMessages: [],
    chatInput: '',
    isThinking: false,
    thinkingTicker: 'Formulating execution plan...',
    activePipeline: [], // live step states during execution
    liveEvents: [], // live SSE trace items for "Peek Inside"
    peekInsideOpen: false,
    eventSource: null,

    // --- Contextual Slide-Over Drawer State ---
    drawerOpen: false,
    drawerView: 'system', // 'doc' | 'table' | 'draft' | 'memory' | 'system'
    drawerTitle: 'System & Diagnostics',
    activeDocument: null,
    activeDraft: null,
    vaultDocuments: [],
    memoryItems: [],
    memorySearchQuery: '',
    memoryNetworkFilter: 'all',
    detailedModels: [],
    isSwitchingModel: false,
    isRestartingInfra: false,

    // --- Debug Traces (Inside System Drawer) ---
    debugEvents: [],
    debugFilterType: 'all',
    debugSearchQuery: '',
    debugExpandedEventId: null,

    // --- Global Command Palette (Cmd+K) ---
    commandPaletteOpen: false,
    commandQuery: '',
    commandSelectedIndex: 0,

    // --- File Ingestion ---
    isUploading: false,
    uploadFeedback: null,

    // =========================================================================
    // INITIALIZATION & LIFECYCLE
    // =========================================================================
    async initApp() {
      // Configure Marked.js
      if (window.marked) {
        window.marked.setOptions({
          gfm: true,
          breaks: true,
          highlight: function (code, lang) {
            if (window.hljs && lang && window.hljs.getLanguage(lang)) {
              try {
                return window.hljs.highlight(code, { language: lang }).value;
              } catch (e) {}
            }
            return code;
          },
        });
      }

      // Initialize default session ID if not set
      if (!this.currentSessionId || this.currentSessionId === 'session_default') {
        this.currentSessionId = 'session_' + Math.random().toString(36).substring(2, 9);
      }

      await Promise.all([
        this.loadStatus(),
        this.loadStats(),
        this.loadSessions(),
        this.loadModels(),
      ]);

      // Connect SSE stream for live subagent thoughts
      this.connectSSE(this.currentSessionId);

      // Register Global Keyboard Shortcuts
      window.addEventListener('keydown', (e) => {
        // Cmd+K or Ctrl+K -> Command Palette
        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
          e.preventDefault();
          this.commandPaletteOpen = !this.commandPaletteOpen;
          if (this.commandPaletteOpen) {
            this.commandQuery = '';
            this.commandSelectedIndex = 0;
            this.$nextTick(() => {
              const input = document.getElementById('commandPaletteInput');
              if (input) input.focus();
            });
          }
        }
        // Esc -> Close Modals & Slide-over
        if (e.key === 'Escape') {
          if (this.commandPaletteOpen) this.commandPaletteOpen = false;
          else if (this.drawerOpen) this.drawerOpen = false;
        }
        // N key (when not typing in an input) -> New Chat
        if (
          e.key.toLowerCase() === 'n' &&
          !['input', 'textarea'].includes(document.activeElement?.tagName?.toLowerCase()) &&
          !e.metaKey &&
          !e.ctrlKey
        ) {
          e.preventDefault();
          this.newSession();
        }
      });
    },

    // =========================================================================
    // TELEMETRY & SYSTEM HEALTH
    // =========================================================================
    async loadStatus() {
      try {
        const res = await fetch('/api/status');
        if (res.ok) {
          this.status = await res.json();
        }
      } catch (e) {
        console.warn('Status fetch error:', e);
      }
    },

    async loadStats() {
      try {
        const res = await fetch('/api/stats');
        if (res.ok) {
          this.stats = await res.json();
        }
      } catch (e) {
        console.warn('Stats fetch error:', e);
      }
    },

    async loadModels() {
      try {
        const res = await fetch('/api/models');
        if (res.ok) {
          const data = await res.json();
          this.detailedModels = data.detailed_models || [];
        }
      } catch (e) {
        console.warn('Models fetch error:', e);
      }
    },

    async togglePolicy() {
      const nextLevel = this.status.autonomy_level === 'FULL' ? 'SUPERVISED' : 'FULL';
      try {
        const res = await fetch('/api/policy', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ autonomy_level: nextLevel }),
        });
        if (res.ok) {
          const data = await res.json();
          this.status.autonomy_level = data.autonomy_level;
          await this.loadStats();
        }
      } catch (e) {
        console.error('Policy toggle error:', e);
      }
    },

    async switchModel(modelId) {
      this.isSwitchingModel = true;
      try {
        const res = await fetch('/api/model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_name: modelId }),
        });
        if (res.ok) {
          const data = await res.json();
          this.status.model_name = data.model_name;
          await this.loadModels();
        }
      } catch (e) {
        console.error('Model switch error:', e);
      } finally {
        this.isSwitchingModel = false;
      }
    },

    async restartInfra() {
      this.isRestartingInfra = true;
      try {
        await fetch('/api/infra/restart', { method: 'POST' });
        await Promise.all([this.loadStatus(), this.loadStats()]);
      } catch (e) {
        console.error('Infra restart error:', e);
      } finally {
        this.isRestartingInfra = false;
      }
    },

    // =========================================================================
    // SESSION MANAGEMENT & AUTO-TITLING
    // =========================================================================
    async loadSessions() {
      try {
        const res = await fetch('/api/sessions');
        if (res.ok) {
          const data = await res.json();
          this.sessions = data.sessions || [];
        }
      } catch (e) {
        console.warn('Sessions fetch error:', e);
      }
    },

    get groupedSessions() {
      const now = Date.now() / 1000;
      const oneDay = 86400;
      const groups = { today: [], yesterday: [], week: [], older: [] };

      for (const s of this.sessions) {
        const activity = s.last_activity || now;
        const diff = now - activity;
        if (diff < oneDay) {
          groups.today.push(s);
        } else if (diff < oneDay * 2) {
          groups.yesterday.push(s);
        } else if (diff < oneDay * 7) {
          groups.week.push(s);
        } else {
          groups.older.push(s);
        }
      }
      return groups;
    },

    async switchSession(sessionId) {
      if (this.currentSessionId === sessionId) return;
      this.currentSessionId = sessionId;
      const current = this.sessions.find((s) => s.session_id === sessionId);
      this.currentSessionTitle = current?.title || `Session ${sessionId.substring(0, 8)}`;
      this.chatMessages = [];
      this.activePipeline = [];
      this.liveEvents = [];

      // Reconnect SSE
      this.connectSSE(sessionId);

      // Load conversation turns
      try {
        const res = await fetch(`/api/session/${sessionId}`);
        if (res.ok) {
          const data = await res.json();
          this.chatMessages = (data.turns || []).map((t) => ({
            role: t.role,
            content: t.content,
            timestamp: t.created_at,
          }));
          this.scrollToBottom();
        }
      } catch (e) {
        console.warn('Failed to load session dialogue:', e);
      }
    },

    newSession() {
      const newId = 'session_' + Math.random().toString(36).substring(2, 9);
      this.currentSessionId = newId;
      this.currentSessionTitle = 'New Executive Workspace';
      this.chatMessages = [];
      this.activePipeline = [];
      this.liveEvents = [];
      this.connectSSE(newId);
      this.$nextTick(() => {
        const composer = document.getElementById('chatInputComposer');
        if (composer) composer.focus();
      });
    },

    async renameSessionPrompt(sessionId, currentTitle) {
      const newTitle = prompt('Rename workspace conversation:', currentTitle);
      if (!newTitle || !newTitle.trim()) return;
      try {
        const res = await fetch(`/api/session/${sessionId}/title`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: newTitle.trim() }),
        });
        if (res.ok) {
          if (this.currentSessionId === sessionId) {
            this.currentSessionTitle = newTitle.trim();
          }
          await this.loadSessions();
        }
      } catch (e) {
        console.error('Failed to rename session:', e);
      }
    },

    async deleteSession(sessionId) {
      if (!confirm('Are you sure you want to delete this conversation workspace?')) return;
      try {
        await fetch(`/api/session/${sessionId}`, { method: 'DELETE' });
        if (this.currentSessionId === sessionId) {
          this.newSession();
        }
        await this.loadSessions();
      } catch (e) {
        console.error('Failed to delete session:', e);
      }
    },

    // =========================================================================
    // REAL-TIME SSE STREAMING ("PEEK BEHIND THE CURTAIN")
    // =========================================================================
    connectSSE(sessionId) {
      if (this.eventSource) {
        try {
          this.eventSource.close();
        } catch (e) {}
      }

      this.eventSource = new EventSource(`/api/stream/${sessionId}`);

      this.eventSource.addEventListener('trace', (e) => {
        try {
          const ev = JSON.parse(e.data);
          this.handleLiveTraceEvent(ev);
        } catch (err) {
          console.debug('SSE parse error:', err);
        }
      });

      this.eventSource.onerror = (err) => {
        // SSE automatic reconnect handles standard keep-alive drops
      };
    },

    handleLiveTraceEvent(ev) {
      // Append to live mission control stream (keep last 25)
      this.liveEvents.unshift({
        id: ev.id,
        timestamp: ev.timestamp ? ev.timestamp.split('T')[1]?.substring(0, 8) : '',
        component: ev.component,
        event_type: ev.event_type,
        message: ev.message,
        payload: ev.payload,
        duration_ms: ev.duration_ms,
      });
      if (this.liveEvents.length > 25) {
        this.liveEvents.pop();
      }

      // Update interactive thinking ticker with real subagent context
      if (ev.event_type === 'PLAN_FORMULATED') {
        this.thinkingTicker = `Executive plan compiled (${ev.payload?.step_count || 'DAG'} steps)...`;
      } else if (ev.event_type === 'STEP_START') {
        const agent = ev.component || 'Specialist Agent';
        this.thinkingTicker = `${agent} dispatched: ${ev.message}...`;
      } else if (ev.event_type === 'TOOL_CALL') {
        const tool = ev.payload?.tool_name || 'tool';
        this.thinkingTicker = `${ev.component} executing ${tool}...`;
      } else if (ev.event_type === 'TOOL_RESULT') {
        this.thinkingTicker = `${ev.component} processed tool output...`;
      } else if (ev.event_type === 'LLM_PROMPT') {
        this.thinkingTicker = `Argus reasoning with 12K cognitive context...`;
      }
    },

    // =========================================================================
    // CONVERSATIONAL CHAT & GOAL DISPATCH
    // =========================================================================
    async sendMessage(overrideText = null) {
      const text = (overrideText || this.chatInput).trim();
      if (!text || this.isThinking) return;

      this.chatInput = '';
      this.isThinking = true;
      this.thinkingTicker = 'Formulating execution plan...';
      this.liveEvents = [];
      this.activePipeline = [];

      // Append user turn immediately
      this.chatMessages.push({
        role: 'user',
        content: text,
        timestamp: Date.now() / 1000,
      });
      this.scrollToBottom();

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            session_id: this.currentSessionId,
          }),
        });

        if (res.ok) {
          const data = await res.json();
          this.chatMessages.push({
            role: 'assistant',
            content: data.message || 'Workflow executed.',
            timestamp: Date.now() / 1000,
            type: data.type,
            duration_ms: data.duration_ms,
            citations: data.citations || [],
            plan: data.plan || null,
          });

          // Refresh sessions and workspace stats (titles and document counts)
          await Promise.all([this.loadSessions(), this.loadStats()]);

          // Update current session title if updated
          const current = this.sessions.find((s) => s.session_id === this.currentSessionId);
          if (current?.title) {
            this.currentSessionTitle = current.title;
          }
        } else {
          const errData = await res.json().catch(() => ({}));
          this.chatMessages.push({
            role: 'assistant',
            content: `Execution notice: ${errData.detail || res.statusText}`,
            timestamp: Date.now() / 1000,
            type: 'error',
          });
        }
      } catch (e) {
        this.chatMessages.push({
          role: 'assistant',
          content: `Connection notice: ${e.message}`,
          timestamp: Date.now() / 1000,
          type: 'error',
        });
      } finally {
        this.isThinking = false;
        this.scrollToBottom();
      }
    },

    scrollToBottom() {
      this.$nextTick(() => {
        const stream = document.getElementById('chatViewport');
        if (stream) {
          stream.scrollTop = stream.scrollHeight;
        }
      });
    },

    // =========================================================================
    // CONTEXTUAL SLIDE-OVER DRAWER (DOCS, DRAFTS, MEMORY, SYSTEM)
    // =========================================================================
    async openDocViewer(docId) {
      try {
        const res = await fetch(`/api/vault/${docId}`);
        if (res.ok) {
          this.activeDocument = await res.json();
          this.drawerView = 'doc';
          this.drawerTitle = this.activeDocument.filename || 'Document Reader';
          this.drawerOpen = true;
        }
      } catch (e) {
        console.error('Error opening document viewer:', e);
      }
    },

    async openVaultWorkspace() {
      try {
        const res = await fetch('/api/vault');
        if (res.ok) {
          const data = await res.json();
          this.vaultDocuments = data.documents || [];
          this.drawerView = 'vault_list';
          this.drawerTitle = 'Document Vault & Knowledge Base';
          this.drawerOpen = true;
        }
      } catch (e) {
        console.error('Error opening vault list:', e);
      }
    },

    async openMailDrafts() {
      try {
        const res = await fetch('/api/mail/drafts');
        if (res.ok) {
          const data = await res.json();
          this.activeDrafts = data.drafts || [];
          this.drawerView = 'drafts_list';
          this.drawerTitle = 'Supervised Outbound Mail Drafts';
          this.drawerOpen = true;
        }
      } catch (e) {
        console.error('Error opening drafts:', e);
      }
    },

    async openMemoryInspector() {
      this.drawerView = 'memory';
      this.drawerTitle = 'Executive Memory & Beliefs';
      this.drawerOpen = true;
      await this.searchMemory();
    },

    async searchMemory() {
      try {
        const query = encodeURIComponent(this.memorySearchQuery);
        const net = this.memoryNetworkFilter !== 'all' ? `&network=${this.memoryNetworkFilter}` : '';
        const url = query ? `/api/memory?query=${query}${net}` : '/api/memory';
        const res = await fetch(url);
        if (res.ok) {
          const data = await res.json();
          this.memoryItems = data.results || data.active_beliefs || [];
        }
      } catch (e) {
        console.error('Error searching memory:', e);
      }
    },

    async openSystemHub() {
      this.drawerView = 'system';
      this.drawerTitle = 'System & Telemetry Hub';
      this.drawerOpen = true;
      await Promise.all([this.loadDebugEvents(), this.loadModels(), this.loadStats()]);
    },

    async loadDebugEvents() {
      try {
        let url = `/api/debug/events?limit=80`;
        if (this.debugFilterType !== 'all') url += `&event_type=${this.debugFilterType}`;
        if (this.debugSearchQuery) url += `&search=${encodeURIComponent(this.debugSearchQuery)}`;
        const res = await fetch(url);
        if (res.ok) {
          const data = await res.json();
          this.debugEvents = data.events || [];
        }
      } catch (e) {
        console.error('Failed to load debug events:', e);
      }
    },

    async clearDebugBuffer() {
      try {
        await fetch('/api/debug/clear', { method: 'POST' });
        this.debugEvents = [];
      } catch (e) {}
    },

    // =========================================================================
    // FILE INGESTION / DROPZONE
    // =========================================================================
    async handleFileUpload(event) {
      const files = event.target.files || event.dataTransfer?.files;
      if (!files || files.length === 0) return;
      const file = files[0];
      this.isUploading = true;
      this.uploadFeedback = `Uploading and parsing ${file.name}...`;

      const formData = new FormData();
      formData.append('file', file);
      formData.append('parse_and_index', 'true');

      try {
        const res = await fetch('/api/vault/upload', {
          method: 'POST',
          body: formData,
        });
        if (res.ok) {
          const data = await res.json();
          this.uploadFeedback = `✔ Ingested ${data.filename} into Document Vault.`;
          await this.loadStats();
          setTimeout(() => {
            this.uploadFeedback = null;
          }, 4000);
          // Offer to inspect document immediately
          this.openDocViewer(data.doc_id);
        } else {
          this.uploadFeedback = `Upload error: ${res.statusText}`;
        }
      } catch (e) {
        this.uploadFeedback = `Upload failed: ${e.message}`;
      } finally {
        this.isUploading = false;
      }
    },

    // =========================================================================
    // UTILITIES & FORMATTING
    // =========================================================================
    renderMarkdown(text) {
      if (!text) return '';
      if (window.marked) {
        let rendered = window.marked.parse(text);
        // Replace citations like [1] or [2] with interactive citation buttons
        rendered = rendered.replace(/\[(\d+)\]/g, (match, p1) => {
          return `<button class="inline-flex items-center px-1.5 py-0.2 mx-0.5 rounded text-[11px] font-mono font-medium bg-indigo-500/10 text-indigo-400 hover:bg-indigo-500/20 border border-indigo-500/30 transition" onclick="window.argusCitationClick(${p1})">[${p1}]</button>`;
        });
        return rendered;
      }
      return text;
    },

    formatModelName(rawName) {
      if (!rawName) return 'Local Inference Model';
      if (rawName.includes('refinedtoolcall')) return 'RefinedToolCallV5 (3.1B)';
      if (rawName.includes('smollm')) return 'SmolLM3 (3.1B)';
      if (rawName.includes('qwen')) return 'Qwen 2.5 (3B)';
      const parts = rawName.split('/');
      return parts[parts.length - 1].split(':')[0];
    },

    formatTime(epochSec) {
      if (!epochSec) return '';
      const date = new Date(epochSec * 1000);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    },

    copyToClipboard(text) {
      if (navigator.clipboard) {
        navigator.clipboard.writeText(text);
      }
    },
  };
}

// Global citation handler for inline markdown pills
window.argusCitationClick = function (citationNumber) {
  const citId = `[${citationNumber}]`;
  // Dispatches custom event to open contextual slide-over if available
  window.dispatchEvent(new CustomEvent('open-citation', { detail: { id: citId } }));
};
