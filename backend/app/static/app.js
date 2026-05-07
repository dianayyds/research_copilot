const WORKSPACE_TITLES = ["__workspace__", "Workspace"];
const WORKSPACE_CREATE_TITLE = "__workspace__";
const WORKSPACE_DESCRIPTION = "Hidden default workspace";

const state = {
  workspaceProjectId: "",
  sessions: [],
  selectedSessionId: "",
  runs: [],
  streamingRun: null,
  assets: [],
  assetDrawerOpen: false,
  latsMode: false,
};

const sessionListEl = document.getElementById("session-list");
const chatTitleEl = document.getElementById("chat-title");
const chatStreamEl = document.getElementById("chat-stream");
const sequenceHintEl = document.getElementById("sequence-hint");
const composerForm = document.getElementById("composer-form");
const composerQueryEl = document.getElementById("composer-query");
const composerSubmitEl = document.getElementById("composer-submit");
const planSolveModeToggleEl = document.getElementById("plan-solve-mode-toggle");
const latsModeToggleEl = document.getElementById("lats-mode-toggle");
const newSessionButton = document.getElementById("new-session-button");
const assetToggleButton = document.getElementById("asset-toggle-button");
const assetCloseButton = document.getElementById("asset-close-button");
const assetBackdropEl = document.getElementById("asset-backdrop");
const assetDrawerEl = document.getElementById("asset-drawer");
const assetForm = document.getElementById("asset-form");
const assetResetButton = document.getElementById("asset-reset");
const assetUploadForm = document.getElementById("asset-upload-form");
const assetListEl = document.getElementById("asset-list");
const assetUploadSubmitEl = assetUploadForm.querySelector('button[type="submit"]');

async function request(url, options = {}) {
  const isFormData = options.body instanceof FormData;
  const response = await fetch(url, {
    headers: isFormData ? {} : { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  if (response.status === 204) {
    return null;
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderText(value) {
  return escapeHtml(value).replaceAll("\n", "<br />");
}

function renderMarkdownInline(value) {
  const codeSpans = [];
  let html = escapeHtml(value);
  html = html.replace(/`([^`]+)`/g, (_, code) => {
    const index = codeSpans.length;
    codeSpans.push(`<code>${code}</code>`);
    return `@@CODE_SPAN_${index}@@`;
  });
  html = html.replace(/\*\*([\s\S]+?)\*\*/g, (_, content) => `<strong>${content.trim()}</strong>`);
  html = html.replace(/__([\s\S]+?)__/g, (_, content) => `<strong>${content.trim()}</strong>`);
  html = html.replace(/@@CODE_SPAN_(\d+)@@/g, (_, index) => codeSpans[Number(index)] || "");
  return html;
}

function renderMarkdown(value) {
  const lines = String(value || "").replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let paragraph = [];
  let listItems = [];

  const flushParagraph = () => {
    if (!paragraph.length) {
      return;
    }
    blocks.push(`<p>${paragraph.map(renderMarkdownInline).join("<br />")}</p>`);
    paragraph = [];
  };

  const flushList = () => {
    if (!listItems.length) {
      return;
    }
    blocks.push(`<ul>${listItems.map((item) => `<li>${renderMarkdownInline(item)}</li>`).join("")}</ul>`);
    listItems = [];
  };

  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      return;
    }

    const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      flushParagraph();
      flushList();
      const level = Math.min(headingMatch[1].length + 2, 6);
      blocks.push(`<h${level}>${renderMarkdownInline(headingMatch[2])}</h${level}>`);
      return;
    }

    const bulletMatch = trimmed.match(/^[-*]\s+(.+)$/);
    if (bulletMatch) {
      flushParagraph();
      listItems.push(bulletMatch[1]);
      return;
    }

    const numberedMatch = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (numberedMatch) {
      flushParagraph();
      listItems.push(numberedMatch[1]);
      return;
    }

    flushList();
    paragraph.push(trimmed);
  });

  flushParagraph();
  flushList();
  return blocks.join("");
}

function formatTime(value) {
  const date = new Date(value);
  return `${date.getMonth() + 1}/${date.getDate()} ${String(date.getHours()).padStart(2, "0")}:${String(
    date.getMinutes(),
  ).padStart(2, "0")}`;
}

function selectedSession() {
  return state.sessions.find((session) => session.id === state.selectedSessionId) || null;
}

function visibleRuns() {
  return state.streamingRun && state.streamingRun.session_id === state.selectedSessionId
    ? [...state.runs, state.streamingRun]
    : state.runs;
}

function nextSequenceId() {
  return (selectedSession()?.last_sequence_id || 0) + 1;
}

function runModeLabel() {
  if (state.latsMode) {
    return "LATS Agent";
  }
  return "Plan-and-Solve Agent";
}

function plannerModeForStreamingRun() {
  if (state.latsMode) {
    return "lats_agent_mcts";
  }
  return "two_stage";
}

function runPathForMode() {
  if (state.latsMode) {
    return "lats/run/stream";
  }
  return "run/stream";
}

function resetAssetForm() {
  assetForm.reset();
  assetForm.asset_id.value = "";
  assetForm.asset_type.value = "note";
}

function fillAssetForm(asset) {
  assetForm.asset_id.value = asset.id;
  assetForm.title.value = asset.title;
  assetForm.asset_type.value = asset.asset_type;
  assetForm.content.value = asset.content;
}

function adjustComposerHeight() {
  composerQueryEl.style.height = "0px";
  composerQueryEl.style.height = `${Math.min(composerQueryEl.scrollHeight, 220)}px`;
}

function setSubmitting(isSubmitting) {
  composerQueryEl.disabled = isSubmitting;
  composerSubmitEl.disabled = isSubmitting;
  planSolveModeToggleEl.disabled = isSubmitting;
  latsModeToggleEl.disabled = isSubmitting;
  composerSubmitEl.innerHTML = `<span>${isSubmitting ? "…" : "↑"}</span>`;
}

function scrollChatToBottom() {
  chatStreamEl.scrollTop = chatStreamEl.scrollHeight;
}

function toggleAssetDrawer(forceOpen) {
  state.assetDrawerOpen = typeof forceOpen === "boolean" ? forceOpen : !state.assetDrawerOpen;
  assetDrawerEl.hidden = !state.assetDrawerOpen;
  assetBackdropEl.hidden = !state.assetDrawerOpen;
  assetToggleButton.classList.toggle("active", state.assetDrawerOpen);
  document.body.classList.toggle("drawer-open", state.assetDrawerOpen);
}

function renderHeader() {
  const session = selectedSession();
  chatTitleEl.textContent = session ? session.title : "新对话";
  sequenceHintEl.textContent = `${runModeLabel()} · 下一轮 #${nextSequenceId()}`;
  planSolveModeToggleEl.checked = !state.latsMode;
  latsModeToggleEl.checked = state.latsMode;
}

function renderSessions() {
  if (!state.sessions.length) {
    sessionListEl.innerHTML = `<div class="empty-list">还没有对话</div>`;
    return;
  }

  sessionListEl.innerHTML = state.sessions
    .map(
      (session) => `
        <article class="session-item ${session.id === state.selectedSessionId ? "active" : ""}">
          <button class="session-trigger" type="button" data-session-id="${session.id}">
            <span class="session-name">${escapeHtml(session.title)}</span>
            <span class="session-meta">${escapeHtml(session.summary || formatTime(session.updated_at))}</span>
          </button>
          <button class="session-delete" type="button" data-delete-session="${session.id}" aria-label="删除会话">
            ×
          </button>
        </article>
      `,
    )
    .join("");

  sessionListEl.querySelectorAll("[data-session-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      state.selectedSessionId = button.dataset.sessionId;
      await loadSessionRuns();
      renderAll();
      scrollChatToBottom();
    });
  });

  sessionListEl.querySelectorAll("[data-delete-session]").forEach((button) => {
    button.addEventListener("click", async () => {
      const session = state.sessions.find((item) => item.id === button.dataset.deleteSession);
      if (!session || !confirm(`删除会话「${session.title}」？`)) {
        return;
      }
      await request(`/api/v1/projects/${state.workspaceProjectId}/sessions/${session.id}`, { method: "DELETE" });
      if (state.selectedSessionId === session.id) {
        state.selectedSessionId = "";
      }
      await loadSessions();
      await loadSessionRuns();
      renderAll();
    });
  });
}

function renderWelcome() {
  const session = selectedSession();
  return `
    <div class="welcome">
      <h1>${session ? escapeHtml(session.title) : "今天想处理什么？"}</h1>
      <p>${session ? "直接继续这个对话，或在下方输入新的业务。" : "新建一个会话，或者直接在下方输入任务。"}</p>
    </div>
  `;
}

function renderCitations(citations) {
  if (!citations.length) {
    return "";
  }
  return `
    <div class="citation-list">
      ${citations
        .map(
          (citation) => `
            <span class="citation-chip">
              <span>来源</span>
              <span>${escapeHtml(citation.label)}</span>
            </span>
          `,
        )
        .join("")}
    </div>
  `;
}

function formatScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(2) : "0.00";
}

function renderTraceTreeNode(node) {
  const children = node.children || [];
  const depth = Math.max(0, Number(node.depth) || 0);
  const action = node.action || "root";
  const skill = node.skill || "";
  const summary = node.observation_summary || node.reflection || node.thought || "";
  const args = node.arguments && Object.keys(node.arguments).length ? JSON.stringify(node.arguments) : "";
  return `
    <div class="trace-tree-node ${node.best_path ? "best-path" : ""} ${node.terminal ? "terminal-node" : ""}" style="--tree-depth: ${depth}">
      <div class="trace-tree-main">
        <span class="trace-tree-action">${escapeHtml(action)}</span>
        ${skill ? `<span class="trace-tree-skill">${escapeHtml(skill)}</span>` : ""}
        <span class="trace-tree-metric">score ${formatScore(node.score)}</span>
        <span class="trace-tree-metric">visits ${escapeHtml(node.visits || 0)}</span>
        ${node.best_path ? '<span class="trace-tree-best">best</span>' : ""}
      </div>
      <div class="trace-tree-detail">
        ${summary ? `<span>${escapeHtml(summary)}</span>` : ""}
        ${args ? `<code>${escapeHtml(args)}</code>` : ""}
      </div>
    </div>
    ${children.map((child) => renderTraceTreeNode(child)).join("")}
  `;
}

function renderTraceTree(traceTree) {
  if (!traceTree || !traceTree.root) {
    return "";
  }
  const bestActions = traceTree.best_actions?.length ? traceTree.best_actions.join(" → ") : "未选择";
  return `
    <div class="trace-tree">
      <div class="trace-tree-header">
        <strong>Agent 搜索树</strong>
        <span>${escapeHtml(traceTree.iterations || 0)} iter · ${escapeHtml(traceTree.node_count || 0)} nodes</span>
      </div>
      <div class="trace-tree-path">best path: ${escapeHtml(bestActions)}</div>
      <div class="trace-tree-list">
        ${renderTraceTreeNode(traceTree.root)}
      </div>
    </div>
  `;
}

function renderExecutionTrace(run) {
  const plan = run.plan || {};
  const steps = plan.execution_trace || [];
  const summary = plan.solver_summary || "";
  const planSummary = plan.plan_summary || "";
  const replanLabel = plan.replan_count ? `重规划 ${plan.replan_count} 次` : run.status === "streaming" ? "思考中" : "已完成";
  if (!steps.length && !summary && !planSummary && run.status !== "streaming") {
    return "";
  }
  return `
    <div class="thinking-card ${run.status === "streaming" ? "is-streaming" : ""}">
      <div class="thinking-header">
        <div class="thinking-title">执行轨迹</div>
        <div class="thinking-state">${escapeHtml(replanLabel)}</div>
      </div>
      <div class="thinking-plan">${escapeHtml(planSummary || "正在规划执行路径…")}</div>
      ${renderTraceTree(plan.trace_tree)}
      ${
        steps.length
          ? `
            <div class="trace-list">
              ${steps
                .map(
                  (step) => `
                    <div class="trace-item">
                      <span class="trace-action">${escapeHtml(step.action)}</span>
                      <div class="trace-content">
                        <strong>${escapeHtml(step.title)}</strong>
                        <span>${escapeHtml(step.summary)}</span>
                      </div>
                    </div>
                  `,
                )
                .join("")}
            </div>
          `
          : `<div class="thinking-placeholder">正在生成执行步骤…</div>`
      }
      ${
        summary
          ? `<div class="solver-summary"><strong>Solver 总结：</strong>${escapeHtml(summary)}</div>`
          : ""
      }
    </div>
  `;
}

function renderChat() {
  const runs = visibleRuns();
  if (!runs.length) {
    chatStreamEl.innerHTML = renderWelcome();
    return;
  }

  chatStreamEl.innerHTML = runs
    .map(
      (run) => `
        <section class="message-group">
          <div class="user-row">
            <div class="user-bubble">${renderText(run.query)}</div>
          </div>
          <div class="assistant-row">
            <div class="assistant-avatar">AI</div>
            <div class="assistant-block">
              ${renderExecutionTrace(run)}
              <div class="assistant-body">
                ${renderMarkdown(run.answer?.answer || (run.status === "streaming" ? "正在整理答案…" : ""))}
                ${run.status === "streaming" ? '<span class="stream-caret"></span>' : ""}
              </div>
              ${renderCitations(run.answer?.citations || [])}
            </div>
          </div>
        </section>
      `,
    )
    .join("");
}

function renderAssets() {
  if (!state.assets.length) {
    assetListEl.innerHTML = `<div class="empty-list">还没有资产</div>`;
    return;
  }

  assetListEl.innerHTML = state.assets
    .map(
      (asset) => `
        <article class="asset-item">
          <div class="asset-item-head">
            <strong>${escapeHtml(asset.title)}</strong>
            <span class="asset-item-type">${escapeHtml(asset.asset_type)}</span>
          </div>
          <p>${escapeHtml(asset.content.slice(0, 180))}</p>
          <div class="asset-item-actions">
            <button class="ghost-button" type="button" data-edit-asset="${asset.id}">编辑</button>
            <button class="ghost-button danger-button" type="button" data-delete-asset="${asset.id}">删除</button>
          </div>
        </article>
      `,
    )
    .join("");

  assetListEl.querySelectorAll("[data-edit-asset]").forEach((button) => {
    button.addEventListener("click", () => {
      const asset = state.assets.find((item) => item.id === button.dataset.editAsset);
      if (!asset) {
        return;
      }
      fillAssetForm(asset);
      toggleAssetDrawer(true);
    });
  });

  assetListEl.querySelectorAll("[data-delete-asset]").forEach((button) => {
    button.addEventListener("click", async () => {
      await request(`/api/v1/assets/${button.dataset.deleteAsset}`, { method: "DELETE" });
      await loadAssets();
      renderAssets();
    });
  });
}

function renderAll() {
  renderHeader();
  renderSessions();
  renderChat();
  renderAssets();
  adjustComposerHeight();
}

async function ensureWorkspaceProject() {
  const projects = await request("/api/v1/projects");
  const existing = projects.find((project) => WORKSPACE_TITLES.includes(project.title));
  if (existing) {
    state.workspaceProjectId = existing.id;
    return;
  }

  const workspace = await request("/api/v1/projects", {
    method: "POST",
    body: JSON.stringify({
      title: WORKSPACE_CREATE_TITLE,
      description: WORKSPACE_DESCRIPTION,
      status: "active",
    }),
  });
  state.workspaceProjectId = workspace.id;
}

async function loadSessions() {
  if (!state.workspaceProjectId) {
    state.sessions = [];
    state.selectedSessionId = "";
    return;
  }
  state.sessions = await request(`/api/v1/projects/${state.workspaceProjectId}/sessions`);
  state.selectedSessionId =
    state.sessions.find((session) => session.id === state.selectedSessionId)?.id || state.sessions[0]?.id || "";
}

async function loadSessionRuns() {
  if (!state.workspaceProjectId || !state.selectedSessionId) {
    state.runs = [];
    return;
  }
  state.runs = await request(`/api/v1/projects/${state.workspaceProjectId}/sessions/${state.selectedSessionId}/runs`);
}

async function loadAssets() {
  state.assets = await request("/api/v1/assets");
}

async function initializeWorkspace() {
  await ensureWorkspaceProject();
  await Promise.all([loadSessions(), loadAssets()]);
  await loadSessionRuns();
}

async function createSession() {
  if (!state.workspaceProjectId) {
    return;
  }
  const session = await request(`/api/v1/projects/${state.workspaceProjectId}/sessions`, {
    method: "POST",
    body: JSON.stringify({ title: "新会话" }),
  });
  state.selectedSessionId = session.id;
  await loadSessions();
  state.selectedSessionId = session.id;
  await loadSessionRuns();
  renderAll();
  composerQueryEl.focus();
}

function createStreamingRun(query, sessionId, sequenceId) {
  return {
    id: `stream-${Date.now()}`,
    session_id: sessionId,
    sequence_id: sequenceId,
    query,
    status: "streaming",
    answer: { answer: "", citations: [] },
    plan: {
      planner_mode: plannerModeForStreamingRun(),
      plan_summary: "",
      tasks: [],
      execution_trace: [],
      trace_tree: {},
      solver_summary: "",
      replan_count: 0,
      replan_reason: "",
    },
  };
}

async function streamRequest(url, payload, onEvent) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorPayload = await response.json().catch(() => ({}));
    throw new Error(errorPayload.detail || `Request failed: ${response.status}`);
  }
  if (!response.body) {
    throw new Error("当前浏览器不支持流式响应。");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      await onEvent(JSON.parse(trimmed));
    }
  }
  if (buffer.trim()) {
    await onEvent(JSON.parse(buffer.trim()));
  }
}

async function runTurn(query) {
  const prompt = query.trim();
  if (!prompt) {
    return;
  }

  if (!state.selectedSessionId) {
    await createSession();
  }

  const sequenceId = nextSequenceId();
  setSubmitting(true);
  try {
    composerForm.reset();
    adjustComposerHeight();
    state.streamingRun = createStreamingRun(prompt, state.selectedSessionId, sequenceId);
    renderAll();
    scrollChatToBottom();
    const runPath = runPathForMode();
    await streamRequest(`/api/v1/projects/${state.workspaceProjectId}/sessions/${state.selectedSessionId}/${runPath}`, {
      user_query: prompt,
      asset_ids: [],
      sequence_id: sequenceId,
    }, async (event) => {
      if (!state.streamingRun) {
        return;
      }
      if (event.type === "plan" && event.plan) {
        state.streamingRun.plan = { ...state.streamingRun.plan, ...event.plan };
      }
      if (event.type === "trace" && event.step) {
        state.streamingRun.plan.execution_trace = [...state.streamingRun.plan.execution_trace, event.step];
      }
      if (event.type === "trace_tree" && event.trace_tree) {
        state.streamingRun.plan.trace_tree = event.trace_tree;
      }
      if (event.type === "solver_summary") {
        state.streamingRun.plan.solver_summary = event.solver_summary || "";
        state.streamingRun.plan.replan_count = event.replan_count || 0;
        state.streamingRun.plan.replan_reason = event.replan_reason || "";
      }
      if (event.type === "answer_delta") {
        state.streamingRun.answer.answer = event.answer || `${state.streamingRun.answer.answer}${event.delta || ""}`;
      }
      if (event.type === "complete") {
        await loadSessions();
        await loadSessionRuns();
        state.streamingRun = null;
      }
      if (event.type === "error") {
        throw new Error(event.detail || "流式执行失败");
      }
      renderAll();
      scrollChatToBottom();
    });
  } catch (error) {
    state.streamingRun = null;
    renderAll();
    alert(error.message);
  } finally {
    setSubmitting(false);
  }
}

newSessionButton.addEventListener("click", async () => {
  await createSession();
});

assetToggleButton.addEventListener("click", () => {
  toggleAssetDrawer();
});

planSolveModeToggleEl.addEventListener("change", () => {
  state.latsMode = false;
  renderHeader();
});

latsModeToggleEl.addEventListener("change", () => {
  state.latsMode = true;
  renderHeader();
});

assetCloseButton.addEventListener("click", () => {
  toggleAssetDrawer(false);
});

assetBackdropEl.addEventListener("click", () => {
  toggleAssetDrawer(false);
});

assetResetButton.addEventListener("click", () => {
  resetAssetForm();
});

assetForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const formData = new FormData(assetForm);
    const assetId = String(formData.get("asset_id") || "");
    const url = assetId ? `/api/v1/assets/${assetId}` : "/api/v1/assets";
    const method = assetId ? "PATCH" : "POST";
    await request(url, {
      method,
      body: JSON.stringify({
        title: String(formData.get("title") || "").trim(),
        asset_type: formData.get("asset_type"),
        content: String(formData.get("content") || "").trim(),
      }),
    });
    resetAssetForm();
    await loadAssets();
    renderAssets();
  } catch (error) {
    alert(error.message);
  }
});

const RESUMABLE_UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024;
const RESUMABLE_UPLOAD_CONCURRENCY = 3;
const MD5_S = [
  7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
  5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
  4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
  6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];
const MD5_K = Array.from({ length: 64 }, (_, index) => Math.floor(Math.abs(Math.sin(index + 1)) * 2 ** 32) >>> 0);

function leftRotate(value, amount) {
  return ((value << amount) | (value >>> (32 - amount))) >>> 0;
}

function createMd5() {
  let a0 = 0x67452301;
  let b0 = 0xefcdab89;
  let c0 = 0x98badcfe;
  let d0 = 0x10325476;
  let totalBytes = 0n;
  let pending = new Uint8Array(0);

  function processBlock(block, offset) {
    const words = new Array(16);
    for (let i = 0; i < 16; i += 1) {
      const j = offset + i * 4;
      words[i] = (block[j] | (block[j + 1] << 8) | (block[j + 2] << 16) | (block[j + 3] << 24)) >>> 0;
    }
    let a = a0;
    let b = b0;
    let c = c0;
    let d = d0;
    for (let i = 0; i < 64; i += 1) {
      let f;
      let g;
      if (i < 16) {
        f = (b & c) | (~b & d);
        g = i;
      } else if (i < 32) {
        f = (d & b) | (~d & c);
        g = (5 * i + 1) % 16;
      } else if (i < 48) {
        f = b ^ c ^ d;
        g = (3 * i + 5) % 16;
      } else {
        f = c ^ (b | ~d);
        g = (7 * i) % 16;
      }
      const next = d;
      d = c;
      c = b;
      b = (b + leftRotate((a + f + MD5_K[i] + words[g]) >>> 0, MD5_S[i])) >>> 0;
      a = next;
    }
    a0 = (a0 + a) >>> 0;
    b0 = (b0 + b) >>> 0;
    c0 = (c0 + c) >>> 0;
    d0 = (d0 + d) >>> 0;
  }

  function append(bytes, countBytes = true) {
    const input = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    if (countBytes) {
      totalBytes += BigInt(input.length);
    }
    let combined = input;
    if (pending.length) {
      combined = new Uint8Array(pending.length + input.length);
      combined.set(pending);
      combined.set(input, pending.length);
      pending = new Uint8Array(0);
    }
    const fullLength = combined.length - (combined.length % 64);
    for (let offset = 0; offset < fullLength; offset += 64) {
      processBlock(combined, offset);
    }
    pending = combined.slice(fullLength);
  }

  function digest() {
    const bitLength = totalBytes * 8n;
    const paddingLength = Number((56n - ((totalBytes + 1n) % 64n) + 64n) % 64n);
    const padding = new Uint8Array(1 + paddingLength + 8);
    padding[0] = 0x80;
    for (let i = 0; i < 8; i += 1) {
      padding[1 + paddingLength + i] = Number((bitLength >> BigInt(8 * i)) & 0xffn);
    }
    append(padding, false);
    const words = [a0, b0, c0, d0];
    return words
      .flatMap((word) => [word & 0xff, (word >>> 8) & 0xff, (word >>> 16) & 0xff, (word >>> 24) & 0xff])
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
  }

  return { update: append, digest };
}

async function computeFileMd5(file, onProgress) {
  const md5 = createMd5();
  const hashChunkSize = 4 * 1024 * 1024;
  let offset = 0;
  while (offset < file.size) {
    const end = Math.min(offset + hashChunkSize, file.size);
    const buffer = await file.slice(offset, end).arrayBuffer();
    md5.update(new Uint8Array(buffer));
    offset = end;
    onProgress?.(offset, file.size);
  }
  return md5.digest();
}

function setUploadProgress(message) {
  const progressEl = document.getElementById("asset-upload-progress");
  if (progressEl) {
    progressEl.textContent = message;
  }
}

async function uploadAssetFileResumable(file, formData) {
  assetUploadSubmitEl.disabled = true;
  assetUploadSubmitEl.textContent = "计算MD5...";
  setUploadProgress("正在计算文件 MD5...");
  const fileMd5 = await computeFileMd5(file, (done, total) => {
    setUploadProgress(`正在计算 MD5：${Math.round((done / total) * 100)}%`);
  });
  assetUploadSubmitEl.textContent = "准备上传...";
  const initStatus = await request("/api/v1/assets/uploads/init", {
    method: "POST",
    body: JSON.stringify({
      filename: file.name,
      file_size: file.size,
      file_md5: fileMd5,
      chunk_size: RESUMABLE_UPLOAD_CHUNK_SIZE,
      title: String(formData.get("upload_title") || "").trim(),
      asset_type: String(formData.get("asset_type") || ""),
    }),
  });
  if (initStatus.finalized && initStatus.asset) {
    setUploadProgress("文件已上传过，已直接复用资产。");
    return initStatus.asset;
  }
  const chunkSize = initStatus.chunk_size || RESUMABLE_UPLOAD_CHUNK_SIZE;
  const missing = [...(initStatus.missing_chunks || [])];
  let uploaded = initStatus.uploaded_count || 0;
  setUploadProgress(`已找到 ${uploaded}/${initStatus.total_chunks} 个分片，继续上传缺失部分。`);

  async function uploadOne(chunkIndex) {
    const start = chunkIndex * chunkSize;
    const end = Math.min(start + chunkSize, file.size);
    const chunkBody = new FormData();
    chunkBody.append("chunk_index", String(chunkIndex));
    chunkBody.append("chunk", file.slice(start, end), `${file.name}.part-${chunkIndex}`);
    await request(`/api/v1/assets/uploads/${fileMd5}/chunks`, {
      method: "POST",
      body: chunkBody,
    });
    uploaded += 1;
    assetUploadSubmitEl.textContent = `上传中 ${uploaded}/${initStatus.total_chunks}`;
    setUploadProgress(`上传分片 ${uploaded}/${initStatus.total_chunks}`);
  }

  let cursor = 0;
  const workers = Array.from({ length: Math.min(RESUMABLE_UPLOAD_CONCURRENCY, missing.length) }, async () => {
    while (cursor < missing.length) {
      const chunkIndex = missing[cursor];
      cursor += 1;
      await uploadOne(chunkIndex);
    }
  });
  await Promise.all(workers);
  assetUploadSubmitEl.textContent = "合并中...";
  setUploadProgress("所有分片已上传，正在 MinIO 合并并导入资产...");
  const completeStatus = await request(`/api/v1/assets/uploads/${fileMd5}/complete`, {
    method: "POST",
    body: JSON.stringify({
      title: String(formData.get("upload_title") || "").trim(),
      asset_type: String(formData.get("asset_type") || ""),
    }),
  });
  if (!completeStatus.asset) {
    throw new Error("分片上传已完成，但资产创建失败。");
  }
  setUploadProgress("导入完成。");
  return completeStatus.asset;
}

assetUploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const formData = new FormData(assetUploadForm);
    const file = formData.get("file");
    if (!(file instanceof File) || !file.size) {
      alert("请选择一个文件。");
      return;
    }
    const uploadStartedAt = performance.now();
    console.info("[asset-upload] start", {
      filename: file.name,
      size_bytes: file.size,
      mime_type: file.type || "unknown",
      asset_type: String(formData.get("asset_type") || ""),
    });
    const asset = await uploadAssetFileResumable(file, formData);
    console.info("[asset-upload] complete", {
      filename: file.name,
      asset_id: asset.id,
      asset_type: asset.asset_type,
      elapsed_ms: Math.round(performance.now() - uploadStartedAt),
    });
    assetUploadForm.reset();
    await loadAssets();
    renderAssets();
  } catch (error) {
    console.error("[asset-upload] failed", error);
    alert(error.message);
  } finally {
    assetUploadSubmitEl.disabled = false;
    assetUploadSubmitEl.textContent = "导入文件";
    setUploadProgress("");
  }
});

composerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runTurn(composerQueryEl.value);
});

composerQueryEl.addEventListener("input", () => {
  adjustComposerHeight();
});

composerQueryEl.addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    await runTurn(composerQueryEl.value);
  }
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.assetDrawerOpen) {
    toggleAssetDrawer(false);
  }
});

initializeWorkspace()
  .then(() => {
    renderAll();
    adjustComposerHeight();
    composerQueryEl.focus();
    scrollChatToBottom();
  })
  .catch((error) => {
    alert(error.message);
  });
