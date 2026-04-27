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
  agentMode: false,
  latsMode: false,
};

const sessionListEl = document.getElementById("session-list");
const chatTitleEl = document.getElementById("chat-title");
const chatStreamEl = document.getElementById("chat-stream");
const sequenceHintEl = document.getElementById("sequence-hint");
const composerForm = document.getElementById("composer-form");
const composerQueryEl = document.getElementById("composer-query");
const composerSubmitEl = document.getElementById("composer-submit");
const agentModeToggleEl = document.getElementById("agent-mode-toggle");
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
    return "LATS";
  }
  return state.agentMode ? "Agent" : "Research";
}

function plannerModeForStreamingRun() {
  if (state.latsMode) {
    return "lats_agent_mcts";
  }
  return state.agentMode ? "agent_loop" : "two_stage";
}

function runPathForMode() {
  if (state.latsMode) {
    return "lats/run/stream";
  }
  return state.agentMode ? "agent/run/stream" : "run/stream";
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
  agentModeToggleEl.disabled = isSubmitting;
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
  agentModeToggleEl.checked = state.agentMode;
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
                ${renderText(run.answer?.answer || (run.status === "streaming" ? "正在整理答案…" : ""))}
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

agentModeToggleEl.addEventListener("change", () => {
  state.agentMode = agentModeToggleEl.checked;
  if (state.agentMode) {
    state.latsMode = false;
  }
  renderHeader();
});

latsModeToggleEl.addEventListener("change", () => {
  state.latsMode = latsModeToggleEl.checked;
  if (state.latsMode) {
    state.agentMode = false;
  }
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
    assetUploadSubmitEl.disabled = true;
    assetUploadSubmitEl.textContent = "导入中...";
    const uploadBody = new FormData();
    uploadBody.append("file", file);
    uploadBody.append("title", String(formData.get("upload_title") || "").trim());
    uploadBody.append("asset_type", String(formData.get("asset_type") || ""));
    const asset = await request("/api/v1/assets/upload-file", {
      method: "POST",
      body: uploadBody,
    });
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
