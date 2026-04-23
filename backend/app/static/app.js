const state = {
  projects: [],
  selectedProjectId: "",
  assets: [],
  todos: [],
  runs: [],
  memory: [],
  config: null,
  dashboard: null,
  currentRun: null,
};

const projectListEl = document.getElementById("project-list");
const assetListEl = document.getElementById("asset-list");
const todoListEl = document.getElementById("todo-list");
const runListEl = document.getElementById("run-list");
const memoryListEl = document.getElementById("memory-list");
const dashboardEl = document.getElementById("dashboard");
const providerConfigEl = document.getElementById("provider-config");
const projectTitleEl = document.getElementById("project-title");
const projectDescriptionEl = document.getElementById("project-description");
const runResultEl = document.getElementById("run-result");
const runStatusEl = document.getElementById("run-status");
const runDetailModalEl = document.getElementById("run-detail-modal");
const runDetailBodyEl = document.getElementById("run-detail-body");
const runDetailTitleEl = document.getElementById("run-detail-title");
const runDetailCloseButton = document.getElementById("run-detail-close");

const projectForm = document.getElementById("project-form");
const assetForm = document.getElementById("asset-form");
const assetUploadForm = document.getElementById("asset-upload-form");
const assetResetButton = document.getElementById("asset-reset");
const todoForm = document.getElementById("todo-form");
const todoResetButton = document.getElementById("todo-reset");
const runForm = document.getElementById("run-form");

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
  return response.json();
}

function badgeClass(status) {
  return status === "done" || status === "active" ? "badge success" : "badge warn";
}

function selectedProject() {
  return state.projects.find((project) => project.id === state.selectedProjectId) || null;
}

function emptyCard(message) {
  return `<div class="empty">${message}</div>`;
}

function runDetailMarkup(run) {
  const planItems = run.plan.tasks.map((task) => `<li>${task.title}: ${task.goal}</li>`).join("");
  const evidenceItems = run.retrieval.evidence_items
    .map((item) => `<li>[${item.label}] ${item.title}: ${item.snippet}</li>`)
    .join("");
  const memoryItems = run.memory.memory_updates
    .map((item) => `<li>${item.memory_type}.${item.key}: ${item.value}</li>`)
    .join("");
  return `
    <article class="result-card">
      <h3>任务</h3>
      <p><strong>问题：</strong>${run.query}</p>
      <p class="muted"><strong>状态：</strong>${run.status} · <strong>时间：</strong>${new Date(run.created_at).toLocaleString()}</p>
    </article>
    <article class="result-card">
      <h3>答案</h3>
      <pre>${run.answer.answer}</pre>
    </article>
    <article class="result-card">
      <h3>执行计划</h3>
      <ol>${planItems}</ol>
    </article>
    <article class="result-card">
      <h3>引用证据</h3>
      <ol>${evidenceItems || "<li>暂无证据</li>"}</ol>
    </article>
    <article class="result-card">
      <h3>记忆更新</h3>
      <ul>${memoryItems || "<li>暂无记忆更新</li>"}</ul>
    </article>
  `;
}

function openRunModal(run) {
  state.currentRun = run;
  runDetailTitleEl.textContent = "运行详情";
  runDetailBodyEl.innerHTML = runDetailMarkup(run);
  runDetailModalEl.hidden = false;
  document.body.classList.add("modal-open");
}

function closeRunModal() {
  runDetailModalEl.hidden = true;
  document.body.classList.remove("modal-open");
}

function renderDashboard() {
  const dashboard = state.dashboard;
  if (!dashboard) {
    dashboardEl.innerHTML = emptyCard("暂无数据");
    return;
  }
  const cards = [
    ["项目", dashboard.project_count],
    ["TODO", dashboard.todo_count],
    ["未完成", dashboard.open_todo_count],
    ["运行", dashboard.run_count],
  ];
  dashboardEl.innerHTML = cards
    .map(([label, value]) => `<div class="stat-card"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderProviderConfig() {
  if (!state.config) {
    providerConfigEl.innerHTML = "";
    return;
  }
  providerConfigEl.innerHTML = [
    `<div><strong>LLM</strong>: ${state.config.llm.provider} / ${state.config.llm.model}</div>`,
    `<div><strong>Embedding</strong>: ${state.config.embedding.provider} / ${state.config.embedding.model}</div>`,
    `<div><strong>Reranker</strong>: ${state.config.reranker.provider} / ${state.config.reranker.model}</div>`,
    `<div><strong>Mode</strong>: ${state.config.execution_mode}</div>`,
  ].join("");
}

function renderProjects() {
  if (!state.projects.length) {
    projectListEl.innerHTML = emptyCard("先创建一个项目。");
    return;
  }
  projectListEl.innerHTML = state.projects
    .map(
      (project) => `
        <article class="list-item project-card ${project.id === state.selectedProjectId ? "active" : ""}">
          <p class="item-title">${project.title}</p>
          <p class="muted">${project.description || "暂无描述"}</p>
          <div class="item-meta">
            <span class="badge">${project.asset_count} 资产</span>
            <span class="badge">${project.todo_count} TODO</span>
            <span class="badge">${project.run_count} 运行</span>
          </div>
          <div class="project-card-actions">
            <button class="ghost" data-project-id="${project.id}">打开</button>
            <button class="danger" data-delete-project="${project.id}">删除</button>
          </div>
        </article>
      `,
    )
    .join("");
  projectListEl.querySelectorAll("[data-project-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      closeRunModal();
      state.selectedProjectId = button.dataset.projectId;
      await hydrateProjectWorkspace();
      renderAll();
    });
  });
  projectListEl.querySelectorAll("[data-delete-project]").forEach((button) => {
    button.addEventListener("click", async () => {
      const projectId = button.dataset.deleteProject;
      const project = state.projects.find((item) => item.id === projectId);
      if (!project || !confirm(`删除项目「${project.title}」及其资产、TODO、运行记录和记忆？`)) {
        return;
      }
      closeRunModal();
      await request(`/api/v1/projects/${projectId}`, { method: "DELETE" });
      state.selectedProjectId =
        state.selectedProjectId === projectId
          ? state.projects.find((item) => item.id !== projectId)?.id || ""
          : state.selectedProjectId;
      await hydrateDashboard();
      await loadProjects();
      await hydrateProjectWorkspace();
      renderAll();
    });
  });
}

function renderProjectHeader() {
  const project = selectedProject();
  projectTitleEl.textContent = project ? project.title : "选择一个项目开始";
  projectDescriptionEl.textContent = project
    ? project.description || "项目已创建，可以继续补充知识资产与 TODO。"
    : "可以先创建项目，再补充资产、TODO 和研究请求。";
}

function renderAssets() {
  if (!state.selectedProjectId) {
    assetListEl.innerHTML = emptyCard("选择项目后再添加资产。");
    return;
  }
  if (!state.assets.length) {
    assetListEl.innerHTML = emptyCard("暂无资产。可以先粘贴论文摘要、代码说明或笔记。");
    return;
  }
  assetListEl.innerHTML = state.assets
    .map(
      (asset) => `
        <article class="list-item">
          <p class="item-title">${asset.title}</p>
          <div class="item-meta">
            <span class="badge">${asset.asset_type}</span>
          </div>
          <p class="muted">${asset.content}</p>
          <div class="actions">
            <button class="ghost" data-edit-asset="${asset.id}">编辑</button>
            <button class="ghost" data-delete-asset="${asset.id}">删除</button>
          </div>
        </article>
      `,
    )
    .join("");
  assetListEl.querySelectorAll("[data-edit-asset]").forEach((button) => {
    button.addEventListener("click", () => {
      const asset = state.assets.find((item) => item.id === button.dataset.editAsset);
      if (asset) fillAssetForm(asset);
    });
  });
  assetListEl.querySelectorAll("[data-delete-asset]").forEach((button) => {
    button.addEventListener("click", async () => {
      await request(`/api/v1/assets/${button.dataset.deleteAsset}`, { method: "DELETE" });
      await hydrateProjectWorkspace();
      renderAll();
    });
  });
}

function fillTodoForm(todo) {
  todoForm.todo_id.value = todo.id;
  todoForm.title.value = todo.title;
  todoForm.description.value = todo.description;
  todoForm.priority.value = todo.priority;
  todoForm.status.value = todo.status;
}

function fillAssetForm(asset) {
  assetForm.asset_id.value = asset.id;
  assetForm.title.value = asset.title;
  assetForm.asset_type.value = asset.asset_type;
  assetForm.content.value = asset.content;
}

function resetAssetForm() {
  assetForm.reset();
  assetForm.asset_id.value = "";
  assetForm.asset_type.value = "note";
}

function resetTodoForm() {
  todoForm.reset();
  todoForm.todo_id.value = "";
  todoForm.priority.value = "medium";
  todoForm.status.value = "todo";
}

function renderTodos() {
  if (!state.selectedProjectId) {
    todoListEl.innerHTML = emptyCard("选择项目后再维护 TODO。");
    return;
  }
  if (!state.todos.length) {
    todoListEl.innerHTML = emptyCard("暂无 TODO。可以先定义研究任务。");
    return;
  }
  todoListEl.innerHTML = state.todos
    .map(
      (todo) => `
        <article class="list-item">
          <p class="item-title">${todo.title}</p>
          <p class="muted">${todo.description || "暂无描述"}</p>
          <div class="item-meta">
            <span class="${badgeClass(todo.status)}">${todo.status}</span>
            <span class="badge">${todo.priority}</span>
          </div>
          <div class="actions">
            <button class="ghost" data-edit-todo="${todo.id}">编辑</button>
            <button data-run-todo="${todo.id}">执行</button>
            <button class="danger" data-delete-todo="${todo.id}">删除</button>
          </div>
        </article>
      `,
    )
    .join("");
  todoListEl.querySelectorAll("[data-edit-todo]").forEach((button) => {
    button.addEventListener("click", () => {
      const todo = state.todos.find((item) => item.id === button.dataset.editTodo);
      if (todo) fillTodoForm(todo);
    });
  });
  todoListEl.querySelectorAll("[data-delete-todo]").forEach((button) => {
    button.addEventListener("click", async () => {
      await request(`/api/v1/todos/${button.dataset.deleteTodo}`, { method: "DELETE" });
      await hydrateProjectWorkspace();
      renderAll();
    });
  });
  todoListEl.querySelectorAll("[data-run-todo]").forEach((button) => {
    button.addEventListener("click", async () => {
      const todo = state.todos.find((item) => item.id === button.dataset.runTodo);
      if (todo) await runResearch({ user_query: todo.description || todo.title, todo_id: todo.id });
    });
  });
}

function renderRuns() {
  if (!state.selectedProjectId) {
    runListEl.innerHTML = emptyCard("选择项目后查看运行记录。");
    return;
  }
  if (!state.runs.length) {
    runListEl.innerHTML = emptyCard("暂无运行记录。");
    return;
  }
  runListEl.innerHTML = state.runs
    .map(
      (run) => `
        <article class="list-item">
          <p class="item-title">${run.query}</p>
          <p class="muted">${run.answer_preview || "暂无答案"}</p>
          <div class="item-meta">
            <span class="${badgeClass(run.status)}">${run.status}</span>
            <span class="badge">${new Date(run.created_at).toLocaleString()}</span>
          </div>
          <div class="actions">
            <button class="ghost" data-run-detail="${run.id}">查看详情</button>
          </div>
        </article>
      `,
    )
    .join("");
  runListEl.querySelectorAll("[data-run-detail]").forEach((button) => {
    button.addEventListener("click", async () => {
      openRunModal(await request(`/api/v1/runs/${button.dataset.runDetail}`));
    });
  });
}

function renderMemory() {
  if (!state.selectedProjectId) {
    memoryListEl.innerHTML = emptyCard("选择项目后查看长期记忆。");
    return;
  }
  if (!state.memory.length) {
    memoryListEl.innerHTML = emptyCard("暂无长期记忆。完成第一次运行后会自动沉淀。");
    return;
  }
  memoryListEl.innerHTML = state.memory
    .map(
      (item) => `
        <article class="list-item">
          <p class="item-title">${item.memory_type}.${item.memory_key}</p>
          <p class="muted">${item.memory_value}</p>
        </article>
      `,
    )
    .join("");
}

function renderRunResult() {
  const latest = state.runs[0];
  if (!latest) {
    runResultEl.innerHTML = emptyCard("执行完成后，请到右侧运行记录点击“查看详情”，系统会以弹窗方式展示结果。");
    return;
  }
  runResultEl.innerHTML = `
    <div class="empty">
      最新运行已保存到右侧“运行记录”。
      点击“查看详情”会以弹窗展示答案、计划、引用和记忆更新。
    </div>
  `;
}

function renderAll() {
  renderDashboard();
  renderProviderConfig();
  renderProjects();
  renderProjectHeader();
  renderAssets();
  renderTodos();
  renderRuns();
  renderMemory();
  renderRunResult();
}

async function loadProjects() {
  state.projects = await request("/api/v1/projects");
  state.selectedProjectId = state.selectedProjectId || state.projects[0]?.id || "";
}

async function hydrateProjectWorkspace() {
  if (!state.selectedProjectId) {
    state.assets = [];
    state.todos = [];
    state.runs = [];
    state.memory = [];
    state.currentRun = null;
    return;
  }
  const [assets, todos, runs, memory] = await Promise.all([
    request(`/api/v1/projects/${state.selectedProjectId}/assets`),
    request(`/api/v1/projects/${state.selectedProjectId}/todos`),
    request(`/api/v1/projects/${state.selectedProjectId}/runs`),
    request(`/api/v1/projects/${state.selectedProjectId}/memory`),
  ]);
  state.assets = assets;
  state.todos = todos;
  state.runs = runs;
  state.memory = memory;
}

async function hydrateDashboard() {
  state.dashboard = await request("/api/v1/dashboard");
}

async function hydrateConfig() {
  state.config = await request("/api/v1/config/providers");
}

async function runResearch(payload) {
  if (!state.selectedProjectId) {
    alert("请先创建或选择项目。");
    return;
  }
  runStatusEl.textContent = "正在执行 plan-and-solve 任务...";
  const run = await request(`/api/v1/projects/${state.selectedProjectId}/run`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  state.currentRun = run;
  runStatusEl.textContent = "执行完成";
  await hydrateDashboard();
  await loadProjects();
  await hydrateProjectWorkspace();
  renderAll();
  openRunModal(run);
}

projectForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(projectForm);
  await request("/api/v1/projects", {
    method: "POST",
    body: JSON.stringify({
      title: formData.get("title"),
      description: formData.get("description") || "",
      status: "active",
    }),
  });
  projectForm.reset();
  await hydrateDashboard();
  await loadProjects();
  await hydrateProjectWorkspace();
  renderAll();
});

assetForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.selectedProjectId) {
    alert("请先选择项目。");
    return;
  }
  const formData = new FormData(assetForm);
  const assetId = formData.get("asset_id");
  const url = assetId ? `/api/v1/assets/${assetId}` : `/api/v1/projects/${state.selectedProjectId}/assets`;
  await request(url, {
    method: assetId ? "PATCH" : "POST",
    body: JSON.stringify({
      title: formData.get("title"),
      asset_type: formData.get("asset_type"),
      content: formData.get("content"),
    }),
  });
  resetAssetForm();
  await hydrateDashboard();
  await loadProjects();
  await hydrateProjectWorkspace();
  renderAll();
});

assetResetButton.addEventListener("click", resetAssetForm);

assetUploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.selectedProjectId) {
    alert("请先选择项目。");
    return;
  }
  const formData = new FormData(assetUploadForm);
  await request(`/api/v1/projects/${state.selectedProjectId}/assets/upload-text`, {
    method: "POST",
    body: formData,
  });
  assetUploadForm.reset();
  await hydrateDashboard();
  await loadProjects();
  await hydrateProjectWorkspace();
  renderAll();
});

todoForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.selectedProjectId) {
    alert("请先选择项目。");
    return;
  }
  const formData = new FormData(todoForm);
  const payload = {
    title: formData.get("title"),
    description: formData.get("description") || "",
    priority: formData.get("priority") || "medium",
    status: formData.get("status") || "todo",
  };
  const todoId = formData.get("todo_id");
  const url = todoId ? `/api/v1/todos/${todoId}` : `/api/v1/projects/${state.selectedProjectId}/todos`;
  await request(url, {
    method: todoId ? "PATCH" : "POST",
    body: JSON.stringify(payload),
  });
  resetTodoForm();
  await hydrateDashboard();
  await loadProjects();
  await hydrateProjectWorkspace();
  renderAll();
});

todoResetButton.addEventListener("click", resetTodoForm);

runForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = new FormData(runForm).get("user_query")?.toString().trim();
  if (!query) {
    alert("请输入研究问题。");
    return;
  }
  await runResearch({ user_query: query });
  runForm.reset();
});

async function bootstrap() {
  await Promise.all([hydrateConfig(), hydrateDashboard(), loadProjects()]);
  await hydrateProjectWorkspace();
  renderAll();
}

runDetailCloseButton.addEventListener("click", closeRunModal);
runDetailModalEl.querySelectorAll("[data-close-run-modal]").forEach((target) => {
  target.addEventListener("click", closeRunModal);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !runDetailModalEl.hidden) {
    closeRunModal();
  }
});

bootstrap().catch((error) => {
  runStatusEl.textContent = error.message;
});
