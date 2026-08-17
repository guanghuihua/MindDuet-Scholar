import * as pdfjsLib from "/static/vendor/pdfjs/pdf.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc = "/static/vendor/pdfjs/pdf.worker.mjs";

const root = document.querySelector(".pdf-reader");
const canvas = root.querySelector('[data-role="canvas"]');
const textLayer = root.querySelector('[data-role="text-layer"]');
const pageShell = root.querySelector('[data-role="page-shell"]');
const pageInput = root.querySelector('[data-role="page-input"]');
const pageCount = root.querySelector('[data-role="page-count"]');
const scaleSelect = root.querySelector('[data-role="scale-select"]');
const statusLine = root.querySelector('[data-role="status"]');
const selectedText = root.querySelector('[data-role="selected-text"]');
const contextPage = root.querySelector('[data-role="context-page"]');
const tabButtons = root.querySelectorAll("[data-tab]");
const tabPanels = root.querySelectorAll("[data-panel]");
const assistantMessages = root.querySelector('[data-role="assistant-messages"]');
const assistantForm = root.querySelector('[data-role="assistant-form"]');
const assistantInput = root.querySelector('[data-role="assistant-input"]');
const assistantSend = root.querySelector('[data-role="assistant-send"]');
const assistantStatus = root.querySelector('[data-role="assistant-status"]');
const assistantStop = root.querySelector('[data-assistant-action="stop"]');
const assistantNew = root.querySelector('[data-assistant-action="new"]');
const quickPrompts = root.querySelectorAll("[data-prompt]");

let pdf = null;
let currentPage = 1;
let currentScale = Number(scaleSelect.value);
let currentPageText = "";
let rendering = false;
let pendingPage = null;
let assistantBusy = false;
let assistantAbortController = null;

function setStatus(message) {
  statusLine.textContent = message;
}

function selectionInsideReader() {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0) {
    return "";
  }
  const range = selection.getRangeAt(0);
  if (!textLayer.contains(range.commonAncestorContainer)) {
    return "";
  }
  return selection.toString().trim();
}

async function saveContext() {
  const selection = selectionInsideReader();
  if (selection) {
    selectedText.value = selection;
  }
  const response = await fetch(root.dataset.contextUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      document_id: Number(root.dataset.documentId),
      page: currentPage,
      scale: currentScale,
      selected_text: selectedText.value,
      page_text: currentPageText,
    }),
  });
  if (!response.ok) {
    throw new Error("Context was not saved.");
  }
  setStatus(selectedText.value.trim() ? "Saved page and selection." : "Saved current page.");
}

function positionTextItem(span, item, viewport) {
  const tx = pdfjsLib.Util.transform(viewport.transform, item.transform);
  const fontHeight = Math.hypot(tx[2], tx[3]);
  span.style.left = `${tx[4]}px`;
  span.style.top = `${tx[5] - fontHeight}px`;
  span.style.fontSize = `${fontHeight}px`;
  span.style.transform = `scaleX(${item.width ? (item.width * viewport.scale) / Math.max(span.textContent.length * fontHeight * 0.52, 1) : 1})`;
}

async function renderPage(pageNumber) {
  if (rendering) {
    pendingPage = pageNumber;
    return;
  }
  rendering = true;
  try {
    const page = await pdf.getPage(pageNumber);
    const viewport = page.getViewport({ scale: currentScale });
    const context = canvas.getContext("2d");
    const outputScale = window.devicePixelRatio || 1;
    const cssWidth = Math.floor(viewport.width);
    const cssHeight = Math.floor(viewport.height);
    canvas.width = Math.floor(viewport.width * outputScale);
    canvas.height = Math.floor(viewport.height * outputScale);
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;
    pageShell.style.width = `${cssWidth}px`;
    pageShell.style.height = `${cssHeight}px`;
    textLayer.replaceChildren();
    textLayer.style.width = `${cssWidth}px`;
    textLayer.style.height = `${cssHeight}px`;

    await page.render({
      canvasContext: context,
      viewport,
      transform: outputScale !== 1 ? [outputScale, 0, 0, outputScale, 0, 0] : null,
    }).promise;

    const text = await page.getTextContent();
    currentPageText = text.items.map((item) => item.str).join(" ").replace(/\s+/g, " ").trim();
    const fragment = document.createDocumentFragment();
    for (const item of text.items) {
      const span = document.createElement("span");
      span.textContent = item.str;
      span.setAttribute("role", "presentation");
      positionTextItem(span, item, viewport);
      fragment.appendChild(span);
    }
    textLayer.appendChild(fragment);

    currentPage = pageNumber;
    pageInput.value = String(currentPage);
    contextPage.textContent = String(currentPage);
    try {
      await saveContext();
    } catch (error) {
      setStatus(`Save failed: ${error.message}`);
    }
  } finally {
    rendering = false;
    if (pendingPage !== null) {
      const next = pendingPage;
      pendingPage = null;
      await renderPage(next);
    }
  }
}

function queuePage(pageNumber) {
  const bounded = Math.min(Math.max(1, pageNumber), pdf.numPages);
  renderPage(bounded).catch((error) => setStatus(`Render failed: ${error.message}`));
}

root.querySelector('[data-action="prev"]').addEventListener("click", () => queuePage(currentPage - 1));
root.querySelector('[data-action="next"]').addEventListener("click", () => queuePage(currentPage + 1));
root.querySelector('[data-action="save-selection"]').addEventListener("click", () => {
  saveContext().catch((error) => setStatus(`Save failed: ${error.message}`));
});

pageInput.addEventListener("change", () => queuePage(Number(pageInput.value)));
scaleSelect.addEventListener("change", () => {
  currentScale = Number(scaleSelect.value);
  queuePage(currentPage);
});

document.addEventListener("selectionchange", () => {
  const selection = selectionInsideReader();
  if (selection) {
    selectedText.value = selection;
  }
});

pdfjsLib.getDocument({ url: root.dataset.pdfUrl }).promise
  .then((loadedPdf) => {
    pdf = loadedPdf;
    pageCount.textContent = `/ ${pdf.numPages}`;
    pageInput.max = String(pdf.numPages);
    return renderPage(1);
  })
  .catch((error) => setStatus(`PDF load failed: ${error.message}`));

function selectTab(name) {
  for (const button of tabButtons) {
    const selected = button.dataset.tab === name;
    button.setAttribute("aria-selected", String(selected));
  }
  for (const panel of tabPanels) {
    panel.hidden = panel.dataset.panel !== name;
  }
}

for (const button of tabButtons) {
  button.addEventListener("click", () => selectTab(button.dataset.tab));
}

function setAssistantStatus(message, kind = "") {
  assistantStatus.textContent = message;
  assistantStatus.dataset.kind = kind;
}

function setAssistantBusy(busy) {
  assistantBusy = busy;
  assistantSend.disabled = busy;
  assistantInput.disabled = busy;
  assistantStop.disabled = !busy;
  assistantNew.disabled = busy;
  for (const button of quickPrompts) {
    button.disabled = busy;
  }
}

function scrollAssistantToEnd() {
  assistantMessages.scrollTop = assistantMessages.scrollHeight;
}

function appendMessage(role, body = "") {
  assistantMessages.querySelector(".assistant-empty")?.remove();
  const message = document.createElement("article");
  message.className = `assistant-message ${role}`;
  const label = document.createElement("span");
  label.className = "assistant-message-role";
  label.textContent = role === "user" ? "You" : "Codex";
  const content = document.createElement("div");
  content.className = "assistant-message-body";
  content.textContent = body;
  message.append(label, content);
  assistantMessages.appendChild(message);
  scrollAssistantToEnd();
  return content;
}

async function typesetMath(element) {
  if (!window.MathJax?.typesetPromise || !element.textContent.trim()) {
    return;
  }
  window.MathJax.typesetClear?.([element]);
  await window.MathJax.typesetPromise([element]);
}

async function typesetAllAssistantMath() {
  const bodies = assistantMessages.querySelectorAll(".assistant-message.assistant .assistant-message-body");
  for (const body of bodies) {
    await typesetMath(body);
  }
}

async function loadAssistantHistory() {
  try {
    const response = await fetch(
      `${root.dataset.assistantStatusUrl}?document_id=${encodeURIComponent(root.dataset.documentId)}`,
    );
    if (!response.ok) {
      throw new Error(`Status request failed (${response.status})`);
    }
    const state = await response.json();
    assistantMessages.replaceChildren();
    if (state.messages.length === 0) {
      const empty = document.createElement("p");
      empty.className = "assistant-empty";
      empty.dataset.role = "assistant-empty";
      empty.textContent = "No messages yet.";
      assistantMessages.appendChild(empty);
    } else {
      for (const message of state.messages) {
        appendMessage(message.role, message.body);
      }
    }
    setAssistantStatus(state.active ? "Codex is responding..." : "Local Codex ready.", "ready");
    await typesetAllAssistantMath();
  } catch (error) {
    setAssistantStatus(`Codex unavailable: ${error.message}`, "error");
  }
}

function parseEventBlock(block) {
  let eventName = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (dataLines.length === 0) {
    return null;
  }
  return { name: eventName, data: JSON.parse(dataLines.join("\n")) };
}

async function readAssistantStream(response, onEvent) {
  if (!response.body) {
    throw new Error("Streaming response is unavailable in this browser.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = parseEventBlock(block);
      if (event) {
        onEvent(event);
      }
      boundary = buffer.indexOf("\n\n");
    }
    if (done) {
      if (buffer.trim()) {
        const event = parseEventBlock(buffer.trim());
        if (event) {
          onEvent(event);
        }
      }
      break;
    }
  }
}

async function sendAssistantMessage(explicitMessage = null) {
  if (assistantBusy) {
    return;
  }
  const message = (explicitMessage || assistantInput.value).trim();
  if (!message) {
    assistantInput.focus();
    return;
  }

  setAssistantBusy(true);
  let answer = null;
  let receivedError = false;

  try {
    setAssistantStatus("Saving page context...", "working");
    await saveContext();
    appendMessage("user", message);
    answer = appendMessage("assistant");
    assistantInput.value = "";
    assistantAbortController = new AbortController();
    setAssistantStatus("Codex is thinking...", "working");
    const response = await fetch(root.dataset.assistantStreamUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_id: Number(root.dataset.documentId), message }),
      signal: assistantAbortController.signal,
    });
    if (!response.ok) {
      const failure = await response.json().catch(() => ({}));
      throw new Error(failure.detail || `Request failed (${response.status})`);
    }

    await readAssistantStream(response, ({ name, data }) => {
      if (name === "start") {
        setAssistantStatus("Codex is responding...", "working");
      } else if (name === "delta") {
        answer.textContent += data.delta;
        scrollAssistantToEnd();
      } else if (name === "replace") {
        answer.textContent = data.text;
      } else if (name === "error") {
        receivedError = true;
        answer.textContent = data.message;
        answer.closest(".assistant-message").classList.add("error");
        setAssistantStatus("Response failed.", "error");
      } else if (name === "done") {
        setAssistantStatus("Local Codex ready.", "ready");
      }
    });
    if (!receivedError) {
      await typesetMath(answer);
    }
  } catch (error) {
    if (!answer) {
      answer = appendMessage("assistant");
    }
    if (error.name === "AbortError") {
      if (!answer.textContent.trim()) {
        answer.textContent = "Response stopped.";
      }
      setAssistantStatus("Response stopped.");
    } else {
      answer.textContent = error.message;
      answer.closest(".assistant-message").classList.add("error");
      setAssistantStatus("Response failed.", "error");
    }
  } finally {
    assistantAbortController = null;
    setAssistantBusy(false);
    scrollAssistantToEnd();
    assistantInput.focus();
  }
}

assistantForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendAssistantMessage().catch((error) => setAssistantStatus(error.message, "error"));
});

assistantInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    assistantForm.requestSubmit();
  }
});

for (const button of quickPrompts) {
  button.addEventListener("click", () => {
    sendAssistantMessage(button.dataset.prompt).catch((error) => setAssistantStatus(error.message, "error"));
  });
}

assistantStop.addEventListener("click", () => {
  fetch(root.dataset.assistantInterruptUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id: Number(root.dataset.documentId) }),
  }).catch(() => {});
  assistantAbortController?.abort();
});

assistantNew.addEventListener("click", async () => {
  setAssistantStatus("Starting a new conversation...", "working");
  try {
    const response = await fetch(root.dataset.assistantResetUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_id: Number(root.dataset.documentId) }),
    });
    if (!response.ok) {
      throw new Error(`Reset failed (${response.status})`);
    }
    assistantMessages.replaceChildren();
    const empty = document.createElement("p");
    empty.className = "assistant-empty";
    empty.dataset.role = "assistant-empty";
    empty.textContent = "No messages yet.";
    assistantMessages.appendChild(empty);
    setAssistantStatus("New conversation ready.", "ready");
  } catch (error) {
    setAssistantStatus(error.message, "error");
  }
});

loadAssistantHistory();
window.addEventListener("load", () => typesetAllAssistantMath());
