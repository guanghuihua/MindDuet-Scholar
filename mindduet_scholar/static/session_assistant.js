const root = document.querySelector(".session-chat");
const messages = root.querySelector('[data-role="assistant-messages"]');
const form = root.querySelector('[data-role="assistant-form"]');
const input = root.querySelector('[data-role="assistant-input"]');
const send = root.querySelector('[data-role="assistant-send"]');
const statusLine = root.querySelector('[data-role="assistant-status"]');
const stop = root.querySelector('[data-assistant-action="stop"]');

let busy = false;
let abortController = null;

function setStatus(message, kind = "") {
  statusLine.textContent = message;
  statusLine.dataset.kind = kind;
}

function setBusy(value) {
  busy = value;
  send.disabled = value;
  input.disabled = value;
  stop.disabled = !value;
}

function scrollToEnd() {
  window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "auto" });
}

function appendMessage(role, body = "", renderedBody = null) {
  messages.querySelector(".assistant-empty")?.remove();
  const message = document.createElement("article");
  message.className = `assistant-message ${role}`;
  const label = document.createElement("span");
  label.className = "assistant-message-role";
  label.textContent = role === "user" ? "You" : "Codex";
  const content = document.createElement("div");
  content.className = "assistant-message-body";
  if (renderedBody) {
    content.innerHTML = renderedBody;
    content.classList.add("rendered");
  } else {
    content.textContent = body;
  }
  message.append(label, content);
  messages.appendChild(message);
  scrollToEnd();
  return content;
}

async function typesetMath(element) {
  if (!window.MathJax?.typesetPromise || !element.textContent.trim()) {
    return;
  }
  window.MathJax.typesetClear?.([element]);
  await window.MathJax.typesetPromise([element]);
}

async function typesetAllMath() {
  const bodies = messages.querySelectorAll(".assistant-message.assistant .assistant-message-body");
  for (const body of bodies) {
    await typesetMath(body);
  }
}

async function loadHistory() {
  try {
    const response = await fetch(root.dataset.assistantStatusUrl);
    if (!response.ok) {
      throw new Error(`Status request failed (${response.status})`);
    }
    const state = await response.json();
    messages.replaceChildren();
    if (state.messages.length === 0) {
      const empty = document.createElement("p");
      empty.className = "assistant-empty";
      empty.textContent = "Ask the first question for this learning goal.";
      messages.appendChild(empty);
    } else {
      for (const message of state.messages) {
        appendMessage(message.role, message.body, message.rendered_body);
      }
    }
    setStatus(state.active ? "Codex is responding in this session..." : "Local Codex ready.", "ready");
    await typesetAllMath();
    scrollToEnd();
  } catch (error) {
    setStatus(`Codex unavailable: ${error.message}`, "error");
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
  return dataLines.length ? { name: eventName, data: JSON.parse(dataLines.join("\n")) } : null;
}

async function readStream(response, onEvent) {
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
      const event = parseEventBlock(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
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

async function sendMessage() {
  if (busy) {
    return;
  }
  const question = input.value.trim();
  if (!question) {
    input.focus();
    return;
  }

  setBusy(true);
  appendMessage("user", question);
  let answer = appendMessage("assistant");
  let receivedError = false;
  input.value = "";
  abortController = new AbortController();
  setStatus("Codex is thinking...", "working");

  try {
    const response = await fetch(root.dataset.assistantStreamUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question }),
      signal: abortController.signal,
    });
    if (!response.ok) {
      const failure = await response.json().catch(() => ({}));
      throw new Error(failure.detail || `Request failed (${response.status})`);
    }
    await readStream(response, ({ name, data }) => {
      if (name === "start") {
        setStatus("Codex is responding...", "working");
      } else if (name === "delta") {
        answer.textContent += data.delta;
        scrollToEnd();
      } else if (name === "replace") {
        answer.textContent = data.text;
      } else if (name === "error") {
        receivedError = true;
        answer.textContent = data.message;
        answer.closest(".assistant-message").classList.add("error");
        setStatus("Response failed.", "error");
      } else if (name === "done") {
        if (data.rendered_body) {
          answer.innerHTML = data.rendered_body;
          answer.classList.add("rendered");
        }
        setStatus("Saved in this session.", "ready");
      }
    });
    if (!receivedError) {
      await typesetMath(answer);
    }
  } catch (error) {
    if (error.name === "AbortError") {
      if (!answer.textContent.trim()) {
        answer.textContent = "Response stopped.";
      }
      setStatus("Response stopped.");
    } else {
      answer.textContent = error.message;
      answer.closest(".assistant-message").classList.add("error");
      setStatus("Response failed.", "error");
    }
  } finally {
    abortController = null;
    setBusy(false);
    scrollToEnd();
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage().catch((error) => setStatus(error.message, "error"));
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    form.requestSubmit();
  }
});

stop.addEventListener("click", () => {
  fetch(root.dataset.assistantInterruptUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: Number(root.dataset.sessionId) }),
  }).catch(() => {});
  abortController?.abort();
});

loadHistory();
window.addEventListener("load", () => typesetAllMath());
