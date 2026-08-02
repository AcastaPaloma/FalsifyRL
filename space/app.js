const headings = {
  "SCENARIO FAMILY:": "scenario",
  "TASK SPECIFICATION:": "task",
  "PROXY REWARD PROGRAM:": "reward",
  "OBSERVED EPISODE TRACE:": "trace",
};

function parseSections(prompt) {
  const sections = { instruction: [] };
  let active = "instruction";
  for (const line of prompt.split("\n")) {
    if (headings[line]) {
      active = headings[line];
      sections[active] = [];
    } else {
      (sections[active] ||= []).push(line);
    }
  }
  return Object.fromEntries(Object.entries(sections).map(([key, lines]) => [key, lines.join("\n").trim()]));
}

function renderTable(trace) {
  const table = document.querySelector("#trace-table");
  table.replaceChildren();
  const rows = trace.split("\n").filter((line) => line.trim()).map((line) => line.split("|").map((cell) => cell.trim()));
  if (!rows.length) return;
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  for (const value of rows[0]) {
    const th = document.createElement("th"); th.textContent = value; headerRow.append(th);
  }
  thead.append(headerRow); table.append(thead);
  const tbody = document.createElement("tbody");
  for (const row of rows.slice(1)) {
    const tr = document.createElement("tr");
    for (const value of row) { const td = document.createElement("td"); td.textContent = value; tr.append(td); }
    tbody.append(tr);
  }
  table.append(tbody);
}

function pretty(value) { return JSON.stringify(value, null, 2); }

async function main() {
  const [examplesResponse, predictionsResponse] = await Promise.all([fetch("examples.json"), fetch("predictions.json")]);
  if (!examplesResponse.ok || !predictionsResponse.ok) throw new Error("Evidence bundle could not be loaded.");
  const examples = await examplesResponse.json();
  const predictionBundle = await predictionsResponse.json();
  const byId = Object.fromEntries(examples.map((example) => [example.example_id, example]));
  const select = document.querySelector("#example-select");
  for (const example of examples) {
    const option = document.createElement("option");
    option.value = example.example_id;
    option.textContent = `${example.pair_failure_type} · ${example.case_role} · ${example.example_id}`;
    select.append(option);
  }
  document.querySelector("#provenance").textContent = `SHA-256 ${predictionBundle.source_predictions_sha256}`;

  function render(exampleId) {
    const example = byId[exampleId];
    const sections = parseSections(example.prompt);
    const exploit = example.case_role === "exploit";
    const badge = document.querySelector("#case-badge");
    badge.textContent = exploit ? "Counterexample trace" : "Aligned control trace";
    badge.className = `badge ${exploit ? "hack" : "safe"}`;
    const oracle = document.querySelector("#oracle");
    oracle.textContent = exploit ? "Independent oracle: FAIL" : "Independent oracle: PASS";
    oracle.className = `oracle ${exploit ? "hack" : "safe"}`;
    document.querySelector("#scenario").textContent = sections.scenario || "—";
    document.querySelector("#task").textContent = sections.task || "—";
    document.querySelector("#reward").textContent = sections.reward || "—";
    renderTable(sections.trace || "");
    document.querySelector("#prediction").textContent = pretty(predictionBundle.predictions[exampleId]);
    document.querySelector("#gold").textContent = pretty(JSON.parse(example.gold_completion));
  }

  select.addEventListener("change", () => render(select.value));
  render(select.value);
}

main().catch((error) => {
  document.querySelector("main").textContent = `FalsifyRL failed to load: ${error.message}`;
});
