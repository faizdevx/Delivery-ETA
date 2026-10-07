const $ = (id) => document.getElementById(id);
const result = $("result");

function show(html, isError) {
  result.hidden = false;
  result.className = isError ? "error" : "";
  result.innerHTML = html;
}
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

function fill(select, values) {
  select.innerHTML = "";
  for (const v of values) select.add(new Option(v, v));
}

async function init() {
  try {
    const r = await fetch("/options");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const o = await r.json();
    fill($("pickup_zone"), o.pickup_zones);
    fill($("dropoff_zone"), o.dropoff_zones);
    fill($("taxi_color"), o.taxi_colors);
    $("passengers").min = o.passengers[0];
    $("passengers").max = o.passengers[1];
    $("pickup_datetime").value = o.training_pickup_range[1].slice(0, 16).replace(" ", "T");
  } catch (e) {
    show(`Could not load model options (${esc(e.message)}). Is the model trained and the API running?`, true);
    $("go").disabled = true;
  }
}

$("form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const pax = Number($("passengers").value);
  if (!$("pickup_datetime").value) return show("Pick a pickup date and time.", true);
  if (!Number.isInteger(pax) || pax < Number($("passengers").min) || pax > Number($("passengers").max))
    return show("Passengers must be a whole number in the allowed range.", true);
  const body = {
    pickup_datetime: $("pickup_datetime").value.length === 16 ? $("pickup_datetime").value + ":00" : $("pickup_datetime").value,
    pickup_zone: $("pickup_zone").value,
    dropoff_zone: $("dropoff_zone").value,
    passengers: pax,
    taxi_color: $("taxi_color").value,
  };
  $("go").disabled = true;
  try {
    const r = await fetch("/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const d = data.detail;
      const msg = Array.isArray(d) ? d.map((x) => `${x.loc.slice(1).join(".")}: ${x.msg}`).join("; ") : (d || `HTTP ${r.status}`);
      return show(`Request failed: ${esc(msg)}`, true);
    }
    show(`<div class="big">${data.predicted_trip_duration_minutes.toFixed(1)} min</div>
          <small>model ${esc(data.model_version)}. Point estimate only &mdash; no uncertainty interval is provided.</small>`, false);
  } catch (e) {
    show(`Could not reach the API: ${esc(e.message)}`, true);
  } finally {
    $("go").disabled = false;
  }
});

init();
