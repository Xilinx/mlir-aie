# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Exercise the page's real data, chart and table code under node, without a browser."""

import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "utils/kernel_checks/index.html"


@pytest.fixture
def page():
    node = shutil.which("node")
    if not node:
        # Set by a CI job with node, so the page is tested somewhere.
        if os.environ.get("MLIR_AIE_REQUIRE_NODE"):
            pytest.fail("MLIR_AIE_REQUIRE_NODE is set but node is not on PATH")
        pytest.skip("node is required to test the kernel checks page")
    html = PAGE.read_text(encoding="utf-8")
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    setup = """
const assert = require('node:assert/strict');
// Disable the automatic data fetch and capture the Chart.js configuration.
global.fetch = async () => ({ ok: false, status: 404 });
// Just enough DOM for the page to build its cards and tables.
class El {
  constructor(tag) { this.tag = tag; this.children = []; this.on = {}; this.hidden = false; }
  append(...nodes) {
    for (const n of nodes) if (n instanceof El) n.parentNode = this;
    this.children.push(...nodes);
  }
  insertBefore(node, before) {
    const at = this.children.indexOf(before);
    this.children.splice(at < 0 ? this.children.length : at, 0, node);
    node.parentNode = this;
  }
  remove() { if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(c => c !== this); }
  replaceChildren(...nodes) { this.children = nodes; }
  addEventListener(type, f) { this.on[type] = f; }
  querySelectorAll() { return []; }
  scrollIntoView() { this.scrolled = true; }
  get text() {
    return (this.textContent || '') +
      this.children.map(c => typeof c === 'string' ? c : c.text).join('');
  }
}
const byId = new Map();
global.document = {
  // Only ids the page has, so a typo is null here as in a browser.
  getElementById: id => !PAGE_IDS.has(id) ? null : byId.get(id) || byId.set(id, new El()).get(id),
  createElement: tag => new El(tag),
  querySelector: () => new El(), querySelectorAll: () => [],
};
global.location = { hash: '' };
global.history = { replaceState: (_state, _title, url) => { location.hash = url; } };
let chart, opened;
global.Chart = function (_canvas, config) { chart = config; };
Chart.prototype.destroy = function () {};
global.IntersectionObserver = class {
  constructor(cb) { this.cb = cb; }
  observe(t) { this.cb([{ isIntersecting: true, target: t }]); }
  unobserve() {} disconnect() {}
};
global.window = { open: (...args) => { opened = args; }, addEventListener: () => {} };
global.Option = class { constructor(text, value) { this.text = text; this.value = value; } };
"""
    data = """
const commit = {
  id: 'abcdef123456', message: 'same revision\\nbody',
  timestamp: '2020-01-01T00:00:00Z', url: 'https://github.com/Xilinx/mlir-aie/commit/abcdef123456',
};
// A run record as publish.py writes it; rows as {'case/metric': [unit, value, range?]}.
const rec = (id, date, pmode, rows, provenance) => {
  const out = {};
  for (const [name, [unit, value, range]] of Object.entries(rows || {})) {
    const cut = name.lastIndexOf('/');
    (out[name.slice(0, cut)] ||= {})[name.slice(cut + 1)] = { value, unit, ...(range ? { range } : {}) };
  }
  return { id, date: new Date(date).toISOString(), commit, pmode, provenance: provenance || {}, rows: out };
};
const fromRecords = byNpu => collect(Object.entries(byNpu).flatMap(([npu, records]) => historiesOf(npu, records)));
const sm = v => v === null ? {} : { 'softmax/1024x16/bfloat16/cycles': ['cycles', v] };
db = fromRecords({
  npu1: [
    rec('t3', 300000, 'turbo', sm(121)), rec('t1', 100000, 'turbo', sm(100)),
    rec('t2', 200000, 'turbo', sm(null)), rec('t25', 250000, 'turbo', sm(110)),
    rec('p1', 100001, 'performance', sm(90)), rec('p25', 250001, 'performance', sm(95)),
  ],
  npu2: [rec('t1', 100000, 'turbo', sm(999))],
});
const s = db.series.find(s => s.npu === 'npu1');
const el0 = {};
"""

    def run(checks):
        ids = sorted(set(re.findall(r'\bid="([^"]+)"', html)))
        known = f"const PAGE_IDS = new Set({json.dumps(ids)});\n"
        # On stdin: the page outgrew the 128 KiB a single argument may hold,
        # and Windows' 32767-character command line.
        source = known + setup + script + data + checks
        env = {**os.environ, "WORKFLOWS": str(ROOT / ".github/workflows")}
        subprocess.run([node, "-"], input=source, encoding="utf-8", check=True, env=env)

    return run


def test_thresholds_match_the_report_and_color_only_past_them(page):
    html = PAGE.read_text(encoding="utf-8")
    match = re.search(r"^\s*const THRESHOLDS = (\{.*\});$", html, re.MULTILINE)
    assert match, "the page carries no THRESHOLDS constant"
    shared = json.loads(
        (ROOT / "utils/kernel_checks/thresholds.json").read_text(encoding="utf-8")
    )
    expected = {m: s for m, s in shared.items() if not m.startswith("_")}
    assert json.loads(match[1]) == expected
    page("""
assert.equal(changeClass(100, 100.9, 'cycles'), '');
assert.equal(changeClass(100, 101, 'cycles'), 'worse');
assert.equal(changeClass(100, 99.1, 'kernel_object_bytes'), '');
assert.equal(changeClass(100, 99, 'kernel_object_bytes'), 'better');
assert.equal(changeClass(100, 114, 'dispatch_us'), '');
assert.equal(changeClass(100, 115, 'dispatch_us'), 'worse');
assert.equal(changeClass(100, 114, 'compile_s'), '');
assert.equal(changeClass(100, 115, 'compile_s'), 'worse');
assert.equal(changeClass(1000, 1009, 'final_cost_mean'), '');
assert.equal(changeClass(1000, 1010, 'final_cost_mean'), 'worse');
assert.deepEqual(GATED, ['cycles', 'kernel_object_bytes', 'max_ulp_bfloat16', 'max_ulp_float32', 'not_correctly_rounded_bfloat16', 'not_correctly_rounded_float32', 'kl_bf16', 'top1_bf16', 'map50_coco128']);
""")


def test_the_direction_of_better_is_said_where_it_is_up(page):
    html = PAGE.read_text(encoding="utf-8")
    assert "Every metric is smaller-is-better" not in html
    assert "Lower is better, except where a metric says higher is better" in html
    page("""
const up = Object.keys(THRESHOLDS).filter(m => THRESHOLDS[m].higher_is_better);
assert.ok(up.includes('tok_s') && up.includes('fps_b1') && up.includes('images_passed'));
// Every metric a section lists says so when higher is better, and only then.
for (const m of sectionMetrics('performance')) {
  assert.equal(termOf(m, COMPONENT_METRICS[m]).includes('higher is better'), up.includes(m), m);
}
assert.equal(termOf('tok_s', 'x'), 'x; higher is better; counts past 5%');
assert.equal(termOf('dispatch_us', 'x'), 'x; counts past 15% and 3× its spread');
// No description says it on its own any more: the list adds it.
assert.ok(Object.values(COMPONENT_METRICS).every(about => !about.includes('higher is better')));
assert.equal(axisTitle('Tokens per second', 'tok_s'), 'Tokens per second · ↑ better');
assert.equal(axisTitle('Cycles per call', 'cycles'), 'Cycles per call · ↓ better');
""")


def test_timing_labels_name_the_statistic_the_checks_take(page):
    mobilenet = re.search(
        r'"--iters", type=int, default=(\d+)',
        (ROOT / "utils/component_checks/sa_placer_hw.py").read_text(encoding="utf-8"),
    )[1]
    page(f"""
const trendOf = id => BANDS.find(b => b.id === id && b.trend).trend;
// MobileNet takes the fastest of its timed calls; so does yolo26n.py --time,
// whose row's range gives its count, median and MAD.
assert.ok(COMPONENT_METRICS.us_per_image.includes('the fastest of {mobilenet}'));
assert.ok(COMPONENT_METRICS.e2e_us_per_image.includes('the fastest of {mobilenet}'));
assert.ok(trendOf('sa-placer-hw').every(t => !/per image/.test(t.axis) || t.axis.includes('fastest')));
assert.ok(['us_per_image_b1', 'dispatch_us_per_image_b1'].every(m => COMPONENT_METRICS[m].includes('the fastest of its timed calls')));
assert.ok(trendOf('yolo26n-hw').every(t => !/per image/.test(t.axis) || t.axis.includes('fastest')));
assert.ok(!/the median of/.test(COMPONENT_METRICS.us_per_image_b1 + trendOf('yolo26n-hw')[0].about));
// Llama's timed run takes the fastest of 20 dispatches; generate.py's decode the median of 5 requests.
assert.ok(COMPONENT_METRICS.tok_s.includes('fastest 4-token dispatch of 20'));
assert.ok(COMPONENT_METRICS.ms_per_token.includes('fastest 4-token dispatch of 20'));
assert.ok(COMPONENT_METRICS.decode_tok_s.includes('the median of its 5 requests'));
assert.ok(trendOf('llama-chain-hw')[0].axis.includes('fastest of 20 dispatches'));
assert.ok(!/median dispatch|dispatch: median/.test(trendOf('llama-chain-hw').map(t => t.axis + ' ' + (t.about || '')).join(' ')));
// A host-observed dispatch is never called NPU time; NPU time is the dispatch less its baseline.
assert.ok(!/NPU time/.test(COMPONENT_METRICS.us_per_image + COMPONENT_METRICS.ms_per_token + COMPONENT_METRICS.dispatch_us_per_image_b1));
assert.ok(COMPONENT_METRICS.npu_us_per_image.includes("less the same run's empty-dispatch baseline"));
// Every timing tab's axis names its statistic.
for (const band of BANDS.filter(b => b.trend)) {{
  for (const t of band.trend) {{
    if (/\\((µs|ms|s)\\b/.test(t.axis)) assert.match(t.axis, /fastest|median|mean|one (run|compile)/, t.axis);
  }}
}}
""")


def test_the_mobilenet_repro_builds_what_the_nightly_builds(page):
    producer = (ROOT / "utils/component_checks/sa_placer_hw.py").read_text(
        encoding="utf-8"
    )
    # The design's default effort is sa_placer_flags' own; full is 1.0.
    common = (
        ROOT / "programming_examples/ml/mobilenet/bottleneck/_common.py"
    ).read_text(encoding="utf-8")
    efforts = {
        "default": float(
            re.search(r"def sa_placer_flags\(.*effort=([\d.]+)", common)[1]
        ),
        "full": float(re.search(r'^\s+"full": ([\d.]+),$', producer, re.M)[1]),
    }
    run_seed = producer.split("def run_seed(", 1)[1].split("\ndef ", 1)[0]
    flags = sorted(set(re.findall(r'"(-{1,2}[a-z][\w-]*)"', run_seed)))
    html = PAGE.read_text(encoding="utf-8")
    help_text = re.search(
        r"<pre>(python -m mobilenet\.aie2_mobilenet_iron [^<]*)</pre>", html
    )[1]
    page(f"""
assert.deepEqual(EFFORT_VALUES, {json.dumps(efforts)});
const repro = HW_RUNS.repro(null, '3', '64', 'full');
for (const flag of {json.dumps(flags)}) {{
  assert.ok(repro.split(' ').includes(flag), flag);
  assert.ok({json.dumps(help_text)}.split(' ').includes(flag), flag);
}}
// --batch builds a static design the nightly no longer runs.
assert.ok(!repro.includes('--batch'));
assert.equal(Number(repro.split('--sa-effort ')[1].split(' ')[0]), {efforts["full"]});
""")


def test_slower_since_names_the_night_it_moved_not_the_window_start(page):
    page(RUNS + """
// bn_conv2dk1_skip on npu1, as published: flat from 10 Oct, a step on 20 Oct.
const day = d => Date.parse(`2026-10-${String(d).padStart(2, '0')}T06:00:00Z`);
const cyclesOn = values => fromRecords({ npu1: values.map(([d, v], i) => rec(String(i), day(d), 'turbo', { 'skip/1/i8/cycles': ['cycles', v] })) });
let w = kernelWindow(cyclesOn([[10, 1556], [11, 1556], [15, 1556], [19, 1556], [20, 1621]]), 'npu1', 'skip');
assert.deepEqual([w.worse, shortDate(w.since)], [1, '20 Oct']);
assert.deepEqual(w.moves, ['skip/1/i8: 1,556 on 10 Oct → 1,621, past the threshold from 20 Oct']);
// A move that held since an earlier night is dated from that night.
w = kernelWindow(cyclesOn([[10, 100], [12, 120], [14, 121], [20, 122]]), 'npu1', 'skip');
assert.equal(shortDate(w.since), '12 Oct');
// Nothing moved: no date.
assert.equal(kernelWindow(cyclesOn([[10, 100], [20, 100.5]]), 'npu1', 'skip').since, null);
// The window spans 30 nightlies; a strip of recent runs, two weeks of them.
assert.equal(WINDOW_NIGHTS, 30);
assert.equal(STRIP_RUNS, 14);
const many = { target: 'npu1', runs: Array.from({ length: 40 }, (_, i) => summary(String(i), new Date(day(1) + i * 3600e3).toISOString(), 'turbo', {})) };
const strip = nightStrip({ runs: many.runs, run: { id: '39' } });
assert.equal(strip.children.length - 1, STRIP_RUNS);
// A table's change says what it is against on hover.
const vals = glanceValues({ value: 110, unit: 'cycles', change: 0.1, delta: 10, cls: 'worse', since: day(19) }, 'cycles', true);
assert.equal(vals.title, 'was 100 on 19 Oct');
""")


def test_times_compare_and_chart_per_part_and_provenance_is_marked(page):
    page(COMPONENT_RUNS + LLAMA_NIGHTS + """
// Llama on alternating parts: Strix 4.7, Krackan 4.0, Strix 4.37 (-7%), Krackan 4.0.
const part = (id, day, tok_s, device) => ({ ...llamaRecord(id, `2026-10-0${day}T06:30:00Z`, tok_s), device: device.replace('NPU ', ''),
  device_raw: device, provenance: { device, peano: 'p1', kernels: `k${day}` } });
const swaps = [part('1', 1, 4.7, 'NPU Strix'), part('2', 2, 4.0, 'NPU Krackan 1'), part('3', 3, 4.37, 'NPU Strix')];
const data = collect(historiesOf('llama-chain-hw', swaps));
// A time compares with the last night on its own part; a size with the night before.
const series = m => data.series.find(s => s.metric === m);
assert.deepEqual(latestMove(series('tok_s'), ['turbo']).since, Date.parse(swaps[0].date));
assert.equal(latestMove(series('tok_s'), ['turbo']).cls, 'worse');
assert.deepEqual(latestMove(series('elf_bytes'), ['turbo']).since, Date.parse(swaps[1].date));
assert.deepEqual(movedSeries(data, 'llama-chain-hw').regressed.map(x => x.series.metric), ['tok_s']);
// The trend: one machine at a time, the latest night's unless another is
// chosen, the others offered; each provenance change marked.
const rate = trendConfig(data, llama.trend.find(t => t.id === 'decode_by_context'));
const lines = rate.data.datasets.filter(d => !d.band);
assert.deepEqual(lines.map(d => [d.label, d.data]), [['64 context', [4.7, 4.37]]]);
assert.deepEqual([rate.data.parts, rate.data.part], [['Strix', 'Krackan 1'], 'Strix']);
assert.equal(rate.options.scales.x.title.text, 'Nightly run (weekly past 90 days), on Strix');
const other = trendConfig(data, llama.trend.find(t => t.id === 'decode_by_context'), { part: 'Krackan 1' });
assert.deepEqual(other.data.datasets.filter(d => !d.band).map(d => d.data), [[4]]);
assert.deepEqual(rate.plugins, [MARKERS]);
// One machine's nights: only what changed between them is marked.
assert.deepEqual(rate.options.plugins.markers.at.map(m => m.label), ['kernels k3']);
assert.deepEqual(rate.options.plugins.tooltip.callbacks.title([{ dataIndex: 1 }]), ['3 Oct, 06:30 UTC', 'changed: kernels k3']);
// Compile time moves with the part's host too: its stack is one machine's,
// the latest run's unless another is chosen, and offers the others.
const compile = stackConfig(data, { ...COMPILE_TABS[0], stack: COMPILE_TABS[0].split, missingIsZero: true });
assert.deepEqual([compile.data.parts, compile.data.part], [['Strix', 'Krackan 1'], 'Strix']);
assert.deepEqual(compile.data.datasets.filter(d => !d.part).map(d => [d.label, d.data]),
                 [['cold (nothing cached)', [60.5, 60.5]], ['warm (kernel objects kept)', [1.2, 1.2]]]);
assert.equal(compile.options.scales.x.title.text, 'Nightly run, on Strix');
assert.deepEqual(stackConfig(data, { ...COMPILE_TABS[0], stack: COMPILE_TABS[0].split, missingIsZero: true }, { part: 'Krackan 1' }).data.labels, ['2 Oct']);
// The part is named on the band. A runner that alternates parts is
// compared with its own part's last night: no change to warn of.
const card = componentBand(llama, { target: 'llama-chain-hw', runs: swaps.map(n => night(n.id, n.date, 'turbo', { device: n.device, provenance: n.provenance })) },
                           swaps[2], NOW, swaps[1]);
assert.ok(card.children[0].text.includes('on Strix'));
assert.ok(!card.text.includes('Changed since the run before'));
// A part's first night is compared with the night before it: the change is
// said in its details, not as a problem.
const first = componentBand(llama, { target: 'llama-chain-hw', runs: swaps.slice(0, 2).map(n => night(n.id, n.date, 'turbo', { device: n.device, provenance: n.provenance })) },
                            swaps[1], NOW, swaps[0]);
const about = one(first, n => n.className === 'band-more');
assert.ok(about.text.includes('Changed since the run before: device NPU Strix → NPU Krackan 1.'));
assert.ok(!all(first, n => /^note (warn|bad)$/.test(n.className || '')).length);
// And what moved is read against the same part's night.
const index3 = { runs: swaps.map(n => night(n.id, n.date, 'turbo', { device: n.device, provenance: n.provenance })) };
assert.equal(previousPublished(index3, { ...swaps[2], pmode: 'turbo' }).id, '1');
assert.equal(previousPublished(index3, { ...swaps[1], pmode: 'turbo' }).id, '1');
""")
    page(KERNELS + """
// The kernel page marks provenance changes and splits a time by part.
const k = (id, day, us, device, kernels) => ({ ...rec(id, Date.parse(`2026-10-0${day}T06:00:00Z`), 'turbo', {
  'softmax/1024/bfloat16/dispatch_us': ['us', us], 'softmax/1024/bfloat16/cycles': ['cycles', 1000] }), provenance: { device, kernels } });
db = fromRecords({ npu2: [k('1', 1, 80, 'NPU Strix', 'a'), k('2', 2, 77, 'NPU Krackan 1', 'a'), k('3', 3, 81, 'NPU Strix', 'b')] });
const us = kernelTrendConfig(db, 'npu2', 'softmax', 'dispatch_us');
assert.deepEqual(us.data.datasets.map(d => d.label), ['1024/bfloat16 · Strix', '1024/bfloat16 · Krackan']);
assert.deepEqual(us.plugins, [MARKERS]);
assert.deepEqual(us.options.plugins.markers.at.map(m => m.index), [1, 2]);
const cycles = kernelTrendConfig(db, 'npu2', 'softmax', 'cycles');
assert.deepEqual(cycles.data.datasets.map(d => [d.label, d.data]), [['1024/bfloat16', [1000, 1000, 1000]]]);
assert.deepEqual(cycles.options.plugins.tooltip.callbacks.title([{ dataIndex: 2 }]), ['3 Oct, 06:00 UTC', 'changed: device NPU Strix, kernels b']);
// The window compares a time on one part only.
assert.deepEqual(windowChanges(db, 'npu2', 'dispatch_us').map(x => [x.then, x.now]), [[80, 81]]);
""")


def test_an_absolute_floor_and_a_zero_baseline(page):
    page(COMPONENT_RUNS + MODEL_NIGHTS + """
// The schema: `pct` of the previous value and `abs`, a floor in the unit.
THRESHOLDS.compile_place_s = { pct: 15, abs: 0.5 };
THRESHOLDS.failed_mutations = { pct: 1, abs: 1 };
THRESHOLDS.max_ulp = { abs: 1 };
// A 15% move of a 0.13 s stage is 20 ms: under the floor, not a move.
assert.equal(changeClass(0.13, 0.15, 'compile_place_s'), '');
assert.equal(changeClass(3, 3.6, 'compile_place_s'), 'worse');
// Past the floor but under the percentage: not a move either.
assert.equal(changeClass(10, 10.6, 'compile_place_s'), '');
// From zero there is no percentage; the floor decides: 0 -> 2 failures is worse,
// 2 -> 0 better, and a metric with no floor is not judged from zero.
assert.equal(changeClass(0, 2, 'failed_mutations'), 'worse');
assert.equal(changeClass(2, 0, 'failed_mutations'), 'better');
assert.equal(changeClass(0, 2, 'cpu_ms_mean'), '');
// A metric with only a floor: any whole ulp counts, from any baseline.
assert.equal(changeClass(30, 31, 'max_ulp'), 'worse');
assert.equal(changeClass(0, 1, 'max_ulp'), 'worse');
assert.equal(changeClass(4, 4, 'max_ulp'), '');
assert.equal(thresholdText('compile_place_s'), 'counts past 15% and a change of 0.5');
assert.equal(thresholdText('max_ulp'), 'counts past a change of 1');
assert.equal(thresholdText('dispatch_us'), 'counts past 15% and 3× its spread');
// The NPU time has its own percentage, spread and floor.
assert.equal(thresholdText('npu_us'), 'counts past 10% and a change of 2 and 5× its spread');
assert.equal(changeClass(5, 6.5, 'npu_us'), '');
assert.equal(changeClass(50, 60, 'npu_us'), 'worse');
// The empty dispatch is the host's and the runtime's: shown, never judged.
assert.equal(changeClass(90, 300, 'dispatch_overhead_us'), '');
// Information only: never worse or better.
THRESHOLDS.ort_matches = { ...THRESHOLDS.ort_matches, neutral: true };
assert.equal(changeClass(8, 6, 'ort_matches'), '');
assert.deepEqual(['dispatch_us', 'dispatch_overhead_us', 'npu_us'].map(label), ['Dispatch time', 'Dispatch overhead', 'NPU time']);
// The router's failed mutations went 0 -> 2: flagged, said as "+2", not a percentage.
const before = routerRecord('6', '2026-10-01T03:17:00Z', 0);
before.rows.npu1.failed_mutations = cell(0, 'cases');
const after = routerRecord('7', '2026-10-02T03:17:00Z', 0);
const routerBand = componentBand(router, { target: 'router', runs: [night('6', before.date, null), night('7', after.date, null)] }, after, NOW, before);
assert.equal(one(routerBand, n => n.className === 'moves').text, 'Since 1 Oct:npu1 failed mutations +2');
assert.equal(one(routerBand.table, n => /^chg /.test(n.className || '') && n.text === '+2').className, 'chg worse strong');
assert.equal(formatChange(relative(0, 2), 2, 'cases'), '+2');
assert.equal(formatChange(relative(2, 0), -2, 'cases'), '−100.0%');
assert.equal(relative(0, 0), 0);
""")


def test_the_error_metrics_are_shown_and_a_zero_baseline_compares(page):
    page(KERNELS + """
// publish.py writes them per case and output type beside the timing rows.
const err = (id, day, ulp, pctOff) => rec(id, Date.parse(`2026-10-0${day}T06:00:00Z`), 'turbo', {
  'softmax/1024/bfloat16/cycles': ['cycles', 1000],
  'softmax/1024/bfloat16/max_ulp_bfloat16': ['ulp', ulp, 'vs the float64 correctly rounded reference'],
  'softmax/1024/bfloat16/not_correctly_rounded_bfloat16': ['%', pctOff] });
db = fromRecords({ npu2: [err('1', 1, 0, 0), err('2', 2, 0, 0), err('3', 3, 5, 49.2)] });
// The kernel page offers each, labelled, on an axis that says which way is better.
assert.deepEqual(ERROR_METRICS, ['max_ulp_bfloat16', 'not_correctly_rounded_bfloat16', 'max_ulp_float32', 'not_correctly_rounded_float32']);
assert.ok(ERROR_METRICS.every(m => ALL_METRICS.includes(m) && GATED.includes(m)));
assert.deepEqual(['max_ulp_bfloat16', 'not_correctly_rounded_float32'].map(label), ['Max error (ULP, bfloat16)', 'Not correctly rounded (%, float32)']);
assert.ok(METRICS.max_ulp_bfloat16.includes('correctly rounded value of the function the kernel computes'));
assert.ok(METRICS.max_ulp_bfloat16.includes("never a LUT build's model of itself"));
const ulp = kernelTrendConfig(db, 'npu2', 'softmax', 'max_ulp_bfloat16');
assert.equal(ulp.options.scales.y.title.text, 'Max error (ULP) · ↓ better');
assert.deepEqual(ulp.data.datasets[0].data, [0, 0, 5]);
// Hover says which reference the error is against.
assert.equal(ulp.options.plugins.tooltip.callbacks.label({ dataset: ulp.data.datasets[0], dataIndex: 2, raw: 5 }),
             '1024/bfloat16: 5 ulp (vs the float64 correctly rounded reference)');
assert.equal(ulp.options.scales.y.ticks.callback(5), '5 ulp');
assert.equal(kernelTrendConfig(db, 'npu2', 'softmax', 'not_correctly_rounded_bfloat16').options.scales.y.ticks.callback(49.2), '49.2%');
// Their floors (1 ulp, 0.1 points) make 0 -> 5 ulp a move from a zero baseline, said "+5 ulp".
const moved = movedSeries(db, 'npu2');
assert.deepEqual(moved.regressed.map(x => [x.series.metric, x.before, x.after]),
  [['max_ulp_bfloat16', 0, 5], ['not_correctly_rounded_bfloat16', 0, 49.2]]);
assert.equal(formatChange(moved.regressed[0].change, 5, 'ulp'), '+5 ulp');
// Gated, they read in the verdict as their own kind of move.
const night = nightOf('npu2', null, db, cat2);
assert.equal(verdictOf(night, Date.parse('2026-10-03T07:00:00Z')).headline, 'softmax less accurate vs 2 Oct.');
// Cycles lead the sentence, whatever order the moves came in.
const mixed = { ...night, moved: { regressed: [
  { series: { metric: 'kernel_object_bytes', kase: 'a/1' } }, { series: { metric: 'cycles', kase: 'b/1' } }], improved: [] } };
assert.equal(verdictOf(mixed, Date.parse('2026-10-03T07:00:00Z')).headline, 'a bigger, b slower vs 2 Oct.');
const item = attentionFor(night, Date.parse('2026-10-03T07:00:00Z')).find(i => i.level === 'bad');
assert.ok(item.body[0].startsWith('1 case got less accurate past its threshold vs 2 Oct:'));
// The Summary's table gains a column for an error metric only when one moved.
location.hash = '#view=night';
renderWindowChanges(db, ['npu2']);
const head = $('window-changes').children[0].children[0].children[0].children[0];
assert.deepEqual(head.children.map(c => c.text.split('1 Oct')[0].split('2 Oct')[0]),
  ['Case', 'NPU', 'Cyclesthen → now (change)', 'Object sizethen → now (change)', 'Max error (ULP, bfloat16)then → now (change)', 'Not correctly rounded (%, bfloat16)then → now (change)']);
""")


def test_a_link_to_a_metric_opens_the_tab_that_charts_it(page):
    page(COMPONENT_RUNS + LLAMA_NIGHTS + YOLO_NIGHTS + """
// A link that names a metric opens the tab that charts it.
const data = collect(historiesOf('llama-chain-hw', [llamaBefore, llamaAfter]));
assert.equal(trendBox(data, llama.trend, 'decode_tok_s').metric(), 'tok_s');
assert.equal(trendBox(data, llama.trend, 'compile_s').metric(), 'compile_s');
// The fixture's nights recorded no whole-compile peak: no memory tab, so the first tab.
assert.equal(trendBox(data, llama.trend, 'compile_peak_mib_warm').metric(), 'decode_by_context');
// A metric no tab charts opens the first tab.
assert.equal(trendBox(data, llama.trend, 'dma_floor_ratio').metric(), 'decode_by_context');
// Every metric a band judges has a label to show it by.
for (const band of BANDS) for (const m of band.metrics) assert.ok(m in LABELS, `${band.id} ${m}`);
// The band's trend is scrolled to.
const slot = document.createElement('div');
revealTrend(slot);
assert.ok(slot.scrolled);
""")


def test_a_check_that_did_not_finish_says_so_with_its_run(page):
    page(COMPONENT_RUNS + LLAMA_NIGHTS + """
// The placeholder record a check writes before its first device run: no rows.
const stopped = night('8', '2026-10-03T06:30:00Z', 'turbo', { published: false, reason: 'did not finish', failed: [LLAMA] });
const index = { target: 'llama-chain-hw', runs: [night('6', llamaBefore.date, 'turbo'), night('7', llamaAfter.date, 'turbo'), stopped] };
const card = componentBand(llama, index, llamaAfter, Date.parse('2026-10-03T09:00:00Z'), llamaBefore);
assert.equal(card.className, 'verdict bad');
assert.equal(one(card, n => n.className === 'headline').text, 'Did not finish: the 16-layer decode.');
// The run is a click away, and its last numbers are said to be older.
assert.ok(one(card, n => n.className === 'repro').text.includes('run 8'));
assert.ok(card.text.includes('Numbers from the run of 2 Oct, 06:30 UTC'));
assert.ok(nightTitle(stopped).endsWith('did not finish: published no numbers'));
""")


def test_a_change_of_measurement_is_marked_and_never_compared_across(page):
    page(KERNELS + """
// The 10-02 trace-decoder fix: measurement "1" before, "2" since.
const night = (id, day, cycles, measurement) => ({ ...rec(id, Date.parse(`2026-10-0${day}T06:00:00Z`), 'turbo', {
  'bn_conv2dk3/1792x8/int8/cycles': ['cycles', cycles] }), provenance: { measurement, peano: 'p' } });
db = fromRecords({ npu2: [night('1', 1, 293058, '1'), night('2', 2, 301, '2'), night('3', 3, 305, '2')] });
const conv = db.series[0];
// Last night compares with the night before, on the same measurement.
assert.equal(latestMove(conv, ['turbo']).since, Date.parse('2026-10-02T06:00:00Z'));
// The window starts at the change: 301 -> 305, not 293058 -> 305.
assert.deepEqual(windowChanges(db, 'npu2', 'cycles').map(x => [x.then, x.now]), [[301, 305]]);
// A change of measurement alone is no comparison at all, and is marked.
const two = fromRecords({ npu2: [night('1', 1, 293058, '1'), night('2', 2, 301, '2')] });
assert.equal(latestMove(two.series[0], ['turbo']), null);
assert.equal(movedSeries(two, 'npu2').improved.length, 0);
assert.deepEqual(kernelTrendConfig(two, 'npu2', 'bn_conv2dk3', 'cycles').options.plugins.markers.at, [{ index: 1, label: 'measurement 2' }]);
""")


def test_llama_decode_by_context_is_a_line_per_context(page):
    page(COMPONENT_RUNS + LLAMA_NIGHTS + """
// The context sweep: one compile, timed at each start position; a series per context.
const swept = (id, date, night) => {
  const r = llamaRecord(id, date, 4.37);
  for (const [ctx, ...ms] of [[1024, 52, 57.2], [8192, 70, 77], [131072, 390, 429]]) {
    r.rows[`${LLAMA}/ctx=${ctx}`] = { ms_per_token: { value: ms[night], unit: 'ms' }, tok_s: { value: 1000 / ms[night], unit: 'tok/s' } };
  }
  return r;
};
const data = collect(historiesOf('llama-chain-hw', [swept('6', llamaBefore.date, 0), swept('7', llamaAfter.date, 1)]));
const tab = llama.trend.find(t => t.id === 'decode_by_context');
assert.equal(tab.name, 'Decode');
const config = trendConfig(data, tab);
// The dispatch rate per context, higher is better, as mlir-air and FastFlowLM tabulate decode.
const lines = config.data.datasets.filter(d => !d.band);
assert.deepEqual(lines.map(d => [d.label, d.data.map(v => Math.round(v * 100) / 100)]),
  [['64 context', [4.37, 4.37]], ['1k context', [19.23, 17.48]], ['8k context', [14.29, 12.99]], ['128k context', [2.56, 2.33]]]);
assert.equal(config.options.scales.y.type, 'logarithmic');
assert.equal(config.options.scales.y.title.text, 'Decode, tokens / s · ↑ better');
// The time per token on hover.
assert.equal(config.options.plugins.tooltip.callbacks.label({ dataset: lines[3], dataIndex: 1, raw: 1000 / 429 }),
             '128k context: 2.33 tok/s, dispatch / token 429 ms');
// The plain decode tab charts the timed run alone, not the sweep summed in.
assert.deepEqual(trendConfig(data, llama.trend.find(t => t.id === 'tok_s')).data.datasets.filter(d => !d.band).map(d => d.data), [[4.37, 4.37]]);
assert.deepEqual(trendBox(data, llama.trend, '').children[0].children[0].children.map(b => b.textContent).slice(0, 3),
  ['Decode', 'End to end', 'Compile time']);
// The table and the summary stay on the timed run.
const card = componentBand(llama, llamaIndex, swept('7', llamaAfter.date, 1), NOW, swept('6', llamaBefore.date, 0));
assert.equal(tableOf(card.table).find(r => r[0] === 'Decode, dispatch only')[1], '4.37 tok/s');
""")


def test_llama_ttft_by_prompt_is_a_line_per_prompt_length(page):
    page(COMPONENT_RUNS + LLAMA_NIGHTS + """
// The prefill sweep: one compile, TTFT timed at each prompt length; a series per length.
const swept = (id, date, night) => {
  const r = llamaRecord(id, date, 4.37);
  r.rows[LLAMA].ttft_ms = { value: 124.6, unit: 'ms', range: 'median of 5' };
  for (const [len, ...ms] of [[512, 8060, 8100], [1024, 17110, 17200], [2048, 37430, 37600]]) {
    r.rows[`${LLAMA}/prompt=${len}`] = {
      ttft_ms: { value: ms[night], unit: 'ms', range: 'median of 5' },
      ttft_npu_dispatch_ms: { value: ms[night] - 150, unit: 'ms', range: 'median of 5' },
    };
  }
  return r;
};
const data = collect(historiesOf('llama-chain-hw', [swept('6', llamaBefore.date, 0), swept('7', llamaAfter.date, 1)]));
const tab = llama.trend.find(t => t.id === 'ttft_by_prompt');
assert.equal(tab.name, 'Time to first token');
const config = trendConfig(data, tab);
const lines = config.data.datasets.filter(d => !d.band);
// generate.py's own short prompt first, then the sweep's.
assert.deepEqual(lines.map(d => d.label), ['short prompt (6 tokens)', '512 token prompt', '1k token prompt', '2k token prompt']);
assert.deepEqual(lines.map(d => d.data), [[124.6, 124.6], [8060, 8100], [17110, 17200], [37430, 37600]]);
assert.equal(config.options.scales.y.type, 'logarithmic');
// The short prompt is its own line, not summed into the sweep's.
assert.equal(lines[0].data[0], 124.6);
""")


def test_a_link_to_a_performance_metric_lands_on_its_trend(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
const files = {
  'sa-placer-hw/runs.json': { ...hwIndex, metrics: ['us_per_image', 'e2e_us_per_image'] },
  'sa-placer-hw/latest.json': hwAfter,
  'sa-placer-hw/runs/6.json': hwBefore,
  ...Object.fromEntries(historiesOf('sa-placer-hw', [hwBefore, hwAfter]).map(h => [`sa-placer-hw/history/${h.metric}.json`, h])),
};
global.fetch = async url => {
  const file = files[url.replace('../component-checks/', '')];
  return { ok: Boolean(file), status: file ? 200 : 404, json: async () => file };
};
location.hash = '#view=performance&metric=e2e_us_per_image';
(async () => {
  await showSection('performance');
  await new Promise(resolve => setTimeout(resolve, 0));
  const [band] = $('performance-bands').children;
  // The end-to-end time has its own tab, every batch a line.
  assert.equal(band.trend.children[0].metric(), 'e2e_us_per_image');
  assert.ok(band.trend.scrolled);
})().catch(e => { console.error(e); process.exit(1); });
""")


def test_an_axis_shows_a_threshold_sized_move_of_its_metric(page):
    page("""
// A ratio near 1 with a 1% threshold: the axis hugs it, so a 1% move reads.
const hops = yRange([{ data: [1.02, 1.05] }], 'hop_ratio');
assert.ok(hops.min > 0.99 && hops.max < 1.07, JSON.stringify(hops));
assert.ok(hops.max - hops.min >= 1.035 * 0.01 * 2.5);
// Bank sharing, 0.31 to 0.33: not 0 to 1.33.
const sharing = yRange([{ data: [0.31, 0.33] }], 'bank_pair_sharing');
assert.ok(sharing.min > 0.28 && sharing.max < 0.36, JSON.stringify(sharing));
// A flat series still spans its threshold: dispatch_us at 100, 15%.
const flat = yRange([{ data: [100, 100] }], 'dispatch_us');
assert.deepEqual(flat, { min: 81.25, max: 118.75 });
// An abs-only metric pads by its floor; all zeros still draw an axis.
assert.deepEqual(yRange([{ data: [0, 0] }], 'max_ulp_float32'), { min: 0, max: 1.25 });
assert.deepEqual(yRange([{ data: [0, 0] }], 'cycles'), { min: 0, max: 1 });
""")


def test_an_unreadable_file_is_not_taken_for_an_unpublished_one(page):
    page(RUNS + COMPONENT_RUNS + """
const answers = {
  'missing.json': { ok: false, status: 404 },
  'busy.json': { ok: false, status: 503 },
  'broken.json': { ok: true, status: 200, json: async () => { throw new SyntaxError('Unexpected token'); } },
  'good.json': { ok: true, status: 200, json: async () => ({ schema: 1, runs: [] }) },
};
global.fetch = async url => {
  if (url === 'offline.json') throw new TypeError('Failed to fetch');
  return answers[url];
};
(async () => {
  assert.equal(await fetchJson('missing.json'), null);
  assert.deepEqual(await fetchJson('good.json'), { schema: 1, runs: [] });
  const [busy, broken, offline] = await Promise.all(['busy.json', 'broken.json', 'offline.json'].map(fetchJson));
  assert.deepEqual([busy.reason, broken.reason, offline.reason], ['HTTP 503', 'not valid JSON', 'the request failed']);
  // An unread file is never data, nor "a newer version".
  assert.equal(readable(busy), null);
  const unread = new Unread('../kernel-checks/npu2/runs.json', 'HTTP 503');
  const run = latestRun('npu2', unread, null, null);
  assert.equal(run.newer, false);
  // The kernels' verdict: could not read, reload; not "no results published".
  const verdict = verdictOf({ npu: 'npu2', run, counts: catalogueCounts(null), moved: null, latest: { cases: new Map() } }, 0);
  assert.deepEqual([verdict.level, verdict.headline], ['bad', 'Could not read kernel-checks/npu2/runs.json (HTTP 503): reload the page.']);
  assert.equal(verdictOf({ npu: 'npu2', run: latestRun('npu2', null, null, null), counts: catalogueCounts(null), moved: null, latest: { cases: new Map() } }, 0).headline,
               'No results published yet.');
  // A component band.
  const band = componentBand(hw, new Unread('../component-checks/sa-placer-hw/runs.json', 'the request failed'), null, NOW);
  assert.equal(band.className, 'verdict bad');
  assert.equal(one(band, n => n.className === 'headline').text, 'Could not read component-checks/sa-placer-hw/runs.json (the request failed): reload the page.');
  // CI health.
  const ci = { status: null, unread: new Unread('../ci-health/status.json', 'HTTP 502'), newer: false, results: [], critical: [], merge: null };
  assert.equal(ciStatus(ci, NOW).text, 'Could not read ci-health/status.json (HTTP 502): reload the page.');
  const unreadNote = one(runnerTable(ci, NOW), n => n.className === 'note bad');
  assert.equal(unreadNote.text, 'Could not read ci-health/status.json (HTTP 502): reload the page.');
  const [tile] = overviewTiles(null, [], NOW, ci);
  assert.deepEqual([tile.level, tile.children[1].text], ['bad', 'Needs attention']);
})().catch(e => { console.error(e); process.exit(1); });
""")


def test_without_chartjs_every_chart_is_drawn_as_svg(page):
    page(COMPONENT_RUNS + SWEEP_NIGHTS + """
global.Chart = undefined;
// The sweep's Seeds chart: dots, the nightly before as ticks, no lines between seeds.
const card = componentBand(sweep, sweepIndex, after, NOW, before);
const seeds = all(card, n => n.className === 'chart')[0];
const svg = all(seeds, n => n.className === 'svg-chart')[0];
assert.ok(svg.innerHTML.startsWith('<svg'));
assert.ok(!svg.innerHTML.includes('<polyline'));
// Three seeds placed, four the night before, one failed.
assert.equal((svg.innerHTML.match(/<circle/g) || []).length, 3 + 4 + 1);
// A band's trend chart.
const data = collect(historiesOf('sa-placer', [before, after]));
const holder = document.createElement('div');
const canvas = document.createElement('canvas');
holder.append(canvas);
document.createElement('div').append(holder);
const drawn = lineChart(canvas, trendConfig(data, { ...sweep.trend[0], variants: null }));
assert.ok(drawn.svg.some(p => p.startsWith('<polyline')));
assert.deepEqual(drawn.legend, ['mean of the seeds', 'worst seed']);
assert.ok(canvas.hidden);
""")


def _contrast(fg, bg):
    def luminance(color):
        rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    hi, lo = sorted([luminance(fg), luminance(bg)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_status_reads_without_color_and_text_meets_aa(page):
    html = PAGE.read_text(encoding="utf-8")
    css = html.split("<style>", 1)[1].split("</style>", 1)[0]
    themes = [
        dict(re.findall(r"--([\w-]+): (#[0-9a-f]{6})", block))
        for block in (
            css.split(":root {", 1)[1].split("}", 1)[0],
            css.split(':root[data-theme="dark"] {', 1)[1].split("}", 1)[0],
        )
    ]
    # Secondary and status text on every background it sits on: WCAG AA.
    for theme in themes:
        for fg, bgs in [
            ("faint", ["paper", "surface"]),
            ("warn", ["surface", "paper", "warn-soft"]),
            ("ok", ["surface", "ok-soft"]),
            ("bad", ["surface", "bad-soft"]),
        ]:
            for bg in bgs:
                ratio = _contrast(theme[fg], theme[bg])
                assert ratio >= 4.5, (fg, bg, theme[fg], theme[bg], ratio)
    # A failure carries a cross and a warning is hollow, in the strips and dots.
    assert ".night.bad::before { content: '\u2715'; }" in css
    assert re.search(
        r"\.night\.warn \{ background: transparent; box-shadow: inset", css
    )
    assert ".ci-row.bad .dot::before" in css and ".k-row .k-bad .dot::before" in css
    # Headings nest: the overview has one, a section's blocks are h3s.
    assert '<h2 class="sr-only">Overview</h2>' in html
    assert "<h2>Needs a look</h2>" not in html and "<h3>Needs a look</h3>" in html
    # The theme button's accessible name is its visible text.
    assert re.search(
        r'<button id="theme" type="button" title="[^"]+">Dark</button>', html
    )
    page(COMPONENT_RUNS + HW_NIGHTS + """
// A chart is an image named by what it charts and in which unit.
const data = collect(historiesOf('sa-placer-hw', [hwBefore, hwAfter]));
const box = trendBox(data, hw.trend, 'us_per_image');
const canvas = all(box, n => n.tag === 'canvas')[0];
assert.equal(canvas.role, 'img');
assert.equal(canvas.ariaLabel, 'Where the time goes, Time per image (µs, fastest of 20 launches), over recent nightlies');
// A run with no link still names itself.
const strip = nightStrip({ runs: [{ id: '1', date: '2026-10-01T06:30:00Z', pmode: 'turbo', published: true }], run: { id: '1' } });
assert.equal(strip.children[1].role, 'img');
assert.ok(strip.children[1].ariaLabel.startsWith('1 Oct, 06:30 UTC'));
""")


def test_every_metric_the_checks_publish_has_a_label_and_a_description(page):
    sys.path.insert(0, str(ROOT / "utils/component_checks"))
    try:
        stages = importlib.import_module("aiecc_stages")
    finally:
        sys.path.remove(str(ROOT / "utils/component_checks"))
    published = set(stages.COMPILE_METRICS)
    for source in (ROOT / "utils/component_checks").glob("*.py"):
        text = source.read_text(encoding="utf-8")
        # row(case, "metric", ...), ("metric", "unit") pairs, cells["metric"] and
        # *_COUNTS entries.
        published |= set(re.findall(r'row\(\s*[^,()]+,\s*"([a-z][a-z0-9_]*)"', text))
        published |= set(
            re.findall(
                r'\(\s*"([a-z][a-z0-9_]*)",\s*"(?:bytes|cores|ms|s|us|tok/s|fps)"\s*\)',
                text,
            )
        )
        published |= set(re.findall(r'cells\["([a-z][a-z0-9_]*)"\] = \(', text))
        for block in re.findall(r"^TIMED = \(([^)]*)\)", text, re.MULTILINE):
            published |= set(re.findall(r'"([a-z][a-z0-9_]*)"', block))
        for block in re.findall(r"_COUNTS = \{([^}]*)\}", text):
            published |= set(re.findall(r'"([a-z][a-z0-9_]*)":', block))
    publish = (ROOT / "utils/kernel_checks/publish.py").read_text(encoding="utf-8")
    published |= set(re.findall(r'"([a-z][a-z0-9_]*)": \{"value"', publish))
    for stem in re.findall(r'f"([a-z_]+)_\{dtype\}"', publish):
        published |= {f"{stem}_bfloat16", f"{stem}_float32"}
    # The timing checks' per-batch YOLO rows, named by the batch they ran.
    published |= {
        f"{metric}_b{batch}"
        for metric in (
            "us_per_image",
            "dispatch_us_per_image",
            "npu_us_per_image",
            "host_overhead_us_per_image",
            "fps",
        )
        for batch in (1, 8)
    }
    # n_cores is dropped (the author's call): the page shows the layout as words.
    published -= {"n_cores"}
    assert {
        "max_ulp_float32",
        "not_correctly_rounded_bfloat16",
        "compile_s",
        "unrouted",
        "tok_s",
        "placement_cost",
        "failed_mutations",
        "npu_us_per_image",
        "host_overhead_us_per_image",
    } <= published
    page(f"""
const published = {json.dumps(sorted(published))};
for (const m of [...published, ...ALL_METRICS]) {{
  assert.ok(m in LABELS, `${{m}} has no label`);
  assert.ok(m in METRICS || m in COMPONENT_METRICS, `${{m}} has no description`);
}}
// The removed helpers stay removed.
assert.equal(typeof latestChange, 'undefined');
""")


def test_repeated_commits_stay_separate_runs(page):
    page("""
assert.equal(db.order.length, 7);
assert.equal(new Set(db.order.map(p => p.id)).size, 7);
assert.deepEqual(db.order.filter(o => o.npu === 'npu1').map(o => o.run), ['t1', 'p1', 't2', 't25', 'p25', 't3']);
assert.deepEqual(s.byMode.get('turbo').map(p => p.row.value), [100, 110, 121]);
assert.equal(s.factory, 'softmax');
assert.equal(latestMove(s, ['turbo']).change, 0.1);
assert.equal(latestMove(s, ['turbo', 'performance']).change, 0.1);
""")


def test_a_published_history_file_is_read_as_is(page):
    page("""
// The shape publish.py writes: one file per metric, a value per run.
const history = {
  target: 'npu2', metric: 'cycles', unit: 'cycles',
  runs: [
    { id: '1', date: '2026-09-28T06:00:00+00:00', commit, pmode: 'default', provenance: { peano: 'a' } },
    { id: '2', date: '2026-09-29T06:00:00+00:00', commit, pmode: 'performance', provenance: { peano: 'b', host: 'h' } },
  ],
  series: {
    'add/1024x16/bfloat16': { values: [78, 80], ranges: ['median 87 max 131 n=16', null] },
    'relu/1024/bf16': { values: [null, 5] },
  },
};
db = collect([history]);
assert.deepEqual(db.order.map(o => [o.id, o.mode, o.provenance.peano]), [['npu2|1', 'default', 'a'], ['npu2|2', 'performance', 'b']]);
const add = db.series.find(s => s.kase === 'add/1024x16/bfloat16');
assert.deepEqual(add.byMode.get('default').map(p => p.row), [{ value: 78, unit: 'cycles', range: 'median 87 max 131 n=16' }]);
assert.deepEqual(add.byMode.get('performance').map(p => p.row), [{ value: 80, unit: 'cycles' }]);
assert.deepEqual([...db.series.find(s => s.kase === 'relu/1024/bf16').byMode.keys()], ['performance']);
// A second metric's file over the same runs adds series, not runs.
db = collect([history, { ...history, metric: 'dispatch_us', unit: 'us', series: { 'add/1024x16/bfloat16': { values: [1, 2] } } }]);
assert.equal(db.order.length, 2);
assert.equal(db.series.length, 3);
""")


def test_histories_are_fetched_once_and_the_previous_run_matches_the_mode(page):
    page("""
let fetched = [];
global.fetch = async url => {
  fetched.push(url);
  return { ok: url.includes('cycles'), json: async () => ({
    target: 'npu1', metric: 'cycles', unit: 'cycles',
    runs: [{ id: '1', date: '2026-09-28T06:00:00Z', commit, pmode: 'performance', provenance: {} }],
    series: { 'add/1/bf16': { values: [3] } },
  }) };
};
(async () => {
  let h = await loadHistories(['npu1'], ['cycles', 'dispatch_us']);
  assert.equal(h.series.length, 1);
  h = await loadHistories(['npu1'], ['cycles']);
  assert.deepEqual(fetched, ['../kernel-checks/npu1/history/cycles.json', '../kernel-checks/npu1/history/dispatch_us.json']);

  const index = { runs: [
    { id: 'a', pmode: 'default', published: true },
    { id: 'b', pmode: 'performance', published: true },
    { id: 'c', pmode: 'performance', published: false },
    { id: 'd', pmode: 'default', published: true },
    { id: 'e', pmode: 'performance', published: true },
  ]};
  assert.equal(previousPublished(index, { id: 'e', pmode: 'performance' }).id, 'b');
  assert.equal(previousPublished(index, { id: 'd', pmode: 'default' }).id, 'a');
  assert.equal(previousPublished(index, { id: 'b', pmode: 'performance' }), null);
  assert.equal(previousPublished(null, { id: 'b', pmode: 'performance' }), null);
})().catch(e => { console.error(e); process.exit(1); });
""")


def test_npu_us_deviation_is_a_band_outside_the_legend_and_tooltip(page):
    page("""
const us = (v, range) => ({ 'softmax/1024x16/bfloat16/dispatch_us': ['us', v, range] });
db = fromRecords({ npu1: [
  rec('1', 100000, 'turbo', us(50, 'min 40.0 max 60.0 n=50')),
  rec('2', 200000, 'turbo', us(40, '± 2.5; min 38.0 max 45.0 n=50')),
]});
const pts = db.series.find(s => s.npu === 'npu1').byMode.get('turbo');
const [low, high] = band(pts.map(p => p.row.value), pts.map(p => p.row), '#1f6fb8');
assert.deepEqual(low.data, [null, 37.5]);
assert.deepEqual(high.data, [null, 42.5]);
assert.equal(high.fill, '-1');
// The band takes its line's color, faded.
assert.equal(low.backgroundColor, 'rgba(31, 111, 184, 0.2)');
assert.ok(low.band && high.band);
""")


def test_cycles_spread_is_a_band_from_the_min_to_the_median(page):
    page("""
const cy = (v, range) => ({ 'softmax/1024x16/bfloat16/cycles': ['cycles', v, range] });
db = fromRecords({ npu1: [
  rec('1', 100000, 'turbo', cy(100)),
  rec('2', 200000, 'turbo', cy(110, 'median 115 max 400 n=16; truncated')),
  rec('3', 300000, 'turbo', cy(105, 'median 105 max 105 n=16')),
]});
const pts = db.series.find(s => s.npu === 'npu1').byMode.get('turbo');
const [low, high] = band(pts.map(p => p.row.value), pts.map(p => p.row), '#1f6fb8');
assert.deepEqual(low.data, [null, 110, 105]);
assert.deepEqual(high.data, [null, 115, 105]);
assert.ok(low.band && high.band);
""")


def test_provenance_changes_are_marked_and_shown_in_the_footer(page):
    page("""
const a = { commit: '111', peano: '22.0.0+aaaa', kernels: 'k1', host: 'bench-1' };
const b = { commit: '222', peano: '22.0.0+bbbb', kernels: 'k2', host: 'bench-1' };
const c = { commit: '333', peano: '22.0.0+bbbb', kernels: 'k3', host: 'bench-2', runtime: 'XRTHostRuntime', xrt: '2.20.0' };
db = fromRecords({ npu1: [
  rec('1', 100000, 'performance', sm(5), a), rec('2', 200000, 'performance', sm(null)),
  rec('3', 300000, 'performance', sm(6), b), rec('4', 400000, 'performance', sm(7), c),
]});
// The run without provenance is skipped; the commit is not marked, the kernels' digest is.
assert.deepEqual(markersFor(db.order), [
  { index: 2, label: 'peano 22.0.0+bbbb, kernels k2' },
  { index: 3, label: 'host bench-2, runtime XRTHostRuntime, xrt 2.20.0, kernels k3' },
]);
// What changed is in the tooltip's title, not drawn over the chart.
assert.deepEqual(changedTitle('3 Jan', markersFor(db.order), 2), ['3 Jan', 'changed: peano 22.0.0+bbbb, kernels k2']);
""")


def test_a_kernel_back_to_its_usual_is_not_a_regression(page):
    page("""
// conv2dk1 on npu2: 343 cycles for a week, 336 for two nights, 343 again;
// mm: 414 for a week, then 427. Night over night both got slower; only mm
// left the week's range.
const week = [343, 343, 343, 343, 343, 336, 336, 343];
const mmWeek = [414, 414, 414, 414, 414, 414, 414, 427];
const records = week.map((v, i) => rec(String(i + 1), (i + 1) * 86400000, 'turbo',
  { 'conv2dk1/2048x8/uint8/cycles': ['cycles', v], 'mm/32x64/bfloat16/cycles': ['cycles', mmWeek[i]] }));
const history = fromRecords({ npu2: records });
const lastTwo = fromRecords({ npu2: records.slice(-2) });
const without = nightOf('npu2', null, lastTwo, null);
assert.deepEqual(without.moved.regressed.map(x => x.series.kase).sort(), ['conv2dk1/2048x8/uint8', 'mm/32x64/bfloat16']);
const withWeek = nightOf('npu2', null, lastTwo, null, history);
assert.deepEqual(withWeek.moved.regressed.map(x => x.series.kase), ['mm/32x64/bfloat16']);
assert.deepEqual([...withWeek.flagged.keys()], ['mm/32x64/bfloat16']);
""")


def test_latest_cases_follow_the_latest_nightly(page):
    page("""
db = fromRecords({ npu1: [
  rec('1', 1, 'turbo', { 'softmax/1024/bfloat16/cycles': ['cycles', 100], 'softmax/64/bfloat16/cycles': ['cycles', 50] }),
  rec('3', 3, 'turbo', {
    'softmax/1024/bfloat16/cycles': ['cycles', 110, 'median 112 max 130 n=16'],
    'softmax/1024/bfloat16/kernel_object_bytes': ['bytes', 4096],
  }),
  rec('2', 2, 'performance', { 'softmax/1024/bfloat16/cycles': ['cycles', 1] }),
]});
const { last, cases } = latestCases(db, 'npu1');
assert.equal(last.mode, 'turbo');
const big = cases.get('softmax/1024/bfloat16');
assert.ok(big.current);
assert.deepEqual(big.metrics.get('cycles'), { value: 110, unit: 'cycles', change: 0.1, delta: 10, cls: 'worse', since: 1, range: 'median 112 max 130 n=16' });
assert.equal(big.metrics.get('kernel_object_bytes').change, null);
assert.ok(!cases.get('softmax/64/bfloat16').current);
assert.equal(latestCases(db, 'npu2').cases.size, 0);
""")


def test_object_size_change_is_in_kib(page):
    page("""
assert.equal(formatBytesDelta(307.2), '+0.3 KiB');
assert.equal(formatBytesDelta(-2048), '\u22122.0 KiB');
assert.equal(formatBytesDelta(48), '+48 B');
// On a card, a size moves in KiB, the percentage on hover.
const vals = glanceValues({ value: 4096, unit: 'bytes', change: 0.08, delta: 307.2, cls: 'worse' }, 'kernel_object_bytes', true);
const chip = vals.children.find(x => x.className && x.className.startsWith('chg'));
assert.equal(chip.text, '+0.3 KiB');
assert.ok(chip.className.includes('worse'));
""")


def test_failures_say_where_the_logs_are_and_how_to_rerun(page):
    page("""
const e2e = 'test.python.npu.test_kernels_e2e::test_kernel_extensive[sigmoid/1024x16/bfloat16/random/s2]';
assert.equal(caseOfTest(e2e), 'sigmoid/1024x16/bfloat16');
assert.equal(reproCommand(e2e),
  "python -m pytest 'test/python/npu/test_kernels_e2e.py::test_kernel_extensive[sigmoid/1024x16/bfloat16/random/s2]' --seeds 3 -v");
assert.equal(reproCommand('test_kernels_perf.py::test_kernel_perf[add/1024x16/bfloat16]'),
  "python -m pytest 'test/python/npu/test_kernels_perf.py::test_kernel_perf[add/1024x16/bfloat16]' -v");
assert.equal(reproByName('add/1024x16/bfloat16', 'test_kernel_perf'),
  "python -m pytest test/python/npu/test_kernels_perf.py -m perf -v -k 'add/1024x16/bfloat16'");
assert.ok(reproByName('conv/8x8/int8/width=28', 'test_kernel_extensive').endsWith("-k 'conv'  # every conv case"));
const run = { id: '7', url: 'https://example.invalid/runs/7', device: 'Krackan', commit: '91ee8c89c8b841f2', failed: [e2e] };
const counts = catalogueCounts(null);
const items = attentionFor({ npu: 'npu2', run, counts, moved: null }, Date.parse('2026-09-30T12:00:00Z'));
const item = items.find(i => i.body.some(b => typeof b === 'string' && b.startsWith('1 test failed')));
assert.ok(item, 'a failed test the catalogue does not name still gets a line');
const block = item.body[item.body.length - 1];
assert.equal(block.tag, 'details');
const pre = block.children.find(c => c.tag === 'pre');
assert.ok(pre.text.startsWith('git checkout 91ee8c89c8b8\\npython -m pytest'));
assert.ok(block.text.includes('kernel-checks-npu2-7'));
""")


def test_a_run_started_by_hand_is_never_the_baseline(page):
    page("""
const index = { runs: [
  { id: 'n1', published: true, pmode: 'turbo', event: 'schedule' },
  { id: 'm', published: true, pmode: 'turbo', event: 'workflow_dispatch' },
  { id: 'n2', published: true, pmode: 'turbo', event: 'schedule' },
] };
assert.equal(previousPublished(index, { id: 'n2', pmode: 'turbo' }).id, 'n1');
assert.equal(previousPublished(index, { id: 'n1', pmode: 'turbo' }), null);
// Records older than the field count as nightlies.
assert.equal(previousPublished({ runs: [{ id: 'a', published: true, pmode: 'turbo' }, { id: 'b' }] }, { id: 'b', pmode: 'turbo' }).id, 'a');
""")


def test_cycle_spread_and_truncation_are_read_from_the_range(page):
    page(KERNELS + """
assert.deepEqual(cyclesSpread('median 2939 max 2990 n=84; truncated'),
                 { median: 2939, max: 2990, n: 84, truncated: true });
assert.deepEqual(cyclesSpread('median 264 max 264 n=16'),
                 { median: 264, max: 264, n: 16, truncated: false });
assert.deepEqual(cyclesSpread('median 5580 max 5580 n=3; init[2] min 16; truncated'),
                 { median: 5580, max: 5580, n: 3, truncated: true });
assert.equal(cyclesSpread('± 1.6; min 172.8 max 202.2 n=50'), null);
assert.equal(cyclesSpread(undefined), null);
// The case table: the spread on hover, a full trace buffer said beside the value.
db = fromRecords({ npu1: [rec('9', 9, 'turbo', {
  'softmax/1024/bfloat16/cycles': ['cycles', 2875, 'median 2939 max 2990 n=84; truncated'],
  'softmax/64/bfloat16/cycles': ['cycles', 78, 'median 87 max 131 n=16'],
})] });
const body = kernelCaseTable('softmax', [nightOf('npu1', null, db, cat1)], db).children[0].children[1];
const row = kase => body.children.find(r => r.children[0].children[0].title === kase);
const cell = row('softmax/1024/bfloat16').children[1];
assert.equal(cell.text, '2,875 trace');
assert.equal(cell.children.find(c => c.className).className, 'note warn');
assert.equal(cell.title, 'fastest 2875, median 2939, slowest 2990 over 84 calls');
const plain = row('softmax/64/bfloat16').children[1];
assert.equal(plain.text, '78');
assert.equal(plain.title, 'fastest 78, median 87, slowest 131 over 16 calls');
""")


MOVED = """
const r1 = rec('1', 1000, 'turbo', {
  'relu/1024/bf16/cycles': ['cycles', 1000], 'relu/1024/bf16/cycles_per_kop': ['cycles/1k-ops', 10],
  'relu/1024/bf16/dispatch_us': ['us', 100], 'relu/1024/bf16/kernel_object_bytes': ['bytes', 4096],
  'gelu/1024/bf16/cycles': ['cycles', 2000], 'gone/1/i8/cycles': ['cycles', 10],
  'flat/1/i8/cycles': ['cycles', 500],
});
const r2 = rec('2', 2000, 'turbo', {
  'relu/1024/bf16/cycles': ['cycles', 1030], 'relu/1024/bf16/cycles_per_kop': ['cycles/1k-ops', 10.3],
  'relu/1024/bf16/dispatch_us': ['us', 150], 'relu/1024/bf16/kernel_object_bytes': ['bytes', 4096],
  'gelu/1024/bf16/cycles': ['cycles', 1500], 'flat/1/i8/cycles': ['cycles', 502],
  'fresh/1/i8/cycles': ['cycles', 5],
});
db = fromRecords({ npu1: [r1, r2] });
"""


def test_moved_series_use_the_thresholds_and_lead_with_gated_metrics(page):
    page(MOVED + """
const moved = movedSeries(db, 'npu1');
// cycles_per_kop is derived, an unchanged object is not a move, gone/ has no
// latest point, fresh/ no previous one, flat/ moved less than 1%.
assert.deepEqual(moved.regressed.map(x => [x.series.kase, x.series.metric, x.before, x.after]), [
  ['relu/1024/bf16', 'cycles', 1000, 1030],
  ['relu/1024/bf16', 'dispatch_us', 100, 150],
]);
assert.deepEqual(moved.improved.map(x => [x.series.kase, x.series.metric]), [['gelu/1024/bf16', 'cycles']]);
assert.equal(movedSeries(db, 'npu2').regressed.length, 0);
""")


RUNS = """
const catalogue = { npu: 'npu1', arch: 'aie2', commit: 'abcdef123456', date: '2020-01-01T00:00:03Z', kernels: [] };
const summary = (id, date, pmode, provenance, extra) => ({
  id, url: `https://example.com/runs/${id}`, date, commit: { ...commit }, pmode, device: 'NPU Strix',
  provenance, sane: true, published: true, failed: [], truncated: [], ...(extra || {}),
});
const index = { target: 'npu1', runs: [
  summary('1', '1970-01-01T00:00:01Z', 'turbo', { peano: '22.0.0+aaaa', host: 'bench-1' }),
  summary('2', '1970-01-01T00:00:02Z', 'default', { peano: '22.0.0+bbbb', host: 'bench-2', xrt: '2.20.0' },
          { failed: ['test_kernels_perf.py::test_kernel_perf[mm/64/i8]'], truncated: ['swiglu/1024x256/bfloat16'] }),
]};
"""


def test_latest_run_resolves_recorded_vs_fallback_state(page):
    page(MOVED + RUNS + """
const run = latestRun('npu1', index, db, catalogue);
assert.equal(run.recorded, true);
assert.equal(run.url, 'https://example.com/runs/2');
assert.equal(run.date, 2000);
assert.equal(run.commit, 'abcdef123456');
assert.equal(run.commitUrl, commit.url);
assert.equal(run.message, 'same revision');
assert.equal(run.pmode, 'default');
assert.equal(run.device, 'NPU Strix');
assert.equal(run.previous.id, '1');
// A migrated run has no Actions link.
const migrated = latestRun('npu1', { runs: [{ ...index.runs[0], url: '' }] }, db, catalogue);
assert.equal(migrated.url, null);
// Without runs.json, the latest run's record stands in.
const fallback = latestRun('npu1', null, db, catalogue);
assert.equal(fallback.recorded, false);
assert.equal(fallback.url, null);
assert.equal(fallback.date, 2000);
assert.equal(fallback.commit, 'abcdef123456');
assert.equal(fallback.pmode, 'turbo');
// Only the catalogue (every run dropped): nothing published, nothing failed.
const empty = latestRun('npu1', { schema: 1, runs: [] }, fromRecords({}), catalogue);
assert.equal(empty.sane, null);
// Nothing at all.
const none = latestRun('npu2', null, db, null);
assert.equal(none.date, null);
""")


def test_warnings_for_flag_mode_failure_truncation_and_provenance_changes(page):
    page(MOVED + RUNS + """
const run = latestRun('npu1', index, db, catalogue);
const now = 3600 * 1000;
assert.deepEqual(warningsFor('npu1', run, now).map(w => [w.level, w.text.split(':')[0]]), [
  ['warn', 'measured in power mode "default", not turbo'],
  ['warn', '1 test failed in this run'],
  ['warn', '1 case with a truncated trace'],
  ['info', 'peano changed since the previous run'],
  ['info', 'host changed since the previous run'],
]);
// A field the run before did not record is no change: xrt is new here.
assert.ok(!warningsFor('npu1', run, now).some(w => w.change === 'xrt'));
""")


def test_warnings_for_handles_refused_stale_and_empty_records(page):
    page(MOVED + RUNS + """
const now = 3600 * 1000;
// Stale, unsane runs say so first.
const bad = latestRun('npu1', { runs: [{ ...index.runs[0], sane: false, published: false }] }, db, catalogue);
assert.deepEqual(warningsFor('npu1', bad, now + 48 * 3600 * 1000).map(w => w.level), ['bad', 'bad']);
// A refused run says why, and its empty provenance is no change: the next
// run compares with the last published one.
const refusal = "power mode is performance, required 'turbo'";
const refusedRuns = [...index.runs, summary('3', '1970-01-01T00:00:03Z', null, {}, { sane: null, published: false, refused: refusal })];
assert.deepEqual(warningsFor('npu1', latestRun('npu1', { runs: refusedRuns }, db, catalogue), now).map(w => w.text),
                 [`the run refused to measure: ${refusal}`]);
const after = latestRun('npu1', { runs: [...refusedRuns, summary('4', '1970-01-01T00:00:04Z', 'turbo', index.runs[1].provenance)] }, db, catalogue);
assert.equal(after.previous.id, '2');
assert.deepEqual(warningsFor('npu1', after, now), []);
// Fallback staleness: silent within the window, loud past it.
const fallback = latestRun('npu1', null, db, catalogue);
assert.deepEqual(warningsFor('npu1', fallback, now), []);
assert.deepEqual(warningsFor('npu1', fallback, now + 48 * 3600 * 1000),
                 [{ level: 'bad', text: 'no run since 1 Jan, 00:00 UTC', stale: true }]);
// Only the catalogue (every run dropped): nothing published, nothing failed.
const empty = latestRun('npu1', { schema: 1, runs: [] }, fromRecords({}), catalogue);
assert.deepEqual(warningsFor('npu1', empty, Date.parse(catalogue.date)).map(w => w.text),
                 ['no numbers have been published yet']);
// Nothing at all.
const none = latestRun('npu2', null, db, null);
assert.deepEqual(warningsFor('npu2', none, now).map(w => w.level), ['bad']);
""")


def test_dashboard_verdicts_attention_map_and_moves(page):
    page(MOVED + RUNS + """
catalogue.kernels = [
  { factory: 'relu', family: 'activation', summary: 'ReLU', sources: ['activation/relu.cc'], builds: ['relu'],
    passed: 3, failed: [], timed: 1, timing_failed: ['relu/2048/bf16'], untimed: ['relu/64/bf16'] },
  { factory: 'mm', family: 'linalg', summary: 'mm', sources: [], builds: ['mm'], passed: 0, failed: ['mm/64/i8'], timed: 0 },
  { factory: 'exp2f_vec', family: 'activation', summary: 'npu2 only', sources: [], builds: [], passed: 0, failed: [], timed: 0 },
];
const clean = { target: 'npu1', runs: [summary('2', '1970-01-01T00:00:02Z', 'turbo',
  { peano: '22.0.0+bbbb', host: 'bench-2', xrt: '2.20.0', xdna: '2.20.0_1', kernels: 'k9' })] };
const shownNights = renderDashboard(['npu1', 'npu2'], db, new Map([['npu1', catalogue]]), new Map([['npu1', clean]]), 3600 * 1000);
assert.deepEqual(shownNights.map(n => n.npu), ['npu1', 'npu2']);

// One verdict per NPU, side by side; the headline leads with what is wrong.
const [v1, v2] = $('verdicts').children;
assert.equal(v1.className, 'verdict bad');
const headline = v1.children.find(c => c.className === 'headline');
// The comparison names its baseline.
assert.equal(headline.text, 'relu cycles +3.0% vs 1 Jan, 1 failing correctness, 1 failed timing. 1 faster.');
// The counts are under Run details, shut.
const runMore = v1.children.find(c => c.className === 'band-more');
assert.ok(!runMore.open);
assert.equal(runMore.children[0].text, 'Run details');
const figures = runMore.children.find(c => c.className === 'figures');
assert.deepEqual(figures.children.map(li => [li.className || '', li.text]), [
  ['', '1timed'], ['', '3pass correctness'], ['bad', '1failing'], ['warn', '1timing failed'],
  ['', '1not timed, by design'], ['', '2 of 2kernels run on the NPU'],
]);
const meta = runMore.children.find(c => c.className === 'meta');
assert.deepEqual(meta.children.map(c => c.text), [
  '1 Jan, 00:00 UTC', 'abcdef1 same revision', 'Strix NPU', 'Peano 22.0.0+bbbb', 'XRT 2.20.0, driver 2.20.0_1',
  'host bench-2', 'turbo mode',
]);
assert.equal(meta.children[0].children[0].href, 'https://example.com/runs/2');
assert.equal(meta.children[1].children[0].href, commit.url);
assert.equal(v2.className, 'verdict none');
assert.equal(v2.children.find(c => c.className === 'headline').text, 'No results published yet.');

// Attention: failures first, then warnings; each names its NPU once.
assert.equal($('attention').hidden, false);
const items = $('attention-list').children;
const all = (n, f) => typeof n === 'string' ? [] : [...(f(n) ? [n] : []), ...n.children.flatMap(c => all(c, f))];
assert.deepEqual(items.map(li => [li.className, li.children[0].text]),
  [['bad', 'npu1'], ['bad', 'npu1'], ['bad', 'npu2'], ['warn', 'npu1']]);
assert.ok(items[0].text.includes('failed correctness'));
assert.equal(items[0].children[1].children[1].children[0].children[0].title, 'mm/64/i8');
assert.equal(items[2].children[1].text, 'No results published yet.');

// A regression is named in Needs a look with how far it moved; its kernel a click away.
const worse = items[1];
assert.ok(worse.text.startsWith('npu11 case got slower past its threshold vs 1 Jan:'));
assert.ok(worse.text.includes('relu/1024/bf16 cycles +3.0%'));
assert.equal(all(worse, n => n.tag === 'a')[0].href, '#view=kernel&kernel=relu');

// All kernels: worst problem first, a correctness failure before a
// regression, both before a kernel not built anywhere (row shape and per-NPU
// text are test_all_kernels_is_a_line_each_problems_first_and_filters' own).
assert.deepEqual(renderKernelList(shownNights, db).map(r => r.factory), ['mm', 'relu', 'exp2f_vec']);
""")


@pytest.mark.parametrize(
    "latest",
    [
        "{ ...r2, id: '3', published: false, sane: false }",
        "{ ...r2, published: false, sane: false }",
        "{ ...r2, id: '3', published: true, sane: true }",
    ],
)
def test_dashboard_does_not_attribute_old_changes_to_the_latest_run(page, latest):
    page(MOVED + f"""
renderDashboard(['npu1'], db, new Map(), new Map([['npu1', {{ runs: [{latest}] }}]]), 3000);
// The latest run is not the one that moved: no regression named, none in the verdict.
assert.ok(!$('attention-list').text.includes('got slower'));
assert.ok(!$('verdicts').text.includes('slower'));
""")


def test_dashboard_can_compare_without_a_run_index(page):
    page(MOVED + """
renderDashboard(['npu1'], db, new Map(), new Map(), 3000);
// relu got worse: named in Needs a look.
assert.ok($('attention-list').text.includes('relu/1024/bf16 cycles +3.0%'));
""")


def test_a_refused_night_says_why_once_and_greys_the_passing_kernels(page):
    page(RUNS + """
const refusal = "power mode is performance, required 'turbo'";
catalogue.timing_refused = refusal;
catalogue.kernels = [
  { factory: 'relu', family: 'activation', summary: 'ReLU', sources: [], builds: ['relu'], passed: 2, failed: [], timed: 0 },
  { factory: 'mm', family: 'linalg', summary: 'mm', sources: [], builds: ['mm'], passed: 0, failed: ['mm/64/i8'], timed: 0 },
];
const refused = { target: 'npu1', runs: [summary('3', '1970-01-01T00:00:03Z', 'performance', {}, {
  sane: null, published: false, refused: refusal,
  failed: ['test_kernels_perf.py::test_kernel_perf[relu/64/bf16]', 'test_kernels_perf.py::test_measurement_is_sane',
           'test_kernels_e2e.py::test_kernel_extensive[mm/64/i8/rand/s0]'],
})] };
const [night] = renderDashboard(['npu1'], fromRecords({}), new Map([['npu1', catalogue]]), new Map([['npu1', refused]]), 3600 * 1000);
assert.deepEqual(kernelState(night, 'relu'), { level: 'untimed', text: '', title: `2 cases pass; not timed: ${refusal}` });
assert.equal(kernelState(night, 'mm').level, 'bad');
assert.deepEqual(caseVerdict(night, 'relu/64/bf16'), { cls: '', text: 'not timed', title: `the night timed nothing: ${refusal}` });
// The refusal once, as the NPU's verdict; Needs a look has the failing
// case, not a line per timing test it failed.
assert.ok($('verdicts').text.includes(`The run refused to measure: ${refusal}.`));
const items = $('attention-list').children.map(li => li.children[1].text);
assert.equal(items.length, 2);
assert.ok(items[0].startsWith('1 case failed correctness'));
assert.ok(items[1].startsWith('Measured in power mode "performance"'));
assert.equal(stateWord(kernelState(night, 'relu')), 'not timed');
""")


def test_a_case_that_passed_on_a_retry_is_flagged_but_not_failing(page):
    page(RUNS + """
catalogue.kernels = [
  { factory: 'relu', family: 'activation', summary: 'ReLU', sources: [], builds: ['relu'], passed: 2, failed: [], timed: 2,
    flaky: ['relu/64/bf16'] },
];
const runs = { target: 'npu1', runs: [summary('3', '1970-01-01T00:00:03Z', 'turbo', {}, {
  cases: { timed: 2, failed: 0, timing_failed: 0, flaky: 1 } })] };
const [night] = renderDashboard(['npu1'], fromRecords({}), new Map([['npu1', catalogue]]), new Map([['npu1', runs]]), 3600 * 1000);
assert.deepEqual(kernelState(night, 'relu'), { level: 'warn', text: 'flaky', title: 'passed only on a retry:\\nrelu/64/bf16' });
const [shown] = $('verdicts').children;
assert.equal(shown.className, 'verdict warn');
assert.equal(shown.children.find(c => c.className === 'headline').text, '1 passed on a retry.');
assert.ok(shown.children.find(c => c.className === 'band-more').children.find(c => c.className === 'figures').children.some(li => li.text === '1passed on a retry'));
const items = $('attention-list').children;
assert.deepEqual(items.map(li => li.className), ['warn']);
assert.ok(items[0].text.includes('passed when the runner retried it'));
assert.equal(nightLevel(runs.runs[0]), 'warn');
assert.ok(nightTitle(runs.runs[0]).endsWith('2 timed, 0 failing, 0 timing failed, 1 passed on a retry'));
const [row] = renderKernelList([night], fromRecords({}));
assert.deepEqual([row.className, row.children[1].text], ['k-row warn', 'npu1passed on a retry']);
""")


def test_changes_over_the_window_are_then_and_now(page):
    page(KERNELS + """
// softmax/1024 on npu1: 1,000 on its first nightly, 1,100 on its latest.
const moved = windowChanges(db, 'npu1', 'cycles');
assert.deepEqual(moved.map(x => [x.kase, x.then, x.now, x.cls]), [['softmax/1024/bfloat16', 1000, 1100, 'worse']]);
assert.equal(moved[0].change.toFixed(2), '0.10');
// A case measured on one night only has nothing to compare.
assert.ok(!moved.some(x => x.kase.startsWith('softmax/64') || x.kase.startsWith('softmax_mask')));
// The Summary's table over the window: case, NPU, measure, then and now
// with their dates, the change.
const rows = renderWindowChanges(db, ['npu1', 'npu2']);
// One row per kernel and NPU; a column per measure, "then → now (change)",
// tagged a step on the night it moved.
assert.deepEqual(rows.map(r => r.children.map(c => c.text)),
  [['softmax/1024/bfloat16', 'npu1', '1,000 → 1,100 (+10.0%)Step 1 Jan', '—']]);
assert.equal(rows[0].children[0].children[0].children[0].href, '#view=kernel&kernel=softmax');
assert.ok($('window-about').text.startsWith("Each case's cycles and object size over its last two weeks of nightlies"));
// Nothing moved: said so.
const flat = fromRecords({ npu1: [rec('1', 1, 'turbo', { 'a/1/i8/cycles': ['cycles', 100] }), rec('2', 2, 'turbo', { 'a/1/i8/cycles': ['cycles', 100] })] });
assert.deepEqual(renderWindowChanges(flat, ['npu1']), []);
assert.equal($('window-changes').text, 'Nothing stepped or crept past its threshold in two weeks.');
// Runs started by hand are not nightlies: left out.
const manual = fromRecords({ npu1: [rec('1', 1, 'turbo', { 'a/1/i8/cycles': ['cycles', 10] }),
  { ...rec('2', 2, 'turbo', { 'a/1/i8/cycles': ['cycles', 20] }), event: 'workflow_dispatch' }] });
assert.equal(nightValues(manual, 'npu1', 'cycles').length, 1);
""")


KERNELS = """
db = fromRecords({ npu1: [
  rec('1', 1, 'turbo', { 'softmax/1024/bfloat16/cycles': ['cycles', 1000], 'softmax/64/bfloat16/cycles': ['cycles', 50] }),
  rec('2', 2, 'turbo', {
    'softmax/1024/bfloat16/cycles': ['cycles', 1100, 'median 1150 max 1200 n=16'],
    'softmax/1024/bfloat16/kernel_object_bytes': ['bytes', 4096],
    'softmax_mask/8/bfloat16/cycles': ['cycles', 7],
  }),
], npu2: [rec('2', 2, 'turbo', { 'softmax/1024/bfloat16/cycles': ['cycles', 2200] })] });
const kernel = (factory, extra) => ({
  factory, family: 'activation', summary: factory, sources: [`activation/${factory}.cc`],
  builds: [factory], passed: 1, failed: [], timed: 1, ...extra,
});
const cat1 = { npu: 'npu1', arch: 'aie2', commit: 'abcdef1', date: '2020-01-01T00:00:00Z', kernels: [
  kernel('softmax', { passed: 3, timed: 2, failed: ['softmax/2048/bfloat16'], timing_failed: ['softmax/16/bfloat16'],
                      untimed: ['softmax/32/bfloat16'] }),
  kernel('relu', { builds: [], passed: 0, timed: 0 }),
  kernel('cascade_mm', { passed: 0, timed: 0, reason: 'no case' }),
]};
const cat2 = { ...cat1, npu: 'npu2', arch: 'aie2p', kernels: [kernel('softmax', {})] };
nights = [nightOf('npu1', null, db, cat1), nightOf('npu2', null, db, cat2)];
"""


def test_all_kernels_is_a_line_each_problems_first_and_filters(page):
    page(KERNELS + """
const rows = renderKernelList(nights, db);
// softmax fails a case on npu1: first; the rest by name.
assert.deepEqual(rows.map(r => [r.factory, r.problem]), [['softmax', true], ['cascade_mm', false], ['relu', false]]);
// Per NPU: its state, its cases' cycles now, and how many moved over the window.
assert.deepEqual(rows[0].children.slice(1, 3).map(c => c.text), ['npu11 failing50 and 1,100 cycles1 slower since 1 Jan', 'npu2passes2,200 cycles']);
assert.deepEqual(rows[1].children.slice(1, 3).map(c => c.text), ['npu1not run', 'npu2not offered']);
// Plain dots and words: not the old map's filled halves.
assert.deepEqual(rows[0].children.slice(1, 3).map(c => c.className), ['k-npu k-bad', 'k-npu k-ok']);
// Its cases' cycles now, and how many moved since the first nightly shown.
assert.deepEqual(kernelWindow(db, 'npu1', 'softmax'), { cycles: [50, 1100], since: 2, worse: 1, better: 0,
  moves: ['softmax/1024/bfloat16: 1,000 on 1 Jan → 1,100, past the threshold from 1 Jan'] });
assert.equal(kernelWindow(db, 'npu1', 'relu'), null);
assert.equal(cyclesWords([50, 1100]), '50 and 1,100 cycles');
assert.equal(cyclesWords([5, 50, 1100]), '5 to 1,100 cycles');
// Filters: text, family, only problems.
$('kernels-filter').value = 'relu';
$('kernels-filter').oninput();
assert.deepEqual(rows.map(r => r.hidden), [true, true, false]);
assert.equal($('kernels-status').text, '1 kernel of 3');
$('kernels-filter').value = '';
$('kernels-problems').checked = true;
$('kernels-problems').onchange();
assert.deepEqual(rows.map(r => r.hidden), [false, true, true]);
""")


def test_refresh_preserves_kernel_filters_and_replaces_their_handlers(page):
    page(KERNELS + """
location.hash = '#view=kernels';
const family = $('kernels-family');
const replace = family.replaceChildren.bind(family);
family.replaceChildren = (...options) => { replace(...options); family.value = ''; };
let listeners = 0;
for (const id of ['kernels-filter', 'kernels-family', 'kernels-problems']) {
  $(id).addEventListener = () => { listeners++; };
}
const oldRows = renderKernelList(nights, db);
family.value = 'activation';
$('kernels-filter').value = 'softmax';
$('kernels-problems').checked = true;
const oldHandler = $('kernels-filter').oninput;
for (let i = 0; i < 3; i++) renderKernelList(nights, db);
assert.equal(family.value, 'activation');
assert.equal($('kernels-filter').value, 'softmax');
assert.equal($('kernels-problems').checked, true);
assert.equal(listeners, 0);
assert.notEqual($('kernels-filter').oninput, oldHandler);
const rows = $('kernel-list').children;
assert.deepEqual(rows.map(r => r.hidden), [false, true, true]);
$('kernels-filter').value = 'relu';
$('kernels-problems').checked = false;
$('kernels-filter').oninput();
assert.deepEqual(rows.map(r => r.hidden), [true, true, false]);
// Detached row trees are no longer visited by filter events.
assert.deepEqual(oldRows.map(r => r.hidden), [false, false, false]);
family.value = 'removed-family';
renderKernelList(nights, db);
assert.equal(family.value, '');
// A deep link initializes filters once, not again on every refresh.
location.hash = '#view=kernels&family=activation&problems=1';
renderKernelList(nights, db);
assert.equal(family.value, 'activation');
assert.equal($('kernels-problems').checked, true);
family.value = '';
$('kernels-problems').checked = false;
renderKernelList(nights, db);
applyKernelsFilter();
assert.equal(family.value, '');
assert.equal($('kernels-problems').checked, false);
""")


def test_kernel_page_names_the_kernel_and_tables_its_cases(page):
    page(KERNELS + """
assert.equal(renderKernel('softmax', nights), true);
const head = $('kernel-head');
assert.equal(head.children[0].text, 'softmax');
assert.equal(head.children[1].text, 'softmaxactivation');
const links = head.children[2].children.map(c => c.children[1].href);
assert.deepEqual(links, [
  'https://github.com/Xilinx/mlir-aie/blob/abcdef1/aie_kernels/activation/softmax.cc',
  'https://github.com/Xilinx/mlir-aie/blob/abcdef1/python/iron/kernels/activation.py',
  'https://github.com/Xilinx/mlir-aie/blob/abcdef1/test/python/npu/kernel_cases.py',
]);
assert.deepEqual(head.children[3].children.map(li => li.text),
  ['npu11 build, 3 cases pass, 2 timed 1 failing', 'npu21 build, 1 case pass, 1 timed']);
// A row per case: per NPU its latest cycles and the change over the window,
// or why it has none.
const [thead, tbody] = kernelCaseTable('softmax', nights, db).children[0].children;
assert.deepEqual(thead.children[0].children.map(c => c.text), ['Case', 'npu1 nowcycles, 1 Jan', 'Then1 Jan', 'npu2 nowcycles, 1 Jan', 'Then1 Jan']);
// Then and now in cycles; a case on one nightly only says so.
assert.deepEqual(tbody.children.map(r => r.children.map(c => c.text)), [
  ['1024bfloat16', '1,100', '1,000 +10.0%', '2,200', 'one nightly'],
  ['16bfloat16', 'timing failed', '—'],
  ['2048bfloat16', 'fails correctness', '—'],
  ['32bfloat16', 'correctness only', '—'],
  ['64bfloat16', '50', 'one nightly', '—'],
]);
// softmax/64 was not measured in the latest nightly: greyed, and said.
assert.ok(tbody.children[4].children[1].className.includes('stale'));
assert.equal(tbody.children[0].children[2].children[2].className, 'chg worse strong');
assert.equal(renderKernel('nope', nights), false);
""")


@pytest.mark.parametrize("visible", [False, True])
def test_kernel_refresh_disposes_charts_and_pending_observers(page, visible):
    page(KERNELS + f"const visible = {json.dumps(visible)};\n" + """
const observers = [], liveCharts = new Set();
let created = 0, destroyed = 0;
global.Chart = function () { created++; liveCharts.add(this); };
Chart.prototype.destroy = function () {
  assert.ok(liveCharts.delete(this), 'a chart must be destroyed exactly once');
  destroyed++;
};
global.IntersectionObserver = class {
  constructor(cb) { this.cb = cb; observers.push(this); }
  observe() {}
  disconnect() { this.disconnected = true; }
  fire() { this.cb([{ isIntersecting: true }]); }
};
hist = db;
loadHistories = async () => db;
location.hash = hashFor({ view: 'kernel', kernel: 'softmax' });
(async () => {
  const show = async factory => { showKernel(factory); await Promise.resolve(); };
  await show('softmax');
  for (let i = 0; i < 4; i++) {
    const old = $('kernel-trend').children[0], observer = observers.at(-1);
    if (visible) {
      observer.fire();
      old.redraw();
      assert.equal(liveCharts.size, 1);
    }
    // Keep the live chart until replacement history arrives.
    showKernel('softmax');
    assert.equal(liveCharts.size, visible ? 1 : 0);
    await Promise.resolve();
    assert.ok(observer.disconnected);
    assert.equal(liveCharts.size, 0);
    const count = created;
    observer.fire();
    old.redraw();
    assert.equal(created, count, 'a disposed chart cannot draw a detached canvas');
    assert.equal(componentCharts.length, 0, 'kernel charts are not retained by sections');
  }
  // NPU changes replace the trend through the same cleanup path.
  observers.at(-1).fire();
  $('kernel-npu').children[1].on.click();
  await Promise.resolve();
  assert.equal(liveCharts.size, 0);
  observers.at(-1).fire();
  await show('relu');
  assert.equal(liveCharts.size, 0);
  assert.ok($('kernel-trend').text.includes('No nightly has timed'));
  await show('softmax');
  observers.at(-1).fire();
  await show('missing');
  assert.equal(liveCharts.size, 0);
  await show('softmax');
  observers.at(-1).fire();
  location.hash = '#view=kernels';
  showView();
  assert.equal(liveCharts.size, 0);
  assert.equal(created, destroyed);
})().catch(e => { console.error(e); process.exitCode = 1; });
""")


def test_case_titles_and_trend_lines(page):
    page("""
assert.deepEqual(caseParts('bn_conv2dk3/1792x8/int8_uint8/input_channels=8/stride=2/lut'),
  { factory: 'bn_conv2dk3', shape: '1792x8', dtype: 'int8_uint8', params: ['input channels 8', 'stride 2', 'lut'] });
assert.equal(caseTitle('mm/64x32x64x16/int16_int32').text, '64 × 32 × 64 × 16int16 int32');
""")


def test_a_kernels_chart_has_a_line_per_case_in_actual_cycles(page):
    page(KERNELS + """
const config = kernelTrendConfig(db, 'npu1', 'softmax', 'cycles');
assert.deepEqual(config.data.labels, ['1 Jan', '1 Jan']);
// Each case's actual cycles, night by night.
assert.deepEqual(config.data.datasets.map(d => [d.label, d.data]), [['64/bfloat16', [50, null]], ['1024/bfloat16', [1000, 1100]]]);
const tips = config.options.plugins.tooltip.callbacks;
assert.equal(tips.label({ dataset: config.data.datasets[1], dataIndex: 1, raw: 1100 }), '1024/bfloat16: 1,100 (median 1150 max 1200 n=16)');
// The tooltip always repeats the axis tick's own formatting of the value.
assert.ok(tips.label({ dataset: config.data.datasets[1], dataIndex: 1, raw: 1100 }).includes(config.options.scales.y.ticks.callback(1100)));
assert.equal(tips.afterTitle([{ dataIndex: 1 }]), 'abcdef1 same revision');
assert.equal(config.options.scales.y.title.text, 'Cycles per call · ↓ better');
// 50 and 1,100 differ over ten times: a log axis, so both lines' moves show.
assert.equal(config.options.scales.y.type, 'logarithmic');
assert.equal(kernelTrendConfig(db, 'npu2', 'softmax', 'cycles').options.scales.y.type, 'linear');
// softmax_mask is another kernel, not a case of softmax.
assert.ok(!config.data.datasets.some(d => d.label.startsWith('mask')));
""")


def test_views_come_from_the_hash_and_old_links_keep_working(page):
    page("""
location.hash = '#view=kernels';
assert.equal(applyView(), 'kernels');
assert.equal($('kernels').hidden, false);
assert.equal($('kernels-section').hidden, false);
assert.equal($('dashboard').hidden, true);
assert.equal($('tab-kernels').className, 'current');
assert.equal($('nav-kernels').className, 'current');
assert.equal($('nav-overview').className, '');
location.hash = '#view=kernel&kernel=softmax';
assert.equal(applyView(), 'kernel');
assert.equal($('tab-kernel').hidden, false);
assert.equal($('kernel').hidden, false);
// No hash, or one unrecognized, is the overview.
location.hash = '';
assert.equal(applyView(), 'overview');
assert.equal($('overview').hidden, false);
assert.equal($('kernels-section').hidden, true);
assert.equal($('nav-overview').className, 'current');
location.hash = '#view=night';
assert.equal(applyView(), 'night');
assert.equal($('dashboard').hidden, false);
assert.equal($('tab-night').className, 'current');
assert.equal($('tab-kernel').hidden, true);
// The kernels' old Trends: a chart per case became each kernel's page.
location.hash = '#view=charts&metric=cycles';
assert.equal(applyView(), 'night');
location.hash = '#view=charts&metric=cycles&show=all&kernel=softmax%2F1024%2Fbf16';
assert.equal(applyView(), 'kernel');
// The old components view goes to the section that charts its metric.
location.hash = '#view=components';
assert.equal(applyView(), 'placement');
assert.equal($('placement').hidden, false);
assert.equal($('kernels-section').hidden, true);
assert.equal($('nav-placement').className, 'current');
location.hash = '#view=components&metric=us_per_image';
assert.equal(applyView(), 'performance');
assert.equal($('performance').hidden, false);
assert.equal($('placement').hidden, true);
// "routing" was its own tab before the router joined placement's as
// compiler checks.
location.hash = '#view=routing';
assert.equal(applyView(), 'placement');
assert.equal($('placement').hidden, false);
// "design" was the performance section's first name; an unrecognized
// hash lands on the overview.
location.hash = '#view=design';
assert.equal(applyView(), 'performance');
location.hash = '#view=nonsense';
assert.equal(applyView(), 'overview');
""")


def test_npu_us_moves_only_past_its_noise(page):
    page("""
const us = (value, mad) => ({ value, unit: 'us', ...(mad === undefined ? {} : { range: `± ${mad}; min 1 max 2 n=50` }) });
// 20% moves: past 3x a 1 us MAD; not past 3x 10 us (either run's); no MAD, the percentage alone.
assert.equal(verdict('dispatch_us', us(100, 1), us(120, 1)), 'worse');
assert.equal(verdict('dispatch_us', us(100, 10), us(120, 1)), '');
assert.equal(verdict('dispatch_us', us(100, 1), us(120, 7)), '');
assert.equal(verdict('dispatch_us', us(100), us(120)), 'worse');
assert.equal(verdict('dispatch_us', us(120, 1), us(100, 1)), 'better');
// Many MADs but under 15%.
assert.equal(verdict('dispatch_us', us(100, 0.1), us(105, 0.1)), '');
assert.equal(verdict('npu_us', us(100, 2), us(109, 2)), '');
assert.equal(verdict('npu_us', us(100, 2), us(110, 2)), '');
assert.equal(verdict('npu_us', us(100, 2), us(111, 2)), 'worse');
assert.equal(verdict('npu_us', us(100, 3), us(110, 2)), '');
// Other metrics ignore the range.
assert.equal(verdict('cycles', { value: 100, range: '± 50' }, { value: 103 }), 'worse');

db = fromRecords({ npu1: [
  rec('1', 1000, 'performance', { 'noisy/1/bf16/dispatch_us': ['us', 100, '± 10.0; min 1 max 2 n=50'],
                                   'steady/1/bf16/dispatch_us': ['us', 100, '± 1.0; min 1 max 2 n=50'] }),
  rec('2', 2000, 'performance', { 'noisy/1/bf16/dispatch_us': ['us', 125, '± 10.0; min 1 max 2 n=50'],
                                  'steady/1/bf16/dispatch_us': ['us', 125, '± 1.0; min 1 max 2 n=50'] }),
]});
assert.deepEqual(movedSeries(db, 'npu1').regressed.map(x => x.series.kase), ['steady/1/bf16']);
const { cases } = latestCases(db, 'npu1');
assert.equal(cases.get('noisy/1/bf16').metrics.get('dispatch_us').cls, '');
assert.equal(cases.get('steady/1/bf16').metrics.get('dispatch_us').cls, 'worse');
const move = latestMove(db.series.find(s => s.kase === 'noisy/1/bf16'), ['performance']);
assert.deepEqual(move, { change: 0.25, delta: 25, cls: '', since: 1000 });
""")


def test_a_newer_format_is_set_aside_and_said_so(page):
    page(MOVED + RUNS + """
assert.deepEqual(readable({ schema: 1 }), { schema: 1 });
assert.ok(readable({}));
assert.equal(readable({ schema: 2 }), null);
assert.equal(readable(null), null);
const run = latestRun('npu1', { ...index, schema: 2 }, db, catalogue);
assert.equal(run.newer, true);
assert.equal(run.recorded, false);
assert.equal(warningsFor('npu1', run, 3600 * 1000)[0].text,
  'the results were published by a newer version of this page: reload it');
""")


def test_the_card_names_the_part_and_keeps_the_raw_name(page):
    page(MOVED + RUNS + """
const index1 = { runs: [{ ...index.runs[1], device: 'Phoenix', device_raw: 'RyzenAI-npu1' }] };
renderDashboard(['npu1'], db, new Map([['npu1', { ...catalogue, kernels: [] }]]), new Map([['npu1', index1]]), 3600 * 1000);
const name = $('verdicts').children[0].children[0].children[0];
assert.equal(name.className, 'npu-name');
assert.equal(name.text, 'npu1Phoenix, aie2');
assert.equal(name.children[1].title, 'reported as RyzenAI-npu1');
""")


COMPONENT_RUNS = """
const all = (n, f) => typeof n === 'string' ? [] : [...(f(n) ? [n] : []), ...n.children.flatMap(c => all(c, f))];
const one = (n, f) => { const found = all(n, f); assert.equal(found.length, 1); return found[0]; };
// A band's latest-numbers table as text, a row each: [head, ...cells].
const tableOf = n => all(n, x => x.tag === 'tr').map(tr => tr.children.map(c => c.text));
const [sweep, hwPlacement, hw] = BANDS;
const night = (id, date, pmode, extra) => ({
  id, url: `https://example.com/runs/${id}`, date, commit: { ...commit }, pmode, provenance: {},
  sane: true, published: true, failed: [], ...(extra || {}),
});
const NOW = Date.parse('2026-10-02T07:00:00Z');
"""


def test_component_bands_judge_steps_and_trends_over_two_weeks(page):
    page(COMPONENT_RUNS + """
{
const yolo = BANDS.find(b => b.id === 'yolo26n-hw');
// Fourteen nightlies, a day apart, each its own commit.
const days = Array.from({ length: 14 }, (_, i) => ({ id: String(i + 1), date: new Date(Date.parse('2026-10-01T06:30:00Z') + i * 86400000).toISOString(),
  commit: { id: `${String(i + 1).padStart(2, '0')}`.repeat(20) }, pmode: 'turbo', provenance: {} }));
const file = (target, metric, unit, series) => ({ schema: 1, target, metric, unit, runs: days, series: Object.fromEntries(
  Object.entries(series).map(([k, values]) => [k, { values }])) });
const flat = (v, n = 14) => Array.from({ length: n }, (_, i) => v + (i % 3) * 0.004 * v);
const steady = flat(11600);
const db = collect([
  // A step last night: outside the whole week before, past 5%.
  file('yolo26n-hw', 'npu_us_per_image_b1', 'us', { 'yolo26n/640': [...flat(11600, 13), 12400] }),
  // A creep: 2% a night for a week, never a step on its own.
  file('yolo26n-hw', 'us_per_image_b1', 'us', { 'yolo26n/640': [...flat(12500, 7), 12750, 13000, 13260, 13530, 13800, 14080, 14080] }),
  // Within its usual bounce: nothing.
  file('yolo26n-hw', 'us_per_image_b8', 'us', { 'yolo26n/640': [6000, 6400, 6000, 6400, 6000, 6400, 6000, 6400, 6000, 6400, 6000, 6400, 6000, 6400] }),
  // Better: listed as such.
  file('yolo26n-hw', 'compile_s', 's', { 'yolo26n/640': [...flat(33, 13), 20] }),
]);
const t = trendMoves(db, yolo);
assert.deepEqual(t.steps.map(x => [x.series.metric, x.kind]), [['npu_us_per_image_b1', 'step']]);
assert.deepEqual(t.trends.map(x => [x.series.metric, x.kind, x.since]), [['us_per_image_b1', 'trend', Date.parse(days[0].date)]]);
assert.deepEqual(t.better.map(x => x.series.metric), ['compile_s']);
assert.equal(t.blips.length, 0);
// The band: amber, saying both, with the commits behind the step.
const latest = { id: '14', date: days[13].date, rows: { 'yolo26n/640': { npu_us_per_image_b1: { value: 12400, unit: 'us' }, images_passed: { value: 128, unit: 'images' } } } };
const index = { target: 'yolo26n-hw', runs: days.map(d => night(d.id, d.date, 'turbo')) };
const card = componentBand(yolo, index, latest, Date.parse(days[13].date) + 3600000, null, db);
assert.equal(card.className, 'verdict warn');
assert.ok(one(card, n => n.className === 'headline').text.includes('NPU time / image (b1) +6.5% stepped worse on 14 Oct; end-to-end / image (b1) +12.2% crept worse over two weeks.'), one(card, n => n.className === 'headline').text);
const changes = one(card, n => n.className === 'changes');
assert.deepEqual(all(changes, n => n.className && n.className.startsWith('tag ')).map(n => n.text), ['Step 14 Oct', 'Since 1 Oct', 'Better 14 Oct']);
const compare = one(changes, n => n.tag === 'a');
assert.equal(compare.href, 'https://github.com/Xilinx/mlir-aie/compare/131313131313...141414141414');
// Each line has its two weeks as bars, the last the step's.
const spark = one(changes.children[0].children[0], n => n.className === 'spark worse');
assert.equal(spark.children.length, 14);
assert.equal(spark.children.at(-1).className, 'last');

// One batch of many moving while the rest held: a blip, folded away, and the band stays green.
const seeds = {};
for (const seed of [2, 3, 7]) for (const b of [1, 2, 4, 8, 16, 64]) seeds[`mobilenet/seed=${seed}/batch=${b}/effort=default`] = flat(90);
seeds['mobilenet/seed=3/batch=1/effort=default'] = [...flat(90, 13), 180];
// Another seed's moves are the placer's, not Performance's: not judged here at all.
seeds['mobilenet/seed=2/batch=8/effort=default'] = [...flat(90, 13), 400];
const hwDb = collect([file('sa-placer-hw', 'host_overhead_us_per_image', 'us', seeds)]);
const ht = trendMoves(hwDb, hw);
assert.deepEqual([ht.steps.length, ht.trends.length, ht.blips.length], [0, 0, 1]);
assert.equal(ht.blips[0].series.kase, 'mobilenet/seed=3/batch=1/effort=default');
const blipBox = changesBlock(ht, hw);
assert.equal(one(blipBox, n => n.tag === 'summary').text, '1 move on one batch only, the rest held: likely noise');
// Without a history (an older publish): the night before, as it was.
assert.equal(trendMoves(null, yolo).steps.length, 0);
}
""")


def test_the_mobilenet_card_shows_the_sweeps_mobilenet_beside_its_placements(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
// The seed sweep's MobileNet snapshot: 20 seeds, one failed.
const sweepLatest = { id: '7', date: hwAfter.date, rows: { mobilenet: {
  final_cost_mean: { value: 294.6, unit: 'cost' }, final_cost_max: { value: 305, unit: 'cost' } } },
  detail: Array.from({ length: 20 }, (_, i) => ({ fixture: 'mobilenet', seed: i + 1, passed: i !== 10 })) };
const card = componentBand(hwPlacement, hwIndex, hwAfter, NOW, hwBefore, null, new Map([['sa-placer', sweepLatest]]));
const last = tableOf(card.table).at(-1);
assert.deepEqual([last[0], last[1], last[2], last.at(-1)], ['seeds 1–20, full effort (sweep)', 'snapshot', '294.6 mean, 305 worst', '19 of 20 placed']);
assert.equal(all(card.table, n => n.className === 'vals worse').length, 1);
// One card for MobileNet's placement: its tabs, the sweep's history among them.
assert.equal(hwPlacement.title, 'MobileNet placement');
assert.deepEqual(hwPlacement.trend.map(t => t.name), ['Placement cost', 'Seed sweep', 'Compile time', 'Compile memory']);
assert.deepEqual(hwPlacement.alsoFrom, { check: 'sa-placer', fixture: 'mobilenet' });
// Without the sweep's record, no row.
assert.equal(tableOf(componentBand(hwPlacement, hwIndex, hwAfter, NOW, hwBefore).table).at(-1)[0], 'seed 7, full effort');
""")


def test_moves_are_named_largest_first_and_the_rest_counted(page):
    page("""
const mv = (metric, kase, before, after) => ({ series: { metric, kase, unit: 'bytes' }, before, after, change: (after - before) / before });
const xs = [mv('elf_bytes', 'llama/N=16', 100, 184), mv('compile_s', 'llama/N=16', 10, 16), mv('compile_s_warm', 'llama/N=16', 10, 15)];
// The overview's: one by name, no stray separator.
assert.equal(namedMoves(xs, [], 1), 'N 16, full ELF +84.0% and 2 more');
assert.equal(namedMoves(xs.slice(0, 1), [], 1), 'N 16, full ELF +84.0%');
assert.equal(namedMoves(xs, ['N 16'], 2), 'full ELF +84.0%, compile time (cold) +60.0% and 1 more');
assert.equal(namedMoves(xs.slice(0, 2), ['N 16'], 2), 'full ELF +84.0% and compile time (cold) +60.0%');
""")


def test_a_failed_sweep_names_its_seeds_and_how_to_run_each_again(page):
    page(COMPONENT_RUNS + """
const failed = ['test_sa_effort/seed4', 'mobilenet/seed11', 'mobilenet/seed12', 'mobilenet/seed13', 'mobilenet/seed14'];
const index = { target: 'sa-placer', metrics: ['failed_seeds'], runs: [night('7', '2026-10-02T06:30:00Z', null, { failed })] };
const latest = { id: '7', date: '2026-10-02T06:30:00Z', rows: {
  test_sa_effort: { failed_seeds: { value: 1, unit: 'seeds' }, final_cost_mean: { value: 12.25, unit: 'cost' } },
  mobilenet: { failed_seeds: { value: 4, unit: 'seeds' }, cpu_ms_max: { value: 21000, unit: 'ms' } },
} };
const card = componentBand(sweep, index, latest, NOW);
assert.equal(card.className, 'verdict bad');
assert.equal(one(card, n => n.className === 'headline').text,
             '5 seeds failed: test_sa_effort/seed4, mobilenet/seed11, mobilenet/seed12, mobilenet/seed13 and 1 more.');
// No power mode: a sweep runs on a hosted runner, with no NPU.
assert.ok(!card.text.includes('power mode'));
// A row per fixture; a nonzero failed-seeds count is red.
assert.deepEqual(tableOf(card.table), [
  ['', 'Seeds placed', 'Cost, mean', 'Cost, worst', 'CPU, mean', 'CPU, slowest', 'Peak memory'],
  ['test_sa_effort', '1', '12.3', '—', '—', '—', '—'],
]);
// MobileNet's sweep is on the MobileNet card; its failures still fail this one.
assert.ok(all(card.table, n => n.className === 'vals worse').length === 1);
// A record without detail, as published before it, has no Seeds chart.
assert.equal(all(card, n => n.className === 'charts').length, 0);
const repro = one(card, n => n.className === 'repro');
// Beside the verdict, shut: a click away.
assert.ok(!repro.open && card.children.includes(repro));
assert.ok(repro.text.includes('artifact component-checks-sa-placer-7'));
assert.deepEqual(one(repro, n => n.tag === 'pre').text.split('\\n'), [
  'git checkout abcdef123456',
  "aie-opt '--aie-place-tiles=placer=sa_placer sa-seed=4 sa-effort=1.0' --mlir-pass-statistics test/place-tiles/sa_placer/test_sa_effort.mlir",
  "aie-opt '--aie-place-tiles=placer=sa_placer sa-seed=11 sa-effort=1.0' --mlir-pass-statistics utils/component_checks/fixtures/mobilenet.mlir",
  "aie-opt '--aie-place-tiles=placer=sa_placer sa-seed=12 sa-effort=1.0' --mlir-pass-statistics utils/component_checks/fixtures/mobilenet.mlir",
  "aie-opt '--aie-place-tiles=placer=sa_placer sa-seed=13 sa-effort=1.0' --mlir-pass-statistics utils/component_checks/fixtures/mobilenet.mlir",
  "aie-opt '--aie-place-tiles=placer=sa_placer sa-seed=14 sa-effort=1.0' --mlir-pass-statistics utils/component_checks/fixtures/mobilenet.mlir",
  '# every seed, as the nightly runs them:',
  sweep.whole,
]);
""")


SWEEP_NIGHTS = """
// Two sweep nightlies of one fixture over seeds 1..4: the latest costs
// more at seed 2 and fails seed 4.
const seedRuns = (costs, cpu) => costs.map((final_cost, i) => ({ fixture: 'test_sa_effort', seed: i + 1,
  passed: final_cost !== null, final_cost, cpu_ms: cpu, peak_rss_mb: 200 }));
const sweepRecord = (id, date, costs, cpu) => {
  const passed = costs.filter(c => c !== null);
  const cell = (value, unit) => ({ value, unit });
  return { id, date, pmode: null, commit: { ...commit }, provenance: {}, detail: seedRuns(costs, cpu), rows: { test_sa_effort: {
    failed_seeds: cell(costs.length - passed.length, 'seeds'),
    final_cost_mean: cell(passed.reduce((a, b) => a + b, 0) / passed.length, 'cost'),
    final_cost_max: cell(Math.max(...passed), 'cost'),
    cpu_ms_mean: cell(cpu, 'ms'), cpu_ms_max: cell(cpu, 'ms'), peak_rss_mb_max: cell(200, 'MB'),
  } } };
};
const before = sweepRecord('6', '2026-10-01T06:30:00Z', [100, 100, 100, 100], 2000);
const after = sweepRecord('7', '2026-10-02T06:30:00Z', [100, 110, 100, null], 2100);
const sweepIndex = { target: 'sa-placer', runs: [night('6', before.date, null), night('7', after.date, null, { failed: ['test_sa_effort/seed4'] })] };
"""


def test_the_sweep_table_and_seeds_chart_compare_with_the_nightly_before(page):
    page(COMPONENT_RUNS + SWEEP_NIGHTS + """
assert.equal(previousPublished(sweepIndex, after).id, '6');
const card = componentBand(sweep, sweepIndex, after, NOW, before);
assert.equal(one(card, n => n.className === 'headline').text, '1 seed failed: test_sa_effort/seed4.');
// The worst cost moved 10%, past its 1%; CPU time 5%, within its 25%.
const moves = one(card, n => n.className === 'moves');
// A seed that failed is a count from zero: its floor of 1 flags it.
assert.equal(moves.text, 'Since 1 Oct:test_sa_effort failed seeds +1test_sa_effort SA final cost (worst) +10.0%test_sa_effort SA final cost (mean) +3.3%');
// A row per fixture: how many seeds placed, then each figure and its change.
assert.deepEqual(tableOf(card.table), [
  ['', 'Seeds placed', 'Cost, mean', 'Cost, worst', 'CPU, mean', 'CPU, slowest', 'Peak memory'],
  ['test_sa_effort', '3 of 4', '103.3 +3.3%', '110 +10.0%', '2,100 ms', '2,100 ms', '200 MiB'],
]);
assert.equal(all(card.table, n => n.tag === 'tbody')[0].children[0].children[0].title, 'test/place-tiles/sa_placer/test_sa_effort.mlir');
// Past the cost's 1%: strong; the CPU's 5% within its 25%: noise, not shown.
assert.deepEqual(all(card.table, n => /^chg /.test(n.className || '')).map(n => n.className),
                 ['chg worse strong', 'chg worse strong']);
assert.ok(all(card.table, n => n.className === 'vals worse').length === 1);
// The Seeds chart: a dot per seed on a zoomed axis (no bars measured from its floor),
// the nightly before as ticks, seed 4 a ✕.
const [bars, ticks, crosses] = chart.data.datasets;
assert.ok(chart.data.datasets.every(d => d.type !== 'bar' && d.showLine === false));
assert.equal(chart.type, 'line');
assert.deepEqual(chart.data.labels, ['1', '2', '3', '4']);
assert.deepEqual(bars.data, [100, 110, 100, null]);
assert.deepEqual(ticks.data, [100, 100, 100, 100]);
assert.deepEqual(crosses.data.map(v => v !== null), [false, false, false, true]);
assert.equal(crosses.data[3], chart.options.scales.y.min);
assert.equal(chart.options.plugins.tooltip.callbacks.afterBody([{ dataIndex: 1 }]), 'CPU 2,100 ms, peak 200 MiB');
assert.equal(chart.options.plugins.tooltip.callbacks.label({ dataset: crosses, raw: 0 }), 'did not place');
""")


HW_NIGHTS = """
// Two hardware nightlies, seeds 3 and 7 at batch 1 and 16, each at the
// design's real default effort and a full-effort baseline: seed 3's batch
// 16 (default effort) is 7% slower in the latest, past its 5% and its MADs.
// The full-effort numbers are fixed (not parameterized by b16) and clearly
// distinct from default's, so a test can tell the two apart.
const hwUs = (b16, seed, effort, batch) =>
  batch === 1 ? 295.7 : (effort === 'default' && seed === 3 ? b16 : (effort === 'default' ? 99 : 120));
const hwRecord = (id, date, b16, extra) => {
  const rows = {};
  for (const seed of [3, 7]) {
    for (const effort of ['default', 'full']) {
      // One compile per seed and effort serves every batch.
      rows[`mobilenet/seed=${seed}/effort=${effort}`] = {
        placement_cost: { value: (effort === 'default' ? 300 : 400) + seed, unit: 'cost', range: `placement 0123456789ab` },
        compile_s: { value: effort === 'default' ? 40 : 55, unit: 's' },
        compile_peak_rss_mb: { value: 850, unit: 'MB' },
        compile_s_warm: { value: effort === 'default' ? 4.6 : 25, unit: 's' },
        compile_peak_rss_mb_warm: { value: 700, unit: 'MB' },
      };
      for (const batch of [1, 16]) {
        const us = hwUs(b16, seed, effort, batch);
        const e2e = us + 100;
        rows[`mobilenet/seed=${seed}/batch=${batch}/effort=${effort}`] = {
          us_per_image: { value: us, unit: 'us', range: `median ${us + 1}, MAD 0.20, p95 ${us + 2}` },
          npu_us_per_image: { value: Math.round((us - 47 / batch) * 10) / 10, unit: 'us', range: 'median 1, MAD 0.20, p95 2' },
          host_overhead_us_per_image: { value: 100, unit: 'us', range: 'median 101, MAD 0.40, p95 103' },
          e2e_us_per_image: { value: e2e, unit: 'us' },
          fps: { value: Math.round(1e6 / e2e * 100) / 100, unit: 'fps' },
        };
      }
    }
  }
  rows.mobilenet = { failed_seeds: { value: 0, unit: 'seeds' } };
  const detail = [3, 7].flatMap(seed => ['default', 'full'].flatMap(effort_label =>
    [1, 16].map(batch => {
      const us = hwUs(b16, seed, effort_label, batch);
      const e2e = us + 100;
      return {
        seed, batch, effort_label, passed: true, placement: '0123456789ab',
        npu_us_per_image: Math.round((us - 47 / batch) * 10) / 10,
        e2e_us_per_image: e2e, fps: Math.round(1e6 / e2e * 100) / 100,
      };
    })));
  return { id, date, pmode: 'turbo', commit: { ...commit }, provenance: {}, rows, detail, ...(extra || {}) };
};
const hwBefore = hwRecord('6', '2026-10-01T06:30:00Z', 98);
const hwAfter = hwRecord('7', '2026-10-02T06:30:00Z', 104.9);
const hwIndex = { target: 'sa-placer-hw', runs: [night('6', hwBefore.date, 'turbo'), night('7', hwAfter.date, 'turbo')] };
"""


def test_a_passing_check_says_so_and_how_to_run_it(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
const card = componentBand(hw, hwIndex, hwAfter, NOW, hwBefore);
// Every run passed, but seed 3's batch 16 got slower: a warning, not "all passing".
assert.equal(card.className, 'verdict warn');
assert.equal(one(card, n => n.className === 'headline').text,
             '2 measures got worse vs 1 Oct.');
const repro = one(card, n => n.className === 'repro');
assert.ok(!repro.open);
// Shown: the verdict, what moved, the latest numbers and the trend chart's
// slot. Behind About: what the check does, how to run it, the run.
const top = card.children.map(c => c.className);
assert.deepEqual(top, ['verdict-top', 'headline', 'moves', 'trend-slot', 'band-fold', 'band-more']);
// The batch table: after the chart, folded.
assert.equal(card.children[4].children[0].text, 'Every batch, last night');
assert.ok(!card.children[4].open && card.children[4].children[1] === card.table);
const more = card.children.at(-1);
assert.ok(!more.open);
assert.equal(more.children[0].text, 'About this check, and how to run it');
assert.ok(all(more, n => n.className === 'repro').length === 1);
assert.equal(card.trend.className, 'trend-slot');
// A quiet night: "nothing moved" is detail too.
const quiet = componentBand(hw, hwIndex, hwRecord('7', hwAfter.date, 98), NOW, hwBefore);
// A quiet night says nothing: no news is good news.
assert.deepEqual(quiet.children.map(c => c.className), ['verdict-top', 'trend-slot', 'band-fold', 'band-more']);
assert.equal(quiet.className, 'verdict ok');
assert.ok(one(repro, n => n.tag === 'pre').text.includes('python -m mobilenet.aie2_mobilenet_iron --sa-seed N --sa-effort 0.05 --dynamic 1 16 -w 20 -i 20'));
assert.ok(card.text.includes('turbo mode'));
// Nothing published yet.
const empty = componentBand(hw, null, null, NOW);
assert.equal(empty.className, 'verdict none');
assert.equal(one(empty, n => n.className === 'headline').text, 'No results published yet.');
""")


def test_the_mobilenet_table_shows_each_seed_against_the_nightly_before(page):
    preview = (ROOT / "utils/kernel_checks/preview.py").read_text(encoding="utf-8")
    detail = re.search(r"^HW_DETAIL = \((.*?)^\)", preview, re.MULTILINE | re.DOTALL)
    assert detail and '"fps"' in detail[1]
    page(COMPONENT_RUNS + HW_NIGHTS + """
// Chart.js configs are normally captured one at a time in `chart`; the
// Batch scaling card draws two, so collect every one drawn here.
const drawnCharts = [];
global.Chart = function (_canvas, config) { drawnCharts.push(config); chart = config; };
Chart.prototype.destroy = function () {};
const card = componentBand(hw, hwIndex, hwAfter, NOW, hwBefore);
assert.equal(one(card, n => n.className === 'moves').text,
             'Since 1 Oct:seed 3, batch 16, default effort NPU time / image +7.3%seed 3, batch 16, default effort dispatch / image +7.0%');
// The design as it ships, a row per batch: the NPU's time, the end-to-end
// time and the rate per image, and whether it verified; colored past the
// threshold only. The other seeds are Compiler checks'.
assert.deepEqual(tableOf(card.table), [
  ['', 'NPU / image', 'End to end / image', 'Images / s'],
  ['batch 1', '248.7 µs', '395.7 µs', '2,527.17 fps'],
  ['batch 16', '102 µs +7.3%', '204.9 µs', '4,880.43 fps'],
]);
assert.ok(!card.table.text.includes('seed 7'));
assert.deepEqual(all(card.table, n => /^chg (worse|better)/.test(n.className || '')).map(n => n.className), ['chg worse']);
assert.equal(all(card.table, n => n.tag === 'th' && n.text === 'NPU / image')[0].title, COMPONENT_METRICS.npu_us_per_image);
// The Batch scaling charts: last night, every batch on one machine, beside
// the trend tabs' over-nights view.
const charts = one(card, n => n.className === 'charts');
const boxes = all(charts, n => n.className === 'chart');
assert.equal(boxes.length, 2);
assert.equal(all(boxes[0], n => n.tag === 'h3')[0].text.startsWith('Time per image, by batch'), true);
assert.equal(all(boxes[1], n => n.tag === 'h3')[0].text.startsWith('Images per second, by batch'), true);
const [timeCfg, rateCfg] = drawnCharts;
assert.deepEqual(timeCfg.data.labels, ['batch 1', 'batch 16']);
assert.deepEqual(timeCfg.data.datasets.map(d => d.label), ['NPU', 'end to end']);
// The design's own placement (seed 3, default effort), last night only.
assert.deepEqual(timeCfg.data.datasets[0].data, [248.7, 102]);
assert.deepEqual(timeCfg.data.datasets[1].data.map(v => Math.round(v * 100) / 100), [395.7, 204.9]);
assert.deepEqual(rateCfg.data.labels, ['batch 1', 'batch 16']);
assert.equal(rateCfg.data.datasets[0].label, 'images / s');
assert.deepEqual(rateCfg.data.datasets[0].data.map(v => Math.round(v * 100) / 100), [2527.17, 4880.43]);
""")


def test_a_failed_run_is_named_by_seed_and_batch_with_its_repro(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
const failedAt = hwRecord('7', hwAfter.date, 104.9);
delete failedAt.rows['mobilenet/seed=7/batch=16/effort=default'];
failedAt.detail.find(d => d.seed === 7 && d.batch === 16 && d.effort_label === 'default').passed = false;
failedAt.failed = ['mobilenet/seed=7/batch=16/effort=default'];
const index = { target: 'sa-placer-hw', runs: [night('6', hwBefore.date, 'turbo'),
  night('7', hwAfter.date, 'turbo', { failed: ['mobilenet/seed=7/batch=16/effort=default'] })] };
// Seed 7 is not the design's placement: Performance stays on the design, a
// line pointing at Compiler checks, no seed named.
const perfCard = componentBand(hw, index, failedAt, NOW, hwBefore);
assert.equal(perfCard.className, 'verdict warn');
assert.equal(one(perfCard, n => n.className === 'headline').text, '2 measures got worse vs 1 Oct.');
assert.ok(perfCard.text.includes("1 run of another seed or effort failed; the design's own placement is fine: see Compiler checks."));
assert.ok(!perfCard.table.text.includes('seed 7'));
// Compiler checks names it, red, with the repro beside the verdict.
const card = componentBand(hwPlacement, index, failedAt, NOW, hwBefore);
assert.equal(card.className, 'verdict bad');
assert.equal(one(card, n => n.className === 'headline').text, '1 run failed: seed 7 at batch 16, default effort.');
assert.ok(card.children.some(c => c.className === 'repro'));
// The failed seed's row says where, red, never the night before's number.
const seed7default = tableOf(card.table).find(r => r[0] === 'seed 7, default effort');
assert.deepEqual([seed7default[4], seed7default.at(-1)], ['failed', 'failed at b16']);
const repro = one(card, n => n.className === 'repro');
// Beside the verdict, shut: a click away.
assert.ok(!repro.open && card.children.includes(repro));
assert.deepEqual(one(repro, n => n.tag === 'pre').text.split('\\n').slice(0, 3), [
  'git checkout abcdef123456',
  'cd programming_examples/ml && python -m mobilenet.aie2_mobilenet_iron --sa-seed 7 --sa-effort 0.05 --dynamic 1 16 -w 20 -i 20   # batch 16 failed',
  '# every run, as the nightly runs them:',
]);
// A record that carries its efforts and iterations is rerun with its own.
failedAt.efforts = { default: 0.1, full: 1.0 };
failedAt.iters = 30;
assert.equal(hw.repro(failedAt, '7', '16', 'default'),
  'cd programming_examples/ml && python -m mobilenet.aie2_mobilenet_iron --sa-seed 7 --sa-effort 0.1 --dynamic 1 16 -w 20 -i 30   # batch 16 failed');
""")


LLAMA_NIGHTS = """
// Two llama nightlies from today's NPU2 numbers: the latest decodes 7% slower
// (4.70 -> 4.37 tok/s), past tok_s's 5%; nothing else moved.
const LLAMA = 'llama/persist_grow/N=16';
const llama = BANDS.find(b => b.id === 'llama-chain-hw');
const llamaRecord = (id, date, tok_s, extra) => ({
  id, date, pmode: 'turbo', commit: { ...commit }, provenance: {},
  rows: { [LLAMA]: {
    tokens_passed: { value: 4, unit: 'tokens', range: 'of 4' },
    tok_s: { value: tok_s, unit: 'tok/s' },
    ms_per_token: { value: 1000 / tok_s, unit: 'ms' },
    dma_floor_ratio: { value: 20.36, unit: 'x', range: 'compute-bound' },
    elf_bytes: { value: 21460224, unit: 'bytes' },
    pdi_bytes: { value: 243376, unit: 'bytes' },
    n_cores: { value: 30, unit: 'cores' },
    compile_s: { value: 60.5, unit: 's' },
    compile_peak_rss_mb: { value: 2502.4, unit: 'MB' },
    compile_s_warm: { value: 1.2, unit: 's' },
    compile_peak_rss_mb_warm: { value: 410, unit: 'MB' },
  } },
  detail: [{ case: LLAMA, passed: true, tokens_passed: 4, tokens_total: 4, tok_s }],
  failed: [], ...(extra || {}),
});
const llamaBefore = llamaRecord('6', '2026-10-01T06:30:00Z', 4.70);
const llamaAfter = llamaRecord('7', '2026-10-02T06:30:00Z', 4.37);
const llamaIndex = { target: 'llama-chain-hw', runs: [night('6', llamaBefore.date, 'turbo'), night('7', llamaAfter.date, 'turbo')] };
"""


def test_a_compile_is_a_stack_of_where_its_time_went(page):
    page("""
// A real cold compile of mobilenet, seed 3, default effort; the next night
// recorded no wait on kernels (the stage took no time).
const stages = { compile_s: 20.6, compile_place_s: 1.09, compile_route_s: 0.03, compile_lower_s: 0.45, compile_kernels_s: 15.85,
  compile_codegen_s: 0.19, compile_link_s: 0.24, compile_package_s: 0.07, compile_other_s: 0, compile_outside_aiecc_s: 2.79, compile_s_warm: 4.3 };
const night = (id, day, drop) => rec(id, Date.parse(`2026-10-0${day}T06:00:00Z`), 'turbo',
  Object.fromEntries(Object.entries(stages).filter(([k]) => k !== drop).map(([k, v]) => [`yolo26n/640/${k}`, ['s', v]])));
const data = fromRecords({ 'yolo26n-hw': [night('1', 1), night('2', 2, 'compile_kernels_s')] });
const config = stackConfig(data, { ...COMPILE_TABS[0], stack: COMPILE_TABS[0].split, missingIsZero: true });
const parts = config.data.datasets.filter(d => d.part);
// Six parts in pipeline order, each filled on the one below it.
assert.deepEqual(parts.map(d => d.label), COMPILE_GROUPS.map(([name]) => name));
assert.deepEqual(parts.map(d => d.fill), ['origin', '-1', '-1', '-1', '-1', '-1']);
assert.ok(config.options.scales.y.stacked && config.options.scales.y.beginAtZero);
assert.deepEqual(parts.map(d => d.data), [[1.09, 1.09], [0.45, 0.45], [15.85, 0], [0.43, 0.43], [0.1, 0.1], [2.79, 2.79]]);
// The wall times are lines over the stack, not added to it.
const lines = config.data.datasets.filter(d => !d.part);
assert.deepEqual(lines.map(d => [d.label, d.data, d.borderDash.length > 0]),
                 [['cold (nothing cached)', [20.6, 20.6], false], ['warm (kernel objects kept)', [4.3, 4.3], true]]);
assert.ok(new Set(config.data.datasets.map(d => d.stack)).size === 3);
// The tooltip lists the top part first, as the stack reads.
assert.ok(config.options.plugins.tooltip.itemSort({ datasetIndex: 1 }, { datasetIndex: 2 }) > 0);
// Without Chart.js, the SVG chart stacks the parts too: a polygon each, on the parts below.
const holder = new El('div'); holder.clientWidth = 600; holder.clientHeight = 270;
const box = new El('div'); box.append(holder);
const canvas = new El('canvas'); holder.append(canvas);
const drawn = svgChart(canvas, config);
assert.equal(drawn.svg.filter(p => p.startsWith('<polygon')).length, 6);
assert.equal(drawn.svg.filter(p => p.startsWith('<polyline')).length, 2);
assert.deepEqual(drawn.legend, [...COMPILE_GROUPS.map(([name]) => name), 'cold (nothing cached)', 'warm (kernel objects kept)']);
""")


def test_the_llama_table_reads_decode_rate_higher_as_better(page):
    page(COMPONENT_RUNS + LLAMA_NIGHTS + """
assert.equal(llama.section, 'performance');
const card = componentBand(llama, llamaIndex, llamaAfter, NOW, llamaBefore);
assert.equal(one(card, n => n.className === 'headline').text,
             '1 measure got worse vs 1 Oct.');
// generate.py's end-to-end decode rate leads when the night has it; the dispatch rate follows.
assert.equal(llamaSummary({ [LLAMA]: { ...llamaAfter.rows[LLAMA], decode_tok_s: { value: 4.2, unit: 'tok/s' } } }),
             ' 16 layers decode at 4.2 tok/s end to end; a dispatch runs at 4.37 tok/s, 228.8 ms per token, 20.36× the DMA floor (compute-bound).');
// Fewer tokens per second is the regression; time per token repeats it and is not listed twice.
assert.equal(one(card, n => n.className === 'moves').text, 'Since 1 Oct:persist_grow, N 16 dispatch rate −7.0%');
// One case: its figures on their side, a line each, said for what they are.
assert.deepEqual(tableOf(card.table), [
  ['Decode, dispatch only', '4.37 tok/s −7.0%'], ['Dispatch per token', '228.8 ms +7.6%'],
  ['Over the DMA floor', '20.36×'], ['Tokens verified', '4'],
  ['Compile, cold', '60.5 s'], ['Compile, warm', '1.2 s'], ['Full ELF', '20.5 MiB'],
]);
// Down is worse for tokens per second, up is worse for time per token.
assert.deepEqual(all(card.table, n => /^chg (worse|better)/.test(n.className || '')).map(n => n.className), ['chg worse', 'chg worse']);
// One case: no batch chart.
assert.equal(all(card, n => n.className === 'charts').length, 0);
""")


def test_a_failed_llama_run_names_the_decode_and_how_to_rerun_it(page):
    page(COMPONENT_RUNS + LLAMA_NIGHTS + """
const failedAt = llamaRecord('7', llamaAfter.date, 4.70, { failed: [LLAMA] });
failedAt.rows[LLAMA].tokens_passed = { value: 3, unit: 'tokens', range: 'of 4' };
Object.assign(failedAt.detail[0], { passed: false, tokens_passed: 3 });
const index = { target: 'llama-chain-hw', runs: [night('6', llamaBefore.date, 'turbo'),
  night('7', llamaAfter.date, 'turbo', { failed: [LLAMA] })] };
const card = componentBand(llama, index, failedAt, NOW, llamaBefore);
assert.equal(card.className, 'verdict bad');
assert.equal(one(card, n => n.className === 'headline').text, '1 run failed: the 16-layer decode.');
// Fewer tokens verified than generated: red in the table.
const verified = tableOf(card.table).find(r => r[0] === 'Tokens verified');
assert.equal(verified[1], '3 −25.0%');
assert.equal(all(card.table, n => n.className === 'vals worse').length, 1);
const repro = one(card, n => n.className === 'repro');
assert.ok(!repro.open && card.children.includes(repro));
assert.deepEqual(one(repro, n => n.tag === 'pre').text.split('\\n'), [
  'git checkout abcdef123456',
  'cd programming_examples/ml/llama_3_2_1b && python aie2_decode.py --p-max=4 --kset=8 --n-layers=16 --t=128 --full-elf-path build/chain_persist_grow_N16_PT4.elf && python test_decode.py --p-max=4 --kset=8 --n-layers=16 --t=128 --elf build/chain_persist_grow_N16_PT4.elf',
  '# every run, as the nightly runs them:',
  'python utils/component_checks/llama_chain_hw.py --out perf.json --meta meta.json',
]);
""")


YOLO_NIGHTS = """
// Two yolo26n nightlies: the latest takes 8% longer per image at batch 1
// (5000 -> 5400 us), past us_per_image_b1's 5%; nothing else moved.
const YOLO = 'yolo26n/640';
const yolo = BANDS.find(b => b.id === 'yolo26n-hw');
const yoloRecord = (id, date, us_b1, extra) => ({
  id, date, pmode: 'turbo', commit: { ...commit }, provenance: {},
  rows: { [YOLO]: {
    images_passed: { value: 8, unit: 'images', range: 'of 8' },
    ort_matches: { value: 8, unit: 'images', range: 'of 8' },
    us_per_image_b1: { value: us_b1, unit: 'us' },
    fps_b1: { value: 1e6 / us_b1, unit: 'fps' },
    us_per_image_b8: { value: 3000, unit: 'us' },
    fps_b8: { value: 1e6 / 3000, unit: 'fps' },
    elf_bytes: { value: 3 * 1048576, unit: 'bytes' },
    n_cores: { value: 32, unit: 'cores' },
    compile_s: { value: 95.5, unit: 's' },
    compile_peak_rss_mb: { value: 3100, unit: 'MB' },
    compile_s_warm: { value: 12.5, unit: 's' },
    compile_peak_rss_mb_warm: { value: 1200, unit: 'MB' },
  } },
  detail: [{ case: YOLO, passed: true, images_passed: 8, images_total: 8, ort_matches: 8 }],
  failed: [], ...(extra || {}),
});
const yoloBefore = yoloRecord('6', '2026-10-01T06:30:00Z', 5000);
const yoloAfter = yoloRecord('7', '2026-10-02T06:30:00Z', 5400);
const yoloIndex = { target: 'yolo26n-hw', runs: [night('6', yoloBefore.date, 'turbo'), night('7', yoloAfter.date, 'turbo')] };
"""


def test_the_yolo26n_table_shows_each_batch_and_what_it_verified(page):
    page(COMPONENT_RUNS + YOLO_NIGHTS + """
assert.equal(yolo.section, 'performance');
const card = componentBand(yolo, yoloIndex, yoloAfter, NOW, yoloBefore);
assert.equal(one(card, n => n.className === 'headline').text,
             '1 measure got worse vs 1 Oct.');
// The frame rate repeats the time per image and is not listed twice.
assert.equal(one(card, n => n.className === 'moves').text, 'Since 1 Oct:yolo26n end-to-end / image (b1) +8.0%');
// The times, a row per batch; then what it verified and its compile, a line each.
const [times, checked] = all(card.table, n => n.tag === 'table');
assert.deepEqual(tableOf(times).map(r => [r[0], r[2], r[3]]), [
  ['', 'End to end / image', 'Images / s'], ['batch 1', '5,400 µs +8.0%', '185.19 fps −7.4%'], ['batch 8', '3,000 µs', '333.33 fps']]);
// Up is worse for time per image, down for images per second.
assert.deepEqual(all(times, n => /^chg (worse|better)/.test(n.className || '')).map(n => n.className), ['chg worse', 'chg worse']);
// A measure the night did not record (mAP here) is left out.
assert.deepEqual(tableOf(checked).map(r => r[0]), ['Images verified', 'Match onnxruntime', 'Compile, cold', 'Compile, warm', 'Full ELF']);
assert.equal(tableOf(checked).find(r => r[0] === 'Full ELF')[1], '3.0 MiB');
assert.equal(tableOf(checked).find(r => r[0] === 'Compile, cold')[1], '95.5 s');
// One case: no batch chart.
assert.equal(all(card, n => n.className === 'charts').length, 0);
// A run that timed batch 1 alone says so, and has one row of times.
const b1Only = yoloRecord('7', yoloAfter.date, 78365.1);
for (const m of ['us_per_image_b8', 'fps_b8']) delete b1Only.rows[YOLO][m];
assert.equal(yoloSummary(b1Only.rows), ' End to end, one image takes 78,365.1 µs (12.76 fps) at batch 1.');
const alone = componentBand(yolo, yoloIndex, b1Only, NOW);
assert.deepEqual(tableOf(all(alone.table, n => n.tag === 'table')[0]).map(r => r[0]), ['', 'batch 1']);
// A run that was not timed has no sentence to say.
const untimed = yoloRecord('7', yoloAfter.date, 5000);
delete untimed.rows[YOLO].us_per_image_b1;
assert.equal(yoloSummary(untimed.rows), '');
""")


def test_a_failed_yolo26n_run_names_the_detector_and_how_to_rerun_it(page):
    page(COMPONENT_RUNS + YOLO_NIGHTS + """
const failedAt = yoloRecord('7', yoloAfter.date, 5000, { failed: [YOLO] });
failedAt.rows[YOLO].images_passed = { value: 7, unit: 'images', range: 'of 8' };
Object.assign(failedAt.detail[0], { passed: false, images_passed: 7, failed_images: ['000000000025.jpg'] });
const index = { target: 'yolo26n-hw', runs: [night('6', yoloBefore.date, 'turbo'),
  night('7', yoloAfter.date, 'turbo', { failed: [YOLO] })] };
const card = componentBand(yolo, index, failedAt, NOW, yoloBefore);
assert.equal(card.className, 'verdict bad');
assert.equal(one(card, n => n.className === 'headline').text, '1 run failed: the 640×640 detector.');
// Fewer images verified than run: red in the table.
assert.equal(tableOf(card.table).find(r => r[0] === 'Images verified')[1], '7 −12.5%');
assert.equal(all(card.table, n => n.className === 'vals worse').length, 1);
const repro = one(card, n => n.className === 'repro');
assert.ok(!repro.open && card.children.includes(repro));
assert.deepEqual(one(repro, n => n.tag === 'pre').text.split('\\n'), [
  'git checkout abcdef123456',
  'cd programming_examples/ml/yolo26n && python yolo26n.py --images $IRON_EXAMPLE_WEIGHTS_DIR/yolo26n/coco128/images/train2017 --n 128',
  '# every run, as the nightly runs them:',
  'python utils/component_checks/yolo26n_hw.py --out perf.json --meta meta.json',
]);
""")


def test_llama_and_yolo26n_chart_their_trends_beside_the_compile_tabs(page):
    page(COMPONENT_RUNS + LLAMA_NIGHTS + YOLO_NIGHTS + """
// The latest llama night also ran generate.py and profiled its compile's stages.
const staged = { decode_tok_s: { value: 4.2, unit: 'tok/s' }, compile_place_s: { value: 1, unit: 's' },
                 compile_route_s: { value: 0.03, unit: 's' }, compile_package_s: { value: 0.07, unit: 's' } };
const llamaData = collect(historiesOf('llama-chain-hw', [llamaBefore, { ...llamaAfter, rows: { [LLAMA]: { ...llamaAfter.rows[LLAMA], ...staged } } }]));
// A tab with no history is left out; generate's rate is a second line beside the timed decode's.
const box = trendBox(llamaData, llama.trend, '');
assert.deepEqual(box.children[0].children[0].children.map(b => b.textContent), ['Decode', 'End to end', 'Compile time']);
const rate = trendConfig(llamaData, llama.trend.find(t => t.id === 'tok_s'));
assert.deepEqual(rate.data.datasets.map(d => [d.label, d.data]),
                 [['generate.py, end to end (median of 5)', [null, 4.2]], ['the NPU\\'s dispatches alone, at 64 context', [4.7, 4.37]]]);
// One line needs no legend; two do.
assert.equal(trendConfig(llamaData, { id: 'tok_s', axis: 'x', lines: [['', ['tok_s']]] }).options.plugins.legend.display, false);
assert.equal(rate.options.plugins.legend.display, true);
// The compile's stack starts with the night that profiled its stages; the
// night before has its wall time alone.
const stages = stackConfig(llamaData, { ...COMPILE_TABS[0], stack: COMPILE_TABS[0].split, missingIsZero: true });
assert.deepEqual(stages.data.datasets.map(d => [d.label, d.data]), [
  ['place', [null, 1]], ['lower', [null, 0]], ['waiting on kernels', [null, 0]], ['codegen and link', [null, 0]],
  ["route, package, aiecc's own", [null, 0.1]], ['outside aiecc', [null, 0]],
  ['cold (nothing cached)', [60.5, 60.5]], ['warm (kernel objects kept)', [1.2, 1.2]]]);
// YOLO26n: where an image's time goes, batch 1 or batch 8, then mAP and the compile.
// The first night predates the empty-dispatch baseline: no stack, only its end-to-end line.
const split = (npu, dispatch, host, map50) => ({ ...(npu ? { npu_us_per_image_b1: { value: npu, unit: 'us' }, npu_us_per_image_b8: { value: 2850, unit: 'us' } } : {}),
                                          dispatch_us_per_image_b1: { value: dispatch, unit: 'us' }, dispatch_us_per_image_b8: { value: 2900, unit: 'us' },
                                          host_overhead_us_per_image_b1: { value: host, unit: 'us' }, host_overhead_us_per_image_b8: { value: 100, unit: 'us' },
                                          ...(map50 !== undefined ? { map50_coco128: { value: map50, unit: 'mAP' } } : {}) });
const withSplit = (record, npu, dispatch, host, map50) => ({ ...record, rows: { [YOLO]: { ...record.rows[YOLO], ...split(npu, dispatch, host, map50) } } });
// map50_coco128 is itself a newly added metric: its history, like the NPU
// time's, starts partway through.
const yoloData = collect(historiesOf('yolo26n-hw', [withSplit(yoloBefore, null, 4400, 600), withSplit(yoloAfter, 4310, 4400, 1000, 0.5859)]));
// The same tabs as MobileNet's: NPU time, end to end, where the time goes, accuracy, compile.
assert.deepEqual(yolo.trend.map(t => t.name), ['NPU time', 'End to end', 'Where the time goes', 'Accuracy', 'Compile time', 'Compile memory']);
const yoloStack = yolo.trend.find(t => t.id === 'yolo_time_stack');
// A variant per batch of the sweep; this fixture's nights timed 1 and 8.
assert.deepEqual(yoloStack.variants.map(([name]) => name), ['batch 1', 'batch 2', 'batch 4', 'batch 8', 'batch 16']);
const stackAt = name => { const [, v] = yoloStack.variants.find(([n]) => n === name); return stackConfig(yoloData, { ...yoloStack, ...v, variants: null }); };
const [b1, b8] = [stackAt('batch 1'), stackAt('batch 8')];
assert.deepEqual(b1.data.datasets.map(d => [d.label, d.data]), [
  ['NPU work', [null, 4310]], ['launch overhead', [null, 90]], ['host: Python and buffer syncs', [null, 1000]], ['end to end', [5000, 5400]]]);
assert.deepEqual(b8.data.datasets.map(d => [d.label, d.data]), [
  ['NPU work', [null, 2850]], ['launch overhead', [null, 50]], ['host: Python and buffer syncs', [null, 100]], ['end to end', [3000, 3000]]]);
assert.equal(b1.options.scales.y.title.text, 'Time per image (µs) · ↓ better');
// The box offers the two batches; a link to mAP opens its tab.
const yoloBox = trendBox(yoloData, yolo.trend, '');
// NPU time: batch 1 and batch 8 as two lines, no picker.
const shownConfig = yoloBox.draw('npu_us_per_image_b1');
assert.deepEqual(shownConfig.data.datasets.filter(d => !d.band).map(d => [d.label, d.data.at(-1)]), [['batch 1', 4310], ['batch 8', 2850]]);
yoloBox.draw('yolo_time_stack');
// Only the batches this history timed are offered.
assert.equal(yoloBox.children[1].text, 'Show batch 1batch 8');
assert.equal(trendBox(yoloData, yolo.trend, 'map50_coco128').metric(), 'map50_coco128');
// mAP@0.5: the metric's own nights (only the second one published it, as
// every tab's x-axis already does), plus a flat dashed reference line over
// the same nights, with its own label.
const map = trendConfig(yoloData, yolo.trend.find(t => t.id === 'map50_coco128'));
assert.deepEqual(map.data.datasets.filter(d => !d.band).map(d => [d.label, d.data]),
                 [['NPU (int8)', [0.5859]], ['fp32 ONNX (offline, same 128 images)', [0.6365]]]);
const ref = map.data.datasets.find(d => d.label === 'fp32 ONNX (offline, same 128 images)');
assert.deepEqual(ref.borderDash, [6, 4]);
assert.equal(ref.pointRadius, 0);
assert.equal(map.options.scales.y.title.text, 'mAP@0.5 on coco128 · ↑ better');
""")


def test_a_refused_check_says_why_and_shows_the_last_numbers(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
const refused = "power mode is default, required 'turbo'";
const index = { target: 'sa-placer-hw', runs: [
  night('6', '2026-10-01T06:30:00Z', 'turbo'),
  night('7', '2026-10-02T06:30:00Z', 'default', { sane: null, published: false, refused }),
] };
const card = componentBand(hw, index, hwBefore, NOW);
assert.equal(card.className, 'verdict bad');
assert.equal(one(card, n => n.className === 'headline').text, `The run refused to measure: ${refused}.`);
// The refusal once, as the headline; the power mode as a warning.
assert.deepEqual(all(card, n => /^note /.test(n.className || '')).map(n => n.className), ['note warn']);
assert.ok(card.text.includes('Numbers from the run of'));
assert.ok(card.table.text.includes('395.7 µs'));
""")


def test_a_check_s_histories_are_read_under_their_cases(page):
    page(COMPONENT_RUNS + """
const fetched = [];
const runs = [{ id: '7', date: '2026-10-02T06:30:00Z', commit, pmode: null, provenance: { fixtures: 'abc' } }];
global.fetch = async url => {
  fetched.push(url);
  const check = url.split('/')[2];
  return { ok: true, json: async () => ({ schema: 1, target: check, metric: 'failed_seeds', unit: 'seeds', runs,
    series: { mobilenet: { values: [0] }, test_sa_effort: { values: [0] } } }) };
};
(async () => {
  // A check's own histories, every metric its index lists, keep its cases.
  const own = await checkHistories({ check: sweep, index: { schema: 1, metrics: ['failed_seeds'] } });
  assert.deepEqual(fetched, ['../component-checks/sa-placer/history/failed_seeds.json']);
  assert.deepEqual(own.series.map(s => s.key).sort(), ['sa-placer|mobilenet|failed_seeds', 'sa-placer|test_sa_effort|failed_seeds']);
  // A count that never left zero hides its tab: nothing else was read, so no tab is left.
  assert.equal(trendBox(own, sweep.trend, '').text, 'No history published yet.');
  own.series[0].byMode.get('unknown')[0].row.value = 2;
  assert.equal(trendBox(own, sweep.trend, '').children[0].children[0].children.map(b => b.textContent).join(), 'Failed seeds');
})().catch(e => { console.error(e); process.exit(1); });
""")


def test_one_formatter_reads_every_unit_the_checks_emit(page):
    page("""
// Every unit a check publishes, as tick, tooltip, card and table all show it.
assert.equal(formatNumber(295.7, 'us'), '295.7 µs');
assert.equal(formatNumber(3 * 1048576, 'bytes'), '3.0 MiB');
assert.equal(formatNumber(3072, 'bytes'), '3.0 KiB');
// ru_maxrss / 1024 is MiB, whatever the check names it.
assert.equal(formatNumber(850, 'MB'), '850 MiB');
// Both ratio spellings read as one.
assert.equal(formatNumber(1.3, 'x'), '1.3×');
assert.equal(formatNumber(1.0125, '×'), '1.013×');
assert.equal(formatNumber(12345.678, 'cost'), '12,345.7');
assert.equal(formatNumber(1, 'seeds'), '1');
assert.equal(formatNumber(0.2434, 'pairs'), '0.243');
assert.equal(formatNumber(4.37, 'tok/s'), '4.37 tok/s');
assert.equal(formatNumber(49.2, '%'), '49.2%');
assert.equal(formatNumber(1100.4, 'cycles'), '1,100');
assert.equal(typeof formatValue, 'undefined');
// One date format; the year once a chart's dates span two.
assert.equal(shortDate(Date.parse('2026-10-20T06:00:00Z')), '20 Oct');
assert.equal(shortDate(Date.parse('2026-10-20T06:00:00Z'), true), '20 Oct 2026');
assert.equal(formatWhen(Date.parse('2026-10-20T06:00:00Z')), '20 Oct, 06:00 UTC');
assert.equal(formatWhen(Date.parse('2026-10-20T06:00:00Z'), true), '20 Oct 2026, 06:00 UTC');
assert.ok(spansYears([Date.parse('2025-12-30T00:00:00Z'), Date.parse('2026-01-02T00:00:00Z')]));
const across = fromRecords({ npu1: [rec('1', Date.parse('2025-12-30T06:00:00Z'), 'turbo', sm(100)), rec('2', Date.parse('2026-01-02T06:00:00Z'), 'turbo', sm(101))] });
assert.deepEqual(kernelTrendConfig(across, 'npu1', 'softmax', 'cycles').data.labels, ['30 Dec 2025', '2 Jan 2026']);
""")


def test_the_placement_band_shows_each_seeds_placement_and_runs(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
assert.equal(hwPlacement.section, 'placement');
assert.equal(hw.section, 'performance');
const card = componentBand(hwPlacement, hwIndex, hwAfter, NOW, hwBefore);
assert.equal(card.className, 'verdict ok');
// No news is good news: a passing card says nothing, its table has the costs.
assert.equal(all(card, n => n.className === 'headline').length, 0);
// Only the section's metrics count as moves: seed 3's per-image regression is Performance's.
assert.equal(one(card, n => n.className === 'moves').text, 'Since 1 Oct:nothing moved past its threshold.');
// A row per seed and effort: which placement, its cost, how fast it runs
// on the NPU, its compile and its runs: how to judge a seed. Performance
// has the design's own placement at every batch, not the seeds.
assert.deepEqual(tableOf(card.table), [
  ['', 'Placement', 'SA cost', 'NPU, b1', 'NPU, b16', 'Compile, cold', 'Compile, warm', 'Runs'],
  ["seed 3, default effort (the design's)", '0123456789ab', '303', '248.7 µs', '102 µs +7.3%', '40 s', '4.6 s', '2 of 2 passed'],
  ['seed 3, full effort', '0123456789ab', '403', '248.7 µs', '117.1 µs', '55 s', '25 s', '2 of 2 passed'],
  ['seed 7, default effort', '0123456789ab', '307', '248.7 µs', '96.1 µs', '40 s', '4.6 s', '2 of 2 passed'],
  ['seed 7, full effort', '0123456789ab', '407', '248.7 µs', '117.1 µs', '55 s', '25 s', '2 of 2 passed'],
]);
// No seed rows on Performance.
assert.deepEqual(tableOf(componentBand(hw, hwIndex, hwAfter, NOW, hwBefore).table).map(r => r[0]), ['', 'batch 1', 'batch 16']);
// A failed run is named on its seed, red.
const failedAt = hwRecord('7', hwAfter.date, 104.9);
failedAt.detail.find(d => d.seed === 7 && d.batch === 16 && d.effort_label === 'default').passed = false;
failedAt.failed = ['mobilenet/seed=7/batch=16/effort=default'];
const bad = componentBand(hwPlacement, hwIndex, failedAt, NOW, hwBefore);
// The failed seed's row: its runs say where, red.
assert.equal(tableOf(bad.table)[3].at(-1), 'failed at b16');
assert.equal(all(bad.table, n => n.className === 'vals worse').length, 2);
""")


MODEL_NIGHTS = """
// routerRecord and allocatorRecord build one night's record exactly as
// model_based.py writes it, for the three tests below (a failed property, a
// failed lit test, a failed allocator property).
const router = BANDS.find(b => b.id === 'router'), allocator = BANDS.find(b => b.id === 'allocator');
const cell = (value, unit) => ({ value, unit });
const routerRecord = (id, date, npu2, extra) => ({ id, date, pmode: null, commit: { ...commit }, provenance: {}, rows: {
  npu1: { failed_properties: cell(0, 'properties'), unrouted: cell(0, 'routings'), illegal: cell(0, 'routings'),
          hop_ratio: cell(1.012, '×'), amsels_per_design: cell(14.87, 'amsels'), ms_per_design: cell(10.0, 'ms'),
          failed_mutations: cell(2, 'cases'), survived_corruptions: cell(1, 'corruptions') },
  npu2: { failed_properties: cell(npu2, 'properties'), unrouted: cell(0, 'routings'), illegal: cell(npu2, 'routings'),
          hop_ratio: cell(1.015, '×'), amsels_per_design: cell(14.96, 'amsels'), ms_per_design: cell(13.1, 'ms'),
          failed_mutations: cell(0, 'cases'), survived_corruptions: cell(0, 'corruptions') },
}, detail: [
  { device: 'npu1', run: 'properties', first_seed: 20370000, seeds: 2000, failed: [] },
  { device: 'npu1', run: 'hops-on', first_seed: 20370000, seeds: 400, killed: 1000, survived: 0, failures: 0 },
  { device: 'npu1', run: 'hops-off', first_seed: 20370000, seeds: 400, killed: 999, survived: 1, failures: 2 },
  { device: 'npu2', run: 'properties', first_seed: 20370000, seeds: 2000, failed: npu2 ? ['legality'] : [] },
], ...(extra || {}) });
const allocatorRecord = (id, date, sharing, extra) => ({ id, date, pmode: null, commit: { ...commit }, provenance: {}, rows: {
  allocator: { failed_properties: cell(0, 'properties'), unsolved: cell(0, 'designs'), illegal: cell(0, 'designs'),
               avoidable_crossings: cell(0, 'crossings'), bank_pair_sharing: cell(sharing, 'pairs'), wall_s: cell(103, 's') },
}, detail: ['metric-validation', 'legality', 'bank-sharing'].map(name => ({ name, ok: true, detail: '' })), ...(extra || {}) });
"""


def test_a_failed_router_night_names_each_failure_and_how_to_run_it_again(page):
    page(COMPONENT_RUNS + MODEL_NIGHTS + """
assert.equal(router.section, 'placement');
const failed = ['npu2/properties/seed=20370000+2000/legality', 'npu1/hops-off/seed=20370012+1/mutator fanout illegal',
                'npu1/hops-off/seed=20370040+1/kill amsel survived', 'npu1-hops-on: no report'];
const latest = routerRecord('7', '2026-10-02T03:17:00Z', 1, { failed });
const index = { target: 'router', runs: [night('7', latest.date, null, { failed })] };
const card = componentBand(router, index, latest, NOW);
assert.equal(card.className, 'verdict bad');
assert.equal(one(card, n => n.className === 'headline').text,
             '4 checks failed: npu2 legality, npu1 hops off mutator fanout illegal (seed 20370012), '
             + 'npu1 hops off kill amsel survived (seed 20370040), npu1-hops-on: no report.');
const repro = one(card, n => n.className === 'repro');
assert.ok(card.children.includes(repro));
assert.ok(repro.text.includes('artifact component-checks-router-7'));
assert.deepEqual(one(repro, n => n.tag === 'pre').text.split('\\n'), [
  'git checkout abcdef123456',
  'python test/create-packet-flows/nightly/router_properties.py --device npu2 --seeds 2000 --first-seed 20370000',
  'python utils/router_testing/router_mutation.py --device npu1 --hops-off --seeds 1 --first-seed 20370012 --only fanout -v',
  'python utils/router_testing/router_mutation.py --device npu1 --hops-off --seeds 1 --first-seed 20370040 --only kill -v',
  '# every check, as the nightly runs them:',
  ...router.whole.split('\\n'),
]);
// A row per device: its seeds, the five correctness counts (nonzero red),
// then what the routes cost; the verifier's catch rate on hover.
assert.deepEqual(tableOf(card.table), [
  ['', 'Seeds', 'Properties failed', 'Unrouted', 'Illegal', 'Mutations wrong', 'Corruptions missed', 'Hops ÷ known', 'Msels / design', 'Time / design'],
  ['npu1', '20,370,000 + 2,000', '0', '0', '0', '2', '1', '1.012×', '14.9', '10 ms'],
  ['npu2', '20,370,000 + 2,000', '1', '0', '1', '0', '0', '1.015×', '15', '13.1 ms'],
]);
assert.equal(all(card.table, n => n.className === 'vals worse').length, 4);
assert.equal(all(card.table, n => n.title === 'the verifier caught 1999 of 2000 corruptions').length, 1);
// Run by hand with no failure: how to run the whole check.
const quiet = componentBand(router, { target: 'router', runs: [night('7', latest.date, null)] }, routerRecord('7', latest.date, 0), NOW);
assert.equal(all(quiet, n => n.className === 'headline').length, 0);
assert.equal(one(quiet, n => n.tag === 'pre').text.split('\\n')[1],
             'python test/create-packet-flows/nightly/router_properties.py --device npu2 --seeds 2000 --first-seed N');
""")


def test_a_failed_nightly_lit_test_names_the_test_and_its_lit_command(page):
    page(COMPONENT_RUNS + MODEL_NIGHTS + """
const failed = ['lit/create-packet-flows/nightly/hw_verdicts.py'];
const latest = routerRecord('8', '2026-10-03T03:17:00Z', 0, { failed });
const index = { target: 'router', runs: [night('8', latest.date, null, { failed })] };
const card = componentBand(router, index, latest, NOW);
assert.equal(card.className, 'verdict bad');
assert.equal(one(card, n => n.className === 'headline').text,
             '1 check failed: lit create-packet-flows/nightly/hw_verdicts.py.');
assert.deepEqual(one(one(card, n => n.className === 'repro'), n => n.tag === 'pre').text.split('\\n').slice(0, 2), [
  'git checkout abcdef123456',
  'lit -sv -D router_nightly=1 build/test/create-packet-flows/nightly/hw_verdicts.py',
]);
""")


def test_the_allocator_band_names_its_failed_properties_under_placement(page):
    page(COMPONENT_RUNS + MODEL_NIGHTS + """
assert.equal(allocator.section, 'placement');
const before = allocatorRecord('6', '2026-10-01T03:17:00Z', 0.2434);
const failed = ['bank-sharing', 'legality'];
const after = allocatorRecord('7', '2026-10-02T03:17:00Z', 0.2534, { failed });
after.detail[1].ok = after.detail[2].ok = false;
const index = { target: 'allocator', runs: [night('6', before.date, null), night('7', after.date, null, { failed })] };
const card = componentBand(allocator, index, after, NOW, before);
assert.equal(one(card, n => n.className === 'headline').text, '2 properties failed: bank-sharing, legality.');
// Sharing moved 4.1%, past its threshold.
assert.equal(one(card, n => n.className === 'moves').text, 'Since 1 Oct:allocator bank sharing (0–1) +4.1%');
// One repro line however many properties failed: the script has no per-property switch.
assert.deepEqual(one(card, n => n.tag === 'pre').text.split('\\n'), [
  'git checkout abcdef123456',
  "python test/assign-buffer-addresses/allocator_properties.py   # check-aie's smaller corpus",
  '# every property, as the nightly runs them:',
  allocator.whole,
]);
// Its figures a line each; the counts are correctness, red past zero.
assert.deepEqual(tableOf(card.table), [
  ['Properties held', '1 of 3'], ['Unplaced', '0'], ['Illegal', '0'], ['Avoidable crossings', '0'],
  ['Bank sharing (0–1)', '0.253 +4.1%'], ['Wall time', '103 s']]);
assert.equal(all(card.table, n => n.className === 'vals worse').length, 1);
const quiet = componentBand(allocator, { target: 'allocator', runs: [night('6', before.date, null)] }, before, NOW);
assert.equal(all(quiet, n => n.className === 'headline').length, 0);
assert.equal(plural(2, 'property'), '2 properties');
assert.equal(plural(2, 'day'), '2 days');
""")


def test_router_and_allocator_trends_chart_what_their_tables_show(page):
    page(COMPONENT_RUNS + MODEL_NIGHTS + """
// Every metric a model-based band judges is in its table or a trend tab.
for (const band of [router, allocator]) {
  const charted = new Set(band.trend.flatMap(tabMetrics));
  for (const m of band.metrics) assert.ok(charted.has(m), `${band.id} ${m}`);
}
// The router's lengths are a line per device, on one axis.
const data = collect(historiesOf('router', [routerRecord('6', '2026-10-01T03:17:00Z', 0), routerRecord('7', '2026-10-02T03:17:00Z', 0)]));
const hops = trendConfig(data, router.trend[0]);
assert.deepEqual(hops.data.datasets.filter(d => !d.band).map(d => [d.label, d.data]), [['npu1', [1.012, 1.012]], ['npu2', [1.015, 1.015]]]);
// Its failures are a line per device, the counts summed, and only shown once one is above zero.
const failures = trendConfig(data, router.trend[3]);
assert.deepEqual(failures.data.datasets.filter(d => !d.band).map(d => [d.label, d.data]), [['npu1', [3, 3]], ['npu2', [0, 0]]]);
assert.deepEqual(trendBox(data, router.trend, '').children[0].children[0].children.map(b => b.textContent),
                 ['Route length', 'Arbiter use', 'Time per design', 'Failures']);
const clean = collect(historiesOf('router', [routerRecord('6', '2026-10-01T03:17:00Z', 0)].map(r => {
  r.rows.npu1.failed_mutations = cell(0, 'cases'); r.rows.npu1.survived_corruptions = cell(0, 'corruptions'); return r; })));
assert.ok(!trendBox(clean, router.trend, '').children[0].children[0].children.some(b => b.textContent === 'Failures'));
""")


def test_the_overview_has_a_tile_per_section_and_says_when_one_is_late(page):
    page(MOVED + RUNS + COMPONENT_RUNS + HW_NIGHTS + """
const nightsNow = renderDashboard(['npu1'], db, new Map([['npu1', catalogue]]), new Map([['npu1', index]]), 3600 * 1000);
const loaded = [
  { check: hwPlacement, index: hwIndex, latest: hwAfter, previous: hwBefore },
  { check: hw, index: hwIndex, latest: hwAfter, previous: hwBefore },
];
const tiles = overviewTiles(nightsNow, loaded, NOW);
// In tab order: CI health, Kernels, then the sections, placement
// (now the compiler checks' tab, placement and routing together) last.
assert.deepEqual(tiles.map(t => t.href), ['#view=ci', '#view=night', '#view=performance', '#view=placement']);
// A row each: the name, one status word, one line, how long ago.
assert.deepEqual(tiles.map(t => t.children.map(c => c.className)[0]), ['area-name', 'area-name', 'area-name', 'area-name']);
assert.deepEqual(tiles.map(t => t.children[0].text), ['CI health', 'Kernels', 'Performance', 'Compiler checks']);
const [ciTile, kernels, perf, placement] = tiles;
// GitHub not asked yet.
assert.equal(ciTile.children[1].text, 'Loading…');
// A slower night is a warning on the overview too, not "All passing".
assert.equal(perf.level, 'warn');
assert.deepEqual(perf.children.slice(1, 3).map(c => c.text), ['Watch', 'MobileNet: 2 measures got worse vs 1 Oct.']);
// The whole of each line is in the row's tooltip.
assert.ok(perf.title.includes('MobileNet: 2 measures got worse vs 1 Oct.'));
// A section with one loaded, healthy band: its own headline, no prefix
// (the overview rolls up headlines, not a band's quieter detail sentence).
assert.equal(placement.level, 'ok');
// Healthy: the state word alone, no sentence.
assert.equal(placement.children[2].text, '');
// Two days without a run: late and red, the line says so first.
const late = overviewTiles(null, loaded, NOW + 2 * 86400 * 1000);
assert.equal(late[2].level, 'bad');
assert.equal(late[2].children[2].text, 'MobileNet: No run for 2 days 1 h. 2 measures got worse vs 1 Oct.');
assert.equal(late[1].children[1].text, 'Loading…');
// The band says it above its verdict.
const lateBand = componentBand(hw, hwIndex, hwAfter, NOW + 2 * 86400 * 1000, hwBefore);
assert.equal(one(lateBand, n => n.className === 'stale-line').text,
             'No run for 2 days 1 h: the numbers below are from 2 Oct, 06:30 UTC.');
assert.equal(one(lateBand, n => n.className === 'headline').text, '2 measures got worse vs 1 Oct.');
""")


def test_a_healthy_tile_says_nothing_but_its_state(page):
    page("""
// No news is good news: a healthy area shows its state word alone, a
// problem anywhere names only the problems.
const lines = [{ level: 'ok', who: 'Seed sweep', text: 'Every seed placed.' },
               { level: 'ok', who: 'Router', text: 'Every property held and every mutation was answered right.' }];
const tile = sectionTile('placement', 'Compiler checks', 'about', lines, '2 h ago', '');
assert.equal(tile.level, 'ok');
assert.deepEqual([tile.children[1].text, tile.children[2].text], ['All passing', '']);
// Each band is still named on hover.
assert.ok(tile.title.includes('Router: Every property held'));
const mixed = sectionTile('placement', 'Compiler checks', 'about',
  [{ level: 'warn', who: 'Buffer allocator', text: 'Bank sharing moved.' }, lines[0]], '2 h ago', '');
assert.equal(mixed.children[2].text, 'Buffer allocator: Bank sharing moved.');
""")

def test_performance_trends_chart_each_batch_as_the_mean_over_seeds(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
const read = collect(historiesOf('sa-placer-hw', [hwBefore, hwAfter]));
const data = seedMeans(read);
// A series per batch per effort, not per seed/batch/effort; scoped here to
// the design's real default effort, where the fixture's b16 math lives.
const perImage = data.series.filter(s => s.metric === 'us_per_image' && s.kase.endsWith('/effort=default'));
assert.deepEqual(perImage.map(s => s.kase).sort(), ['mobilenet/batch=1/effort=default', 'mobilenet/batch=16/effort=default']);
const b16 = perImage.find(s => s.kase === 'mobilenet/batch=16/effort=default');
// Seeds 3 and 7 at batch 16: (98 + 99) / 2, then (104.9 + 99) / 2; one point per run.
assert.deepEqual(b16.byMode.get('turbo').map(p => p.row.value), [98.5, 101.95]);
assert.equal(b16.byMode.get('turbo')[1].row.range, 'seeds 99 to 104.9, n=2');
// The band runs from the fastest seed to the slowest.
assert.deepEqual(spreadOf(101.95, b16.byMode.get('turbo')[1].row), [99, 104.9]);
// A compile, per seed and effort, means over the seeds the same way, as one
// series for every batch.
const compiled = data.series.find(s => s.kase === 'mobilenet/effort=default' && s.metric === 'compile_s');
assert.deepEqual(compiled.byMode.get('turbo').map(p => p.row.value), [40, 40]);
// Series that are not per seed pass through.
assert.ok(data.series.some(s => s.kase === 'mobilenet' && s.metric === 'failed_seeds'));
// The compile tab draws that series' wall times, cold beside warm.
const trend = stackConfig(read, { ...COMPILE_TABS[0], stack: COMPILE_TABS[0].split, missingIsZero: true });
assert.deepEqual(trend.data.datasets.filter(d => !d.part).map(d => [d.label, d.data]), [['cold (nothing cached)', [40, 40]], ['warm (kernel objects kept)', [4.6, 4.6]]]);
// The time per image at batch 16: each part the mean over the seeds, adding up to the mean end to end.
const b16Stack = stackConfig(read, hw.trend.find(t => t.stack), { key: 16 });
assert.deepEqual(b16Stack.data.datasets.map(d => [d.label, d.data]), [
  ['NPU work', [95.6, 99.05]], ['launch overhead', [2.9, 2.9]], ['host: Python and buffer syncs', [100, 100]], ['end to end', [198.5, 201.95]]]);
""")


def test_mobilenet_charts_the_full_effort_when_that_is_all_published(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
// Every run before efforts were recorded publishes as full effort only.
const fullOnly = rec => ({ ...rec, rows: Object.fromEntries(Object.entries(rec.rows).filter(([k]) => !k.endsWith('/effort=default'))),
                           detail: rec.detail.filter(d => d.effort_label === 'full') });
const before = fullOnly(hwBefore), after = fullOnly(hwAfter);
assert.equal(chartedEffort(Object.keys(after.rows)), 'full');
assert.equal(chartedEffort(Object.keys(hwAfter.rows)), 'default');
const summary = hardwareSummary(after.rows);
assert.equal(summary, " NPU time is 248.7 µs (2,527.17 fps) per image at batch 1, 117.1 µs (4,545.45 fps) at batch 16 (full effort).");
// The trend chart draws the full effort's times.
const trend = stackConfig(collect(historiesOf('sa-placer-hw', [before, after])), hw.trend.find(t => t.stack));
assert.deepEqual(trend.data.keys, [1, 16]);
assert.deepEqual(trend.data.datasets.find(d => d.label === 'end to end').data, [395.7, 395.7]);
""")


def test_a_static_design_s_history_charts_as_its_own_line(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
// A night from before the effort sweep: a static design per batch, its
// times still keyed by seed and batch (publish.py's upgrade keeps them so).
const staticRows = { mobilenet: { failed_seeds: { value: 0, unit: 'seeds' } } };
for (const [seed, batch, us] of [[3, 1, 300], [7, 1, 310], [3, 16, 100], [7, 16, 104]]) {
  staticRows[`mobilenet/seed=${seed}/batch=${batch}`] = { us_per_image: { value: us, unit: 'us' } };
}
const old = { ...hwBefore, id: '5', date: '2026-09-30T06:30:00Z', rows: staticRows, detail: [] };
const read = collect(historiesOf('sa-placer-hw', [old, hwBefore, hwAfter]));
const statics = seedMeans(read).series.filter(s => s.kase.endsWith('/dispatch=static'));
assert.deepEqual(statics.map(s => [s.kase, s.byMode.get('turbo').map(p => p.row.value)]).sort(),
                 [['mobilenet/batch=1/dispatch=static', [305]], ['mobilenet/batch=16/dispatch=static', [102]]]);
// Another design: never stacked with the current one's times, nor joined to them.
const trend = stackConfig(read, hw.trend.find(t => t.stack));
assert.deepEqual(trend.data.labels, ['1 Oct', '2 Oct']);
// Only the static history: nothing to stack.
const onlyOld = stackConfig(collect(historiesOf('sa-placer-hw', [old])), hw.trend.find(t => t.stack));
assert.deepEqual([onlyOld.data.labels, onlyOld.data.datasets.filter(d => !d.part).length], [[], 0]);
""")


def test_every_power_of_two_batch_is_offered(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
const batches = [1, 2, 4, 8, 16, 32, 64];
const rows = { mobilenet: { failed_seeds: { value: 0, unit: 'seeds' } } };
for (const b of batches) rows[`mobilenet/seed=3/batch=${b}/effort=default`] = { us_per_image: { value: 90 + 300 / b, unit: 'us' } };
const powers = { ...hwAfter, rows, detail: batches.map(batch => ({ seed: 3, batch, effort_label: 'default', passed: true, placement: '0123456789ab' })) };
// The table has every batch, a row each; the seeds' table the smallest and the largest.
const card = componentBand(hw, hwIndex, powers, NOW, hwBefore);
assert.deepEqual(tableOf(card.table).slice(1).map(r => r[0]), batches.map(b => `batch ${b}`));
assert.deepEqual(tableOf(componentBand(hwPlacement, hwIndex, powers, NOW, hwBefore).table)[0].slice(3, 5), ['NPU, b1', 'NPU, b64']);
// The trend offers every batch to choose from, the smallest first.
const powersOn = id => ({ ...powers, id, date: id === '6' ? hwBefore.date : hwAfter.date });
const read = collect(historiesOf('sa-placer-hw', [powersOn('6'), powersOn('7')]));
assert.deepEqual(tabKeys(read, hw.trend.find(t => t.stack)), batches);
""")


CI_RUNS = """
const H = 3600 * 1000;
const NOW = Date.parse('2026-10-06T12:00:00Z');
const at = h => new Date(NOW - h * H).toISOString();
const run = (id, hoursAgo, status, conclusion) => ({ id, url: `https://example.com/runs/${id}`, status, conclusion,
  created_at: at(hoursAgo), run_started_at: at(hoursAgo), head_sha: 'abcdef1234567' });
const job = (name, hoursAgo, status, conclusion) => ({ name, url: `https://example.com/jobs/${name}`, status, conclusion, created_at: at(hoursAgo) });
const nightly = { group: 'On the NPUs', file: 'nightlyKernelChecks.yml', title: 'Kernel checks', every: 24, legs: true, why: 'w' };
// A critical workflow as ci_health.py publishes it.
const crit = (extra) => ({ file: 'buildAndTestRyzenAI.yml', title: 'Linux hardware tests', runs: 40,
  main: { passed: 9, failed: 1, rate: 0.9 }, latest_main: run(30, 3, 'completed', 'success'),
  pull_requests: { passed: 20, failed: 10, rate: 0.667 }, merge_queue: { passed: 8, failed: 0, rate: 1 },
  flaky: { commits: 30, flaky: 2, rate: 0.067 }, minutes_median: 84, minutes_p90: 130, queued_median: 3.5,
  slowest_jobs: [{ name: 'build-and-test (aie2p-8col)', n: 6, minutes_median: 70, minutes_p90: 95, queued_median: 12, failed: 1 }],
  jobs_sampled: 6, ...(extra || {}) });
const status = (extra) => ({ schema: 1, generated_at: new Date(NOW - 0.5 * H).toISOString(), window_days: 14,
  critical: [crit()], merge_time: { merged: 31, hours_median: 26.4, hours_p90: 190, open: 42, open_drafts: 9, open_over_30_days: 11 },
  workflows: [{ ...nightly, runs: [run(1, 6, 'completed', 'success')], jobs: [] }], ...(extra || {}) });
const loadedOf = raw => {
  const st = readable(raw);
  return { status: st, newer: Boolean(raw) && !st, results: ((st && st.workflows) || []).map(w => ({ wf: w, runs: w.runs, jobs: w.jobs, error: w.error })),
           critical: (st && st.critical) || [], merge: (st && st.merge_time) || null, history: null };
};
"""


def test_ci_runs_read_as_passed_failed_running_or_waiting_for_a_runner(page):
    page(CI_RUNS + """
assert.deepEqual(ciRunLevel(run(1, 2, 'completed', 'success'), NOW), { level: 'ok', text: 'passed' });
assert.deepEqual(ciRunLevel(run(1, 2, 'completed', 'failure'), NOW), { level: 'bad', text: 'failed' });
// Cancelled is no result: by hand, or by a newer run.
assert.deepEqual(ciRunLevel(run(1, 2, 'completed', 'cancelled'), NOW), { level: 'none', text: 'cancelled' });
assert.deepEqual(ciRunLevel(run(1, 2, 'in_progress', null), NOW), { level: 'running', text: 'running' });
assert.deepEqual(ciRunLevel(run(1, 2, 'queued', null), NOW), { level: 'running', text: 'queued' });
// Queued past six hours: no runner with its labels is online.
assert.deepEqual(ciRunLevel(run(1, 30, 'queued', null), NOW), { level: 'warn', text: 'waiting for a runner for 1 day 6 h' });
// A nightly still going is judged by the run before it.
let v = ciVerdict(nightly, [run(2, 1, 'in_progress', null), run(1, 25, 'completed', 'failure')], [], NOW);
assert.deepEqual([v.level, v.headline], ['bad', 'Running now; the run before failed.']);
// A leg that never found a runner is named, and is red even when the run
// is still going: the nightly will not run until a machine is back.
v = ciVerdict(nightly, [run(2, 30, 'in_progress', null), run(1, 54, 'completed', 'success')],
  [job('Kernel checks (aie2-4col)', 30, 'completed', 'success'), job('Kernel checks (aie2p-8col)', 30, 'queued', null)], NOW);
assert.deepEqual([v.level, v.kind], ['bad', 'runner']);
assert.deepEqual(v.legs.map(l => [l.name, l.text]), [['Kernel checks (aie2p-8col)', 'waiting for a runner for 1 day 6 h']]);
// A job queued a short while is not listed: it is about to start.
assert.deepEqual(ciLegs([job('soon', 1, 'queued', null)], NOW), []);
// Late: no scheduled run for 1.5 of its period.
v = ciVerdict(nightly, [run(1, 40, 'completed', 'success')], [], NOW);
assert.deepEqual([v.level, v.kind, v.late, v.headline], ['bad', 'stopped', true, 'No scheduled run for 1 day 16 h. Its last passed.']);
// A cancelled run is judged by the one before it.
v = ciVerdict(nightly, [run(2, 2, 'completed', 'cancelled'), run(1, 26, 'completed', 'success')], [], NOW);
assert.deepEqual([v.level, v.headline], ['ok', 'Cancelled; the run before passed.']);
// A heavy build still running past 1.5 periods is not late until it has run a day.
const fourHourly = { ...nightly, every: 4 };
assert.equal(ciVerdict(fourHourly, [run(2, 7, 'in_progress', null), run(1, 11, 'completed', 'success')], [], NOW).late, false);
assert.equal(ciVerdict(fourHourly, [run(2, 25, 'in_progress', null), run(1, 29, 'completed', 'success')], [], NOW).late, true);
// Nothing on record.
assert.equal(ciVerdict(nightly, [], [], NOW).level, 'none');
""")


def test_nightly_jobs_are_a_row_each_with_their_recent_runs(page):
    page(CI_RUNS + """
const runs = [run(3, 6, 'completed', 'failure'), run(2, 30, 'completed', 'success'), run(1, 54, 'completed', 'failure')];
const row = nightlyRow({ wf: nightly, runs, jobs: [] }, NOW);
const [dot, name, state, strip] = row.children;
assert.equal(dot.className, 'dot');
assert.equal(row.className, 'ci-row bad');
assert.equal(name.text, 'Kernel checks');
assert.equal(name.href, 'https://github.com/Xilinx/mlir-aie/actions/workflows/nightlyKernelChecks.yml');
assert.equal(state.text, 'Failed 6 h ago');
// Oldest first, the latest outlined, each linked to its run.
assert.deepEqual(strip.children.map(n => n.className), ['night bad', 'night ok', 'night bad latest']);
assert.equal(strip.children[2].href, 'https://example.com/runs/3');
// GitHub did not answer: said, no strip.
const none = nightlyRow({ wf: nightly, runs: null, error: 'HTTP 502' }, NOW);
assert.deepEqual([none.className, none.children.length, none.children[2].text], ['ci-row none', 3, 'GitHub did not answer']);
// Two weeks of runs at most.
const many = Array.from({ length: 20 }, (_, i) => run(i, 24 * i + 1, 'completed', 'success'));
assert.equal(nightlyRow({ wf: nightly, runs: many, jobs: [] }, NOW).children[3].children.length, STRIP_RUNS);
// A merge-gating test's row: its latest run on main, its normal under it, its runs on main.
const crow = criticalRow(crit({ main_runs: [run(30, 3, 'completed', 'success'), run(29, 27, 'completed', 'failure')] }), NOW);
assert.equal(crow.children[2].text, 'Passed on main 3 h ago9 of 10 passed on main · 7% of commits re-run to pass · 1 h 24 min a run');
assert.deepEqual(crow.children[3].children.map(n => n.className), ['night bad', 'night ok latest']);
""")


def test_critical_workflows_judge_main_now_and_show_their_normal(page):
    page(CI_RUNS + """
// Healthy: its latest run on main passed; a third of PRs failing is not judged.
assert.deepEqual(criticalVerdict(crit(), NOW), { level: 'ok', kind: '', notes: [], failedJobs: [], late: false });
// Main's latest run failed: the jobs that failed are named.
let v = criticalVerdict(crit({ latest_main: run(30, 3, 'completed', 'failure'),
  latest_main_jobs: [job('Linux hardware tests (aie2p-8col)', 3, 'completed', 'failure'), job('lint', 3, 'completed', 'success')] }), NOW);
assert.deepEqual(v, { level: 'bad', kind: 'failing', notes: ['its latest run on main failed (aie2p-8col)'], failedJobs: ['aie2p-8col'], late: false });
assert.equal(criticalTile(crit({ latest_main: run(30, 3, 'completed', 'failure'),
  latest_main_jobs: [job('build', 3, 'completed', 'failure')] }), NOW).text, 'Failed on main 3 h ago: build');
// A one-off hardware blip: the run before the latest passed, so it watches
// rather than demanding attention, as CI_NORMAL promises.
v = criticalVerdict(crit({ latest_main: run(30, 3, 'completed', 'failure'),
  main_runs: [run(30, 3, 'completed', 'failure'), run(29, 27, 'completed', 'success')] }), NOW);
assert.deepEqual([v.level, v.kind, v.notes], ['warn', 'intermittent', ['its latest run on main failed; the previous run passed']]);
// Infrastructure loss is not evidence that the previous run passed, even
// if its GitHub conclusion contradicts the judged outcome.
for (const outcome of ['runner_lost', 'no_runner', 'failed', 'timed_out']) {
  v = criticalVerdict(crit({ latest_main: run(30, 3, 'completed', 'failure'),
    main_runs: [run(30, 3, 'completed', 'failure'), { ...run(29, 27, 'completed', 'success'), outcome }] }), NOW);
  assert.equal(v.level, 'bad');
  assert.ok(!v.notes[0].includes('the previous run passed'));
}
assert.equal(criticalVerdict(crit({ latest_main: run(30, 3, 'completed', 'failure'),
  main_runs: [run(30, 3, 'completed', 'failure'), { ...run(29, 27, 'completed', 'success'), outcome: 'passed' }] }), NOW).level, 'warn');
// Two in a row is not a blip: stays bad. No main_runs at all is the same:
// there is no evidence it is a blip, so it is not given the benefit of the doubt.
assert.equal(criticalVerdict(crit({ latest_main: run(30, 3, 'completed', 'failure'),
  main_runs: [run(30, 3, 'completed', 'failure'), run(29, 27, 'completed', 'failure')] }), NOW).level, 'bad');
assert.equal(criticalVerdict(crit({ latest_main: run(30, 3, 'completed', 'failure') }), NOW).level, 'bad');
// A two-week pass rate under 90%, or one commit in ten re-run, is its normal: shown, not flagged.
v = criticalVerdict(crit({ main: { passed: 8, failed: 2, rate: 0.8 }, flaky: { commits: 20, flaky: 3, rate: 0.15 } }), NOW);
assert.deepEqual([v.level, v.notes], ['ok', []]);
assert.equal(ciUsual(crit({ main: { passed: 12, failed: 0, timed_out: 2, rate: 0.857 } })), '12 of 14 passed on main, 2 timed out · 7% of commits re-run to pass · 1 h 24 min a run');
// Failing 2 of its last 5 runs though the latest passed: intermittent, amber.
const mainRuns = ['success', 'failure', 'success', 'failure', 'success'].map((c, i) => ({ ...run(30 - i, 3 + 24 * i, 'completed', c), outcome: c === 'success' ? 'passed' : 'failed' }));
v = criticalVerdict(crit({ main_runs: mainRuns }), NOW);
assert.deepEqual([v.level, v.kind, v.notes], ['warn', 'intermittent', ['its latest run passed, but 2 of its last 5 on main failed']]);
// One failure in five is a wedge now and then: normal.
assert.equal(criticalVerdict(crit({ main_runs: mainRuns.slice(0, 3) }), NOW).level, 'ok');
// Main runs it nightly: no run for 1.5 days means the nightly stopped.
v = criticalVerdict(crit({ every: 24, latest_main: run(30, 40, 'completed', 'success') }), NOW);
assert.deepEqual([v.level, v.kind, v.late, v.notes[0]], ['bad', 'stopped', true, 'no run on main for 1 day 16 h']);
assert.equal(criticalTile(crit({ every: 24, latest_main: run(30, 40, 'completed', 'success') }), NOW).text, 'No run on main for 1 day 16 h');
// A run still going counts as recent.
assert.equal(criticalVerdict(crit({ every: 24, latest_main: run(30, 40, 'completed', 'success'), newest_main: run(31, 2, 'in_progress', null) }), NOW).late, false);
assert.equal(criticalVerdict({ file: 'x.yml', error: 'HTTP 404' }).level, 'none');
""")


def test_only_hardware_workflows_get_first_failure_grace(page):
    page(CI_RUNS + """
const hardware = ['buildAndTestRyzenAI.yml', 'buildAndTestRyzenAIWindows.yml'];
const software = ['lintAndFormat.yml', 'buildAndTestMulti.yml', 'buildAndTestPythons.yml', 'unknown.yml'];
for (const file of [...hardware, ...software]) {
  for (const outcome of [undefined, 'failed', 'timed_out']) {
    const latest = { ...run(30, 3, 'completed', 'failure'), outcome };
    const w = crit({ file, latest_main: latest,
      main_runs: [latest, { ...run(29, 27, 'completed', 'success'), outcome: 'passed' }] });
    const v = criticalVerdict(w, NOW);
    const isHardware = hardware.includes(file);
    assert.deepEqual([v.level, v.kind], isHardware ? ['warn', 'intermittent'] : ['bad', 'failing'], file);
    assert.equal(v.notes[0].includes('the previous run passed'), isHardware);
    const summary = ciSummary(loadedOf(status({ critical: [w] })), NOW);
    assert.equal(summary.includes('Needs attention:'), !isHardware, summary);
    assert.equal(summary.includes('Nothing is broken'), isHardware, summary);
  }
}
""")


def test_merging_reads_as_figures_and_run_times_as_one_chart(page):
    page(CI_RUNS + """
assert.deepEqual(mergeFigures(status().merge_time).map(li => li.text),
  ['31pull requests merged in two weeks', '26.4 hmedian to merge', '42open, 11 for over 30 days']);
// The rest on hover.
assert.deepEqual(mergeFigures(status().merge_time).map(li => li.title), ['', 'the slowest tenth took 7.9 days or more', '9 of them drafts']);
assert.equal(mergeFigures({ error: 'HTTP 403' })[0].text, 'GitHub did not answer: HTTP 403');
assert.equal(formatMinutes(42.4), '42 min');
assert.equal(formatMinutes(120), '2 h');
// The run-time chart: a workflow's runs, oldest first, run time and wait.
const finished = (id, h, took, waited, conclusion) => ({ ...run(id, h, 'completed', conclusion),
  created_at: at(h + waited / 60), run_started_at: at(h), updated_at: at(h - took / 60), title: `run ${id}` });
const runs = [finished(3, 2, 90, 5, 'failure'), finished(2, 26, 80, 1, 'success'), run(4, 1, 'in_progress', null), finished(1, 50, 84, 2, 'success')];
const config = latencyConfig(runs, 'Run on main');
assert.deepEqual(config.data.datasets.map(d => [d.label, d.data]), [['Run time', [84, 80, 90]], ['Wait to start', [2, 1, 5]]]);
// A failed run's point is red and larger.
assert.deepEqual(config.data.datasets[0].pointRadius, [2.5, 2.5, 4]);
assert.equal(config.data.datasets[0].pointBackgroundColor[2], cssVar('--bad', '#bf2c37'));
// Labelled axes; hover names the commit, a click opens the run.
assert.equal(config.options.scales.y.title.text, 'Minutes');
assert.equal(config.options.scales.x.title.text, 'Run on main');
const tips = config.options.plugins.tooltip.callbacks;
assert.equal(tips.title([{ dataIndex: 2 }]), '6 Oct, 10:00 UTC, failed');
assert.equal(tips.afterTitle([{ dataIndex: 2 }]), 'abcdef1 run 3');
assert.equal(tips.label({ dataset: config.data.datasets[0], raw: 90 }), 'Run time: 1 h 30 min');
config.options.onClick(null, [{ index: 2 }]);
assert.deepEqual(opened, ['https://example.com/runs/3', '_blank', 'noopener']);
// The choices: merge-gating tests' runs on main, then the scheduled jobs'.
const loaded = loadedOf(status({ critical: [crit({ main_runs: runs })] }));
assert.deepEqual(latencyChoices(loaded).map(c => [c.title, c.axis, c.runs.length]),
  [['Linux hardware tests', 'Run on main', 4], ['Kernel checks', 'Scheduled run', 1]]);
""")


def test_the_published_status_says_how_fresh_it_is(page):
    page(CI_RUNS + """
assert.equal(ciStatus(loadedOf(status()), NOW).text, 'Read from GitHub 6 Oct, 11:30 UTC by publishCiHealth.yml, which runs every six hours.');
assert.equal(ciStatus(loadedOf(status()), NOW).level, '');
// The publisher stopped: its last answers, said to be old.
// Six hours old is the schedule, not a problem; past thirteen, a run was missed.
assert.equal(ciStatus(loadedOf(status()), NOW + 6 * H).level, '');
const old = ciStatus(loadedOf(status()), NOW + 13.5 * H);
assert.equal(old.level, 'bad');
assert.ok(old.text.startsWith('The CI health publisher last ran 14 h ago'));
// A workflow GitHub did not answer for.
const partly = ciStatus(loadedOf(status({ workflows: [{ ...nightly, runs: null, error: 'HTTP 502' }] })), NOW);
assert.ok(partly.text.endsWith('GitHub did not answer for 1 workflow: Kernel checks.'));
assert.equal(ciStatus(loadedOf(null), NOW).text, 'Nothing published yet: publishCiHealth.yml writes ci-health/status.json every six hours.');
assert.ok(ciStatus(loadedOf({ schema: 2 }), NOW).text.includes('newer version'));
""")


def test_the_overview_lists_the_workflows_that_need_a_look(page):
    page(CI_RUNS + """
const loaded = loadedOf(status({
  critical: [crit({ latest_main: run(30, 3, 'completed', 'failure') }), crit({ file: 'lint.yml', title: 'Lint' })],
  workflows: [{ ...nightly, runs: [run(2, 6, 'completed', 'failure')], jobs: [job('test', 6, 'completed', 'failure')] },
              { ...nightly, file: 'b.yml', title: 'B', runs: [run(3, 6, 'completed', 'success')], jobs: [] }],
}));
assert.deepEqual(ciLines(loaded, NOW).map(l => [l.level, l.who, l.text]), [
  ['bad', 'Linux hardware tests', 'Its latest run on main failed.'],
  ['bad', 'Kernel checks', 'Failed. test: failed.'],
  ['none', 'The rest', '1 of 2 merge-gating tests healthy on main; 1 of 2 scheduled jobs passed their last run.'],
  ['none', 'Merging', 'Median 26.4 h to merge; 42 open.'],
]);
const [tile] = overviewTiles(null, [], NOW, loaded);
assert.equal(tile.level, 'bad');
assert.equal(tile.children.at(-1).text, 'read 30 min ago');
// One line: what needs attention, then what to watch.
assert.equal(tile.children[2].text, 'Needs attention: Linux hardware tests and Kernel checks.');
assert.equal(ciSummary(loadedOf(status()), NOW), 'Nothing needs attention. 1 of 1 merge-gating test healthy on main; 1 of 1 scheduled job passed their last run.');
const flaky = ['success', 'failure', 'failure'].map((c, i) => ({ ...run(30 - i, 3 + 24 * i, 'completed', c), outcome: c === 'success' ? 'passed' : 'failed' }));
assert.equal(ciSummary(loadedOf(status({ critical: [crit({ main_runs: flaky })] })), NOW), 'Nothing is broken; watch: Linux hardware tests.');
assert.equal(overviewTiles(null, [], NOW, loadedOf(status({ critical: [crit({ main_runs: flaky })] })))[0].level, 'warn');
// All healthy.
const fine = ciLines(loadedOf(status()), NOW);
assert.deepEqual(fine[0], { level: 'ok', who: 'All workflows', text: '1 of 1 merge-gating test healthy on main; 1 of 1 scheduled job passed their last run.' });
// Nothing published: says so rather than "no results".
assert.equal(overviewTiles(null, [], NOW, loadedOf(null))[0].children[1].text, 'Not published yet');
""")


def test_overview_keeps_live_tiles_until_both_refresh_sources_arrive(page):
    page(CI_RUNS + """
const pending = () => {
  let resolve;
  const promise = new Promise(r => { resolve = r; });
  return { promise, resolve };
};
(async () => {
  const firstComponents = pending(), firstCi = pending();
  componentsLoad = firstComponents.promise;
  ciLoad = firstCi.promise;
  const first = renderOverview();
  const initial = [...$('overview-tiles').children];
  assert.equal(initial[0].children[1].text, 'Loading…');
  firstComponents.resolve([]);
  await Promise.resolve();
  assert.deepEqual($('overview-tiles').children, initial);
  firstCi.resolve(loadedOf(status()));
  await first;
  assert.notEqual($('overview-tiles').children[0].children[1].text, 'Loading…');
  for (const order of ['components', 'ci']) {
    const components = pending(), ci = pending();
    componentsLoad = components.promise;
    ciLoad = ci.promise;
    const live = [...$('overview-tiles').children];
    const refresh = renderOverview();
    assert.deepEqual($('overview-tiles').children, live);
    if (order === 'components') components.resolve([]);
    else ci.resolve(loadedOf(status()));
    await Promise.resolve();
    await Promise.resolve();
    assert.deepEqual($('overview-tiles').children, live);
    components.resolve([]);
    ci.resolve(loadedOf(status()));
    await refresh;
    assert.notEqual($('overview-tiles').children[0], live[0]);
    assert.notEqual($('overview-tiles').children[0].children[1].text, 'Loading…');
  }
})().catch(e => { console.error(e); process.exit(1); });
""")


def test_ci_health_does_not_read_a_timeout_or_a_stale_file_as_passing(page):
    page(CI_RUNS + """
// A job past its timeout ends the run "cancelled": a warning, not a pass.
let v = criticalVerdict(crit({ latest_main: run(30, 3, 'completed', 'cancelled') }), NOW);
assert.deepEqual([v.level, v.notes], ['warn', ['its latest run on main was cancelled: it timed out, or no runner took it']]);
assert.equal(criticalTile(crit({ latest_main: run(30, 3, 'completed', 'cancelled') }), NOW).text,
             'Its latest run on main was cancelled: it timed out, or no runner took it');
// A dead runner: the newest run on main has sat queued for a day.
v = criticalVerdict(crit({ newest_main: run(31, 26, 'queued', null) }), NOW);
assert.deepEqual([v.level, v.kind, v.notes.at(-1)], ['bad', 'runner', 'its newest run on main is waiting for a runner for 1 day 2 h']);
// "Failed on main" counts from the run's end, not its start.
const ended = { ...run(30, 3, 'completed', 'failure'), updated_at: at(1) };
assert.equal(criticalTile(crit({ latest_main: ended }), NOW).text, 'Failed on main 1 h ago');
// The run-time chart: a failed run red; a cancelled one (an older record, before outcomes) amber.
const done = (id, h, conclusion) => ({ ...run(id, h, 'completed', conclusion), updated_at: at(h - 1) });
const config = latencyConfig([done(3, 2, 'cancelled'), done(2, 26, 'failure'), done(1, 50, 'success')], 'Run on main');
assert.deepEqual(config.data.datasets[0].pointBackgroundColor,
                 [cssVar('--series-1', '#5b6475'), cssVar('--bad', '#bf2c37'), cssVar('--warn', '#9a5c00')]);
// Stale data: a gap is judged as of the publisher's last answer (plus the
// staleness limit), not the reader's clock. Published at NOW - 0.5 h, read 14 h later:
const fresh = status({ workflows: [{ ...nightly, runs: [run(1, 26, 'completed', 'success')], jobs: [] }] });
assert.equal(ciNow(loadedOf(fresh), NOW + 14 * H), NOW + 12.5 * H);
const row = nightlyRow(loadedOf(fresh).results[0], ciNow(loadedOf(fresh), NOW + 14 * H));
assert.equal(row.children[2].text, 'No run for 1 day 15 h');
// A late job whose last run is still going says so in a sentence.
v = ciVerdict(nightly, [run(1, 40, 'in_progress', null)], [], NOW);
assert.equal(v.headline, 'No scheduled run for 1 day 16 h. Its last is still running.');
// history.json is not fetched: the page never charted it.
assert.ok(!loadCi.toString().includes('history.json'));
""")


def test_ci_health_reads_the_outcome_ci_health_judged(page):
    page(CI_RUNS + """
// ci_health.py judges a run on main from its jobs: outcome, waited, took.
const judged = (id, h, outcome, waited, took) => ({ ...run(id, h, 'completed', outcome === 'passed' ? 'success' : 'failure'),
  updated_at: at(h - 1), outcome, waited, took });
let v = criticalVerdict(crit({ latest_main: judged(30, 3, 'timed_out', 7, 30) }), NOW);
assert.deepEqual([v.level, v.notes[0]], ['bad', 'its latest run on main timed out']);
v = criticalVerdict(crit({ latest_main: judged(30, 3, 'runner_lost', 1, 15) }), NOW);
assert.deepEqual([v.level, v.notes[0]], ['warn', 'its latest run on main lost its runner: the machine, not the tests']);
v = criticalVerdict(crit({ latest_main: judged(30, 3, 'no_runner', 0, 0) }), NOW);
assert.deepEqual([v.level, v.notes[0]], ['bad', 'no runner took its latest run on main']);
// Cancelled by hand or by a newer run: not a problem.
assert.equal(criticalVerdict(crit({ latest_main: { ...judged(30, 3, 'cancelled', 0, 5), conclusion: 'cancelled' } }), NOW).level, 'ok');
// A leg that lost its runner says so, not "failed".
const legs = ciLegs([{ ...job('Kernel checks (aie2p-8col)', 6, 'completed', 'failure'), outcome: 'runner_lost', runner: 'aie9' }], NOW);
assert.deepEqual(legs.map(l => [l.level, l.text]), [['warn', 'runner lost']]);
// The chart: the real runner wait and the run without it; timed out red, runner lost amber.
const config = latencyConfig([judged(3, 2, 'runner_lost', 12, 10), judged(2, 26, 'timed_out', 7.3, 30), judged(1, 50, 'passed', 0.5, 43.5)], 'Run on main');
assert.deepEqual(config.data.datasets.map(d => [d.label, d.data]), [['Run time', [43.5, 30, 10]], ['Wait to start', [0.5, 7.3, 12]]]);
assert.deepEqual(config.data.datasets[0].pointBackgroundColor,
                 [cssVar('--series-1', '#5b6475'), cssVar('--bad', '#bf2c37'), cssVar('--warn', '#9a5c00')]);
""")


def test_ci_health_machines_are_judged_on_their_latest_jobs(page):
    page(CI_RUNS + """
// Runs on main, each naming its machines' jobs; newest first, as published.
const onMain = (id, h, runners) => ({ ...run(id, h, 'completed', 'success'), updated_at: at(h), runners });
const ok = name => ({ runner: name, passed: true, outcome: 'passed' });
const no = (name, outcome = 'failed') => ({ runner: name, passed: false, outcome });
const machines = (mainRuns, jobs = []) => loadedOf(status({
  critical: [crit({ main_runs: mainRuns })],
  workflows: [{ ...nightly, runs: [run(1, 6, 'completed', 'success')], jobs }],
}));
const rowsOf = l => runnerRows(l, NOW, NOW).map(r => [r.name, r.level, r.text]);
// Two jobs in a row not passing: red, worst first; one failing where the
// others passed: amber, said; passing: when.
assert.deepEqual(rowsOf(machines([
  onMain(3, 10, [no('aie8'), ok('aie2'), no('aie1')]),
  onMain(2, 34, [no('aie8', 'timed_out'), ok('aie2'), ok('aie1')]),
  onMain(1, 58, [ok('aie8'), ok('aie2'), ok('aie1')]),
])), [
  ['aie8', 'bad', 'Its last 2 jobs did not pass, the latest 10 h ago (Linux hardware tests)'],
  ['aie1', 'warn', 'Its latest job failed 10 h ago (Linux hardware tests)'],
  ['aie2', 'ok', 'Latest job passed 10 h ago'],
]);
assert.equal(rowsOf(machines([onMain(2, 10, [no('aie1'), ok('aie2')]), onMain(1, 34, [ok('aie1'), ok('aie2')])]))[0][2],
             'Its latest job failed 10 h ago, where the other machines passed (Linux hardware tests)');
// A lost runner is the machine's: red at once.
assert.deepEqual(rowsOf(machines([onMain(1, 4, [no('aie9', 'runner_lost')])]))[0].slice(0, 2), ['aie9', 'bad']);
// Quiet for three days while the others ran: is it online? A nightly leg counts as a job.
const quiet = rowsOf(machines([onMain(2, 10, [ok('aie2')]), onMain(1, 100, [ok('aie7'), ok('aie2')])],
  [{ ...job('Kernel checks (aie2-4col)', 7, 'completed', 'success'), runner: 'aie2', started_at: at(7) }]));
assert.deepEqual(quiet[0], ['aie7', 'warn', 'No job since 2 Oct, 4 days ago, while the others ran: is it online?']);
assert.deepEqual(quiet[1], ['aie2', 'ok', 'Latest job passed 7 h ago']);
// One-off cloud machines: one row, never a reboot.
const pool = runnerRows(machines([onMain(1, 5, [ok('ephemeral-devlab-x2-la03-s02-d01-117543124080'), ok('ephemeral-devlab-x2-la03-s05-d02-122552206339')])]), NOW, NOW);
assert.deepEqual(pool.map(r => [r.name, r.level, r.machines]), [['ephemeral-devlab-x2-la03-*', 'none', 2]]);
// The rows: a strip of each machine's jobs, oldest first, the latest marked.
const box = runnerTable(machines([onMain(2, 10, [no('aie8')]), onMain(1, 34, [ok('aie8')])]), NOW);
const [rows, fresh] = box.children;
const [dot, name, state, strip] = rows.children[0].children;
assert.deepEqual([rows.children[0].className, name.text], ['ci-row warn', 'aie8']);
assert.equal(state.children[1].text, '1 of 2 jobs passed in two weeks · 1 failed');
assert.deepEqual(strip.children.map(n => n.className), ['night ok', 'night bad latest']);
assert.equal(fresh.children[0].text, 'Read from GitHub 6 Oct, 11:30 UTC by publishCiHealth.yml, which runs every six hours.');
// The overview names a machine that wants a reboot.
const p = ciProblems(machines([onMain(2, 10, [no('aie8')]), onMain(1, 34, [no('aie8')])]), NOW);
assert.deepEqual(p.problems.map(x => [x.level, x.who]), [['bad', 'aie8']]);
// No self-hosted runner named in the window.
assert.equal(runnerTable(loadedOf(status({ runners: [] })), NOW).children[0].text, 'No self-hosted runner named in a run in the window.');
// Nothing published at all.
assert.equal(runnerTable(loadedOf(null), NOW).children[0].text, 'No CI health published yet.');
""")


def test_ci_lateness_is_judged_when_the_publisher_looked(page):
    page(CI_RUNS + """
// Every four hours, its last run 5 h before the publisher looked: on time
// then, though the reader comes 3 h later and 1.5 periods (6 h) have passed.
const fourHourly = { ...nightly, every: 4 };
const published = status({ workflows: [{ ...fourHourly, runs: [run(1, 5.5, 'completed', 'success')], jobs: [] }] });
const later = NOW + 3 * H;
assert.equal(ciSeen(loadedOf(published), later), NOW - 0.5 * H);
assert.equal(ciVerdict(fourHourly, published.workflows[0].runs, [], later, ciSeen(loadedOf(published), later)).late, false);
assert.equal(ciProblems(loadedOf(published), later).problems.length, 0);
// Late as of the publisher's look: still late.
const stopped = status({ workflows: [{ ...fourHourly, runs: [run(1, 7, 'completed', 'success')], jobs: [] }] });
assert.equal(ciProblems(loadedOf(stopped), later).problems[0].who, 'Kernel checks');
""")


def test_ci_health_rows_have_one_state_each(page):
    page(CI_RUNS + """
// The rows: one state each (nightlyRow and criticalTile's own cases are
// test_nightly_jobs_are_a_row_each_with_their_recent_runs and
// test_critical_workflows_judge_main_now_and_show_their_normal's).
assert.equal(scheduledTile({ wf: nightly, runs: [run(5, 6, 'completed', 'success')], jobs: [] }, NOW).text, 'Passed 6 h ago');
assert.equal(scheduledTile({ wf: nightly, runs: [run(5, 40, 'completed', 'success')], jobs: [] }, NOW).text, 'No run for 1 day 16 h');
assert.equal(scheduledTile({ wf: nightly, runs: null, error: 'HTTP 502' }, NOW).text, 'GitHub did not answer');
assert.equal(jobName('Kernel checks (aie2p-8col)', 'Kernel checks'), 'aie2p-8col');
assert.equal(jobName('build-and-test (aie2p-8col)', 'Linux hardware tests'), 'build-and-test (aie2p-8col)');
""")


def test_performance_charts_the_design_every_batch_and_keys_its_colors(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
const data = collect(historiesOf('sa-placer-hw', [hwBefore, hwAfter]));
// NPU time: a line per batch, every batch, of seed 3 at the default effort
// alone (seed 7's 96.1 µs at batch 16 is the placer's, under Compiler checks).
const npuTab = hw.trend.find(t => t.id === 'npu_us_per_image');
const config = trendConfig(designOnly(data), npuTab);
const lines = config.data.datasets.filter(d => !d.band);
assert.deepEqual(lines.map(d => d.label), ['batch 1', 'batch 16']);
assert.deepEqual(lines[1].data, [95.1, 102]);
// A stack's key: each color, what it is, and its latest value.
const stack = hw.trend.find(t => t.stack);
const key = stackKey(stack, stackConfig(designOnly(data), stack));
assert.deepEqual(key.filter(n => n.tag === 'dt').map(n => n.children[1]), ['end to end', 'host: Python and buffer syncs', 'launch overhead', 'NPU work']);
assert.ok(key.find(n => n.tag === 'dd').text.startsWith('line: the whole Python call'));
assert.ok(Object.keys(COMPILE_KEY).every(k => COMPILE_GROUPS.some(([g]) => g === k) || /^whole (cold|warm) compile$/.test(k)));
assert.ok(COMPILE_GROUPS.every(([g]) => g in COMPILE_KEY));
""")


def test_performance_time_is_one_stack_per_batch(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
const data = collect(historiesOf('sa-placer-hw', [hwBefore, hwAfter]));
const config = stackConfig(data, hw.trend.find(t => t.stack));
// Where an image's time goes at the smallest batch, the mean over the seeds;
// the batches to choose from.
assert.deepEqual(config.data.datasets.map(d => d.label), ['NPU work', 'launch overhead', 'host: Python and buffer syncs', 'end to end']);
assert.deepEqual([config.data.keys, config.data.key], [[1, 16], 1]);
// The caption says what it is and that it is the mean over the seeds.
const box = trendBox(data, hw.trend, 'us_per_image');
// The design's own placement, not a mean over seeds.
assert.ok(!box.children[2].textContent.includes('mean over the seeds'), box.children[2].textContent);
// One point per nightly, by date; a plain axis from zero, so the parts read as shares.
assert.deepEqual(config.data.labels, ['1 Oct', '2 Oct']);
assert.equal(config.options.scales.y.type, 'linear');
assert.equal(config.options.scales.y.title.text, 'Time per image (µs) · ↓ better');
assert.equal(config.options.scales.x.title.text, 'Nightly run');
// Hover names the run's commit; a click opens it.
const tips = config.options.plugins.tooltip.callbacks;
assert.equal(tips.afterTitle([{ dataIndex: 1 }]), 'abcdef1 same revision');
assert.equal(tips.footer([{ dataIndex: 1 }]), 'Click for the commit');
assert.equal(tips.label({ dataset: config.data.datasets[0], raw: 245.8 }), 'NPU work: 245.8 µs');
config.options.onClick(null, [{ index: 1 }]);
assert.deepEqual(opened, ['https://github.com/Xilinx/mlir-aie/commit/abcdef123456', '_blank', 'noopener']);
assert.equal(commitLine({ id: '0123456789abcdef', message: 'x'.repeat(100) + '\\nbody' }), `0123456 ${'x'.repeat(71)}…`);
// Compile time, seconds, on a plain axis.
const compile = stackConfig(data, { ...COMPILE_TABS[0], stack: COMPILE_TABS[0].split, missingIsZero: true });
assert.equal(compile.options.scales.y.title.text, 'Compile time (s) · ↓ better');
// Compile peak memory: the whole compile's alone, cold and warm, from zero.
// Records from before the whole compile was measured have no point on it.
assert.deepEqual(COMPILE_TABS[1].lines.map(([name, ms]) => [name, ms]), [['cold', ['compile_peak_mib']], ['warm', ['compile_peak_mib_warm']]]);
const memory = trendConfig(data, COMPILE_TABS[1]);
assert.equal(memory.options.scales.y.type, 'linear');
assert.equal(memory.options.scales.y.beginAtZero, true);
assert.equal(memory.options.scales.y.title.text, 'Peak memory (MiB) · ↓ better');
assert.deepEqual(memory.data.datasets.filter(d => !d.band).map(d => d.label), []);
// The box: a switch for the tab; a link to a warm metric opens the tab that draws it.
// MobileNet's compile is charted under Compiler checks, beside its placement.
assert.equal(trendBox(data, hwPlacement.trend, 'compile_s').metric(), 'compile_s');
// Each tab its own id, so the switch tells them apart.
assert.equal(new Set(hwPlacement.trend.map(t => t.id)).size, hwPlacement.trend.length);
const seg = box.children[0].children[0];
assert.deepEqual(seg.children.map(b => b.textContent), ['NPU time', 'End to end', 'Where the time goes']);
// Its batches to choose from; a click redraws at that batch.
const batchRow = box.children[1].children[0];
assert.equal(batchRow.text, 'Batch 116');
batchRow.children[1].children[1].on.click();
assert.equal(chart.data.key, 16);
assert.equal(box.children[1].children[0].children[1].children[1].className, 'on');
assert.equal(trendBox(collect([]), hw.trend, '').text, 'No history published yet.');
""")


def test_without_chartjs_the_performance_chart_is_drawn_as_svg(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
// Round ticks: 1, 2 and 5 times a power of ten on a log axis.
// Too few 1-2-5 values in a narrow range: round steps instead.
assert.deepEqual(niceTicks(62, 353, true), [100, 200, 300]);
assert.deepEqual(niceTicks(60, 2100, true), [100, 200, 500, 1000, 2000]);
assert.deepEqual(niceTicks(38, 42, false), [38, 39, 40, 41, 42]);
// A holder as the page has: the canvas inside a box inside the chart.
const outer = document.createElement('div');
const holder = document.createElement('div');
holder.parentNode = outer; holder.clientWidth = 600; holder.clientHeight = 300;
outer.insertBefore = (node, before) => outer.children.unshift(node);
const canvas = document.createElement('canvas');
canvas.parentNode = holder;
holder.append(canvas);
const data = collect(historiesOf('sa-placer-hw', [hwBefore, hwAfter]));
const drawn = svgChart(canvas, stackConfig(data, hw.trend.find(t => t.stack)));
// A filled area per part, the end-to-end line over them, the legend naming each.
assert.deepEqual(drawn.legend, ['NPU work', 'launch overhead', 'host: Python and buffer syncs', 'end to end']);
assert.equal(drawn.svg.filter(p => p.startsWith('<polyline')).length, 1);
assert.equal(drawn.svg.filter(p => p.startsWith('<polygon')).length, 3);
// The axis titles, as the Chart.js chart has them.
assert.ok(drawn.svg.some(p => p.includes('rotate(-90)') && p.includes('Time per image (µs) · ↓ better')));
assert.ok(drawn.svg.some(p => p.includes('>Nightly run<')));
// Each point: hover text with the commit, and a link to it.
assert.ok(drawn.svg.some(p => p.includes('<title>2 Oct, 06:30 UTC\\nabcdef1 same revision\\nend to end: ') && p.includes('href="https://github.com/Xilinx/mlir-aie/commit/abcdef123456"')));
assert.ok(canvas.hidden);
assert.equal(outer.children[0].className, 'note svg-legend');
// A log axis with nothing on it (every stage 0 s) draws an empty chart.
const empty = svgChart(canvas, { data: { labels: ['1 Oct'], datasets: [{ label: 'kernels', data: [0] }] },
                                 options: { scales: { y: { type: 'logarithmic' } } } });
assert.ok(empty.svg.length < 20);
// Chart.js loaded: lineChart uses it.
assert.ok(lineChart(canvas, { data: { labels: [], datasets: [] }, options: { scales: {} } }) instanceof Chart);
""")


def test_the_harness_own_tests_are_said_apart_from_kernels(page):
    page(RUNS + """
const harness = 'test/python/npu/test_kernels_perf.py::test_a_baseline_that_fails_the_contract_is_timed';
const kernelTest = 'test_kernels_e2e.py::test_kernel_e2e[relu/64/bf16]';
const night = runs => renderDashboard(['npu1'], fromRecords({}), new Map([['npu1', { ...catalogue, kernels: [] }]]),
  new Map([['npu1', { target: 'npu1', runs: [summary('3', '1970-01-01T00:00:03Z', 'turbo', {}, runs)] }]]), 3600 * 1000);
night({ failed: [harness, kernelTest] });
const items = $('attention-list').children.map(li => li.children[1].text);
assert.ok(items.some(t => t.startsWith('1 test of the timing harness itself failed: it tests the checks, not a kernel.test_a_baseline_that_fails_the_contract_is_timed')));
assert.ok(items.some(t => t.startsWith('1 test failed in this run:relu/64/bf16')));
// A run that refused to time: every test of the timing step is the refusal's, not listed.
night({ failed: [harness], refused: "power mode is default, required 'turbo'", sane: null, published: false });
assert.ok(!$('attention-list').text.includes('timing harness'));
""")


def test_the_change_table_lists_steps_and_creeps_over_two_weeks(page):
    page(RUNS + """
// Fourteen nightlies, a day apart. d steps last night (100 for a week,
// then 110); e creeps (2% a night for a week, never a step); a comes back
// to where it was (100, 90 for a night, 100): neither.
const day = d => 86400000 * d;
const flat = n => Array.from({ length: n }, () => 100);
const dV = [...flat(13), 110], eV = [...flat(7), 102, 104, 106, 108, 110, 112, 112], aV = [...flat(12), 90, 100];
const db3 = fromRecords({ npu1: dV.map((_, i) => rec(String(i + 1), day(10 + i), 'turbo',
  { 'd/1/i8/cycles': ['cycles', dV[i]], 'e/1/i8/cycles': ['cycles', eV[i]], 'a/1/i8/cycles': ['cycles', aV[i]] })) });
location.hash = '#view=night';
const rows = renderWindowChanges(db3, ['npu1']).map(r => r.children.map(c => c.text));
assert.deepEqual(rows, [
  ['e/1/i8', 'npu1', '100 → 112 (+12.0%)Since 11 Jan', '—'],
  ['d/1/i8', 'npu1', '100 → 110 (+10.0%)Step 24 Jan', '—'],
]);
// No toggle to press: one list.
assert.equal($('window-compare').children.length, 0);
assert.ok($('window-about').text.includes('A step is the latest nightly past the threshold'));
""")

def test_a_kernels_cases_that_moved_together_are_one_row(page):
    page(RUNS + """
db = fromRecords({ npu2: [
  rec('1', 1, 'turbo', { 'mm/1/f32/gelu/kernel_object_bytes': ['bytes', 2048], 'mm/1/f32/relu/kernel_object_bytes': ['bytes', 2048],
                         'swiglu/1/bf16/kernel_object_bytes': ['bytes', 9000] }),
  rec('2', 2, 'turbo', { 'mm/1/f32/gelu/kernel_object_bytes': ['bytes', 4403], 'mm/1/f32/relu/kernel_object_bytes': ['bytes', 4300],
                         'swiglu/1/bf16/kernel_object_bytes': ['bytes', 16700] }),
] });
const rows = renderWindowChanges(db, ['npu2']);
// Cycles, then object size; the dates in the header.
const head = $('window-changes').children[0].children[0].children[0].children[0];
assert.deepEqual(head.children.map(c => c.text), ['Case', 'NPU', 'Cyclesthen → now (change)', 'Object sizethen → now (change)']);
assert.deepEqual(rows.map(r => r.children.map(c => c.text)), [
  ['mm 2 cases', 'npu2', '—', '2.0 KiB → 4.3 KiB (+115.0%)Step 1 Jan2 of 2 cases moved'],
  ['swiglu/1/bf16', 'npu2', '—', '8.8 KiB → 16.3 KiB (+85.6%)Step 1 Jan'],
]);
// Hover gives each case.
assert.equal(rows[0].children[3].title, 'mm/1/f32/gelu: 2.0 KiB → 4.3 KiB (+115.0%)\\nmm/1/f32/relu: 2.0 KiB → 4.2 KiB (+110.0%)');
// A measure that did not move is greyed: size grew, cycles flat.
const both = fromRecords({ npu2: [
  rec('1', 1, 'turbo', { 'k/1/i8/cycles': ['cycles', 100], 'k/1/i8/kernel_object_bytes': ['bytes', 2048] }),
  rec('2', 2, 'turbo', { 'k/1/i8/cycles': ['cycles', 100.5], 'k/1/i8/kernel_object_bytes': ['bytes', 4096] }),
] });
const [row] = renderWindowChanges(both, ['npu2']);
assert.deepEqual(row.children.slice(2).map(c => [c.className, c.text]),
  [['num faint', '—'], ['num', '2.0 KiB → 4.0 KiB (+100.0%)Step 1 Jan']]);
assert.equal(rows[0].children[0].children[0].title, 'mm/1/f32/gelu\\nmm/1/f32/relu');
""")
