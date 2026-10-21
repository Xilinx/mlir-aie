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
// One chart per case and metric, a line per NPU.
const groupOf = (data, kase) => chartGroups(data.series).find(g => !kase || g.kase === kase);
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
assert.equal(changeClass(100, 101.9, 'cycles'), '');
assert.equal(changeClass(100, 102, 'cycles'), 'worse');
assert.equal(changeClass(100, 98, 'kernel_object_bytes'), 'better');
assert.equal(changeClass(100, 109, 'npu_us'), '');
assert.equal(changeClass(100, 110, 'npu_us'), 'worse');
assert.equal(changeClass(100, 114, 'compile_s'), '');
assert.equal(changeClass(100, 115, 'compile_s'), 'worse');
assert.equal(changeClass(1000, 1009, 'final_cost_mean'), '');
assert.equal(changeClass(1000, 1010, 'final_cost_mean'), 'worse');
assert.deepEqual(GATED, ['cycles', 'kernel_object_bytes', 'max_ulp_bfloat16', 'max_ulp_float32', 'not_correctly_rounded_bfloat16', 'not_correctly_rounded_float32']);
""")


def test_the_direction_of_better_is_said_where_it_is_up(page):
    html = PAGE.read_text(encoding="utf-8")
    assert "Every metric is smaller-is-better" not in html
    assert "Lower is better, except where a metric says higher is better" in html
    page("""
const up = Object.keys(THRESHOLDS).filter(m => THRESHOLDS[m].higher_is_better);
assert.ok(up.includes('tok_s') && up.includes('fps_b1') && up.includes('images_passed'));
// Every metric a section lists says so when higher is better, and only then.
for (const m of SECTIONS.performance.metrics) {
  assert.equal(termOf(m, COMPONENT_METRICS[m]).includes('higher is better'), up.includes(m), m);
}
assert.equal(termOf('tok_s', 'x'), 'x; higher is better; counts past 5%');
assert.equal(termOf('npu_us', 'x'), 'x; counts past 10% and 3× its spread');
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
assert.ok(COMPONENT_METRICS.us_per_image_b1.includes('the fastest of its timed calls'));
assert.ok(trendOf('yolo26n-hw')[0].axis.includes('fastest call'));
assert.ok(!/the median of/.test(COMPONENT_METRICS.us_per_image_b1 + trendOf('yolo26n-hw')[0].about));
// Llama's timed run takes the median dispatch.
assert.ok(COMPONENT_METRICS.tok_s.includes('median 4-token dispatch'));
assert.ok(trendOf('llama-chain-hw')[1].axis.includes('median dispatch'));
// A host-observed dispatch is never called NPU time; NPU time is the dispatch less its baseline.
assert.ok(!/NPU time/.test(COMPONENT_METRICS.us_per_image + COMPONENT_METRICS.ms_per_token));
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
    common = (ROOT / "programming_examples/ml/mobilenet/bottleneck/_common.py").read_text(
        encoding="utf-8"
    )
    efforts = {
        "default": float(re.search(r"def sa_placer_flags\(.*effort=([\d.]+)", common)[1]),
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
assert.equal(kernelWindow(cyclesOn([[10, 100], [20, 101]]), 'npu1', 'skip').since, null);
// One constant for the strip, the sparklines and the window.
assert.equal(WINDOW_NIGHTS, 30);
const many = { target: 'npu1', runs: Array.from({ length: 40 }, (_, i) => summary(String(i), new Date(day(1) + i * 3600e3).toISOString(), 'turbo', {})) };
const strip = nightStrip({ runs: many.runs, run: { id: '39' } });
assert.equal(strip.children.length - 1, WINDOW_NIGHTS);
// A card's change says what it is against on hover.
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
// The trend: a line per part, the second dashed; each provenance change marked.
const rate = trendConfig(data, llama.trend[0]);
const lines = rate.data.datasets.filter(d => !d.band);
assert.deepEqual(lines.map(d => [d.label, d.data, d.borderDash.length > 0]), [
  ['dispatch rate · Strix', [4.7, null, 4.37], false], ['dispatch rate · Krackan', [null, 4, null], true]]);
assert.deepEqual(rate.plugins, [MARKERS]);
assert.deepEqual(rate.options.plugins.markers.at.map(m => m.label), ['device NPU Krackan 1, kernels k2', 'device NPU Strix, kernels k3']);
assert.deepEqual(rate.options.plugins.tooltip.callbacks.title([{ dataIndex: 1 }]), ['2 Oct, 06:30 UTC', 'changed: device NPU Krackan 1, kernels k2']);
// Compile time moves with the part's host too.
assert.deepEqual(trendConfig(data, COMPILE_TABS[0]).data.datasets.filter(d => !d.band).map(d => d.label),
                 ['cold · Strix', 'cold · Krackan', 'warm · Strix', 'warm · Krackan']);
// Sparklines of a time follow the latest night's part.
const card = componentBand(llama, { target: 'llama-chain-hw', runs: swaps.map(n => night(n.id, n.date, 'turbo', { device: n.device, provenance: n.provenance })) },
                           swaps[2], NOW, swaps[1]);
fillSparks(card.cards, data);
assert.equal((card.cards[0].sparks.get('tok_s').innerHTML.match(/<circle/g) || []).length, 1);
assert.equal(card.cards[0].sparks.get('tok_s').title, 'the last 2 nightlies');
// The part is named on the band and its card, and the swap is said once, as a warning.
assert.ok(card.children[0].text.includes('on Strix'));
assert.ok(card.cards[0].children[0].text.includes('on Strix'));
assert.ok(card.text.includes('Device changed since the previous run: NPU Krackan 1 → NPU Strix.'));
assert.ok(!card.text.includes('kernels changed'));
""")
    page(KERNELS + """
// The kernel page marks provenance changes and splits a time by part.
const k = (id, day, us, device, kernels) => ({ ...rec(id, Date.parse(`2026-10-0${day}T06:00:00Z`), 'turbo', {
  'softmax/1024/bfloat16/npu_us': ['us', us], 'softmax/1024/bfloat16/cycles': ['cycles', 1000] }), provenance: { device, kernels } });
db = fromRecords({ npu2: [k('1', 1, 80, 'NPU Strix', 'a'), k('2', 2, 77, 'NPU Krackan 1', 'a'), k('3', 3, 81, 'NPU Strix', 'b')] });
const us = kernelTrendConfig(db, 'npu2', 'softmax', 'npu_us');
assert.deepEqual(us.data.datasets.map(d => d.label), ['1024/bfloat16 · Strix', '1024/bfloat16 · Krackan']);
assert.deepEqual(us.plugins, [MARKERS]);
assert.deepEqual(us.options.plugins.markers.at.map(m => m.index), [1, 2]);
const cycles = kernelTrendConfig(db, 'npu2', 'softmax', 'cycles');
assert.deepEqual(cycles.data.datasets.map(d => [d.label, d.data]), [['1024/bfloat16', [1000, 1000, 1000]]]);
assert.deepEqual(cycles.options.plugins.tooltip.callbacks.title([{ dataIndex: 2 }]), ['3 Oct, 06:00 UTC', 'changed: device NPU Strix, kernels b']);
// The window compares a time on one part only.
assert.deepEqual(windowChanges(db, 'npu2', 'npu_us').map(x => [x.then, x.now]), [[80, 81]]);
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
assert.equal(thresholdText('npu_us'), 'counts past 10% and 3× its spread');
// The router's failed mutations went 0 -> 2: flagged, said as "+2", not a percentage.
const before = routerRecord('6', '2026-10-01T03:17:00Z', 0);
before.rows.npu1.failed_mutations = cell(0, 'cases');
const after = routerRecord('7', '2026-10-02T03:17:00Z', 0);
const routerBand = componentBand(router, { target: 'router', runs: [night('6', before.date, null), night('7', after.date, null)] }, after, NOW, before);
assert.equal(one(routerBand, n => n.className === 'moves').text, 'Since 1 Oct:npu1 failed mutations +2');
assert.equal(one(routerBand.cards[0], n => /^chg /.test(n.className || '') && n.text === '+2').className, 'chg worse strong');
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
assert.ok(METRICS.max_ulp_bfloat16.includes('correctly rounded value of the mathematics'));
const ulp = kernelTrendConfig(db, 'npu2', 'softmax', 'max_ulp_bfloat16');
assert.equal(ulp.options.scales.y.title.text, 'Max error (ULP, bfloat16, worst result) · ↓ better');
assert.deepEqual(ulp.data.datasets[0].data, [0, 0, 5]);
assert.equal(ulp.options.scales.y.ticks.callback(5), '5 ulp');
assert.equal(kernelTrendConfig(db, 'npu2', 'softmax', 'not_correctly_rounded_bfloat16').options.scales.y.ticks.callback(49.2), '49.2%');
// Their floors (1 ulp, 0.1 points) make 0 -> 5 ulp a move from a zero baseline, said "+5 ulp".
const moved = movedSeries(db, 'npu2');
assert.deepEqual(moved.regressed.map(x => [x.series.metric, x.before, x.after]),
  [['max_ulp_bfloat16', 0, 5], ['not_correctly_rounded_bfloat16', 0, 49.2]]);
assert.equal(formatChange(moved.regressed[0].change, 5, 'ulp'), '+5 ulp');
// Gated, they read in the verdict as their own kind of move.
const night = nightOf('npu2', null, db, cat2);
assert.equal(verdictOf(night, Date.parse('2026-10-03T07:00:00Z')).headline, '1 case less accurate vs 2 Oct.');
// Cycles lead the sentence, whatever order the moves came in.
const mixed = { ...night, moved: { regressed: [
  { series: { metric: 'kernel_object_bytes', kase: 'a/1' } }, { series: { metric: 'cycles', kase: 'b/1' } }], improved: [] } };
assert.equal(verdictOf(mixed, Date.parse('2026-10-03T07:00:00Z')).headline, '1 case slower, 1 bigger vs 2 Oct.');
const item = attentionFor(night, Date.parse('2026-10-03T07:00:00Z')).find(i => i.level === 'bad');
assert.ok(item.body[0].startsWith('1 case got less accurate past its threshold vs 2 Oct:'));
// The Summary's table gains a column for an error metric only when one moved.
location.hash = '#view=night';
renderWindowChanges(db, ['npu2']);
const head = $('window-changes').children[0].children[0].children[0].children[0];
assert.deepEqual(head.children.map(c => c.text.split('1 Oct')[0].split('2 Oct')[0]),
  ['Case', 'NPU', 'Cycles', 'Object size', 'Max error (ULP, bfloat16)', 'Not correctly rounded (%, bfloat16)']);
""")


def test_a_metric_cell_links_only_to_a_trend_that_exists(page):
    page(COMPONENT_RUNS + LLAMA_NIGHTS + YOLO_NIGHTS + """
// Every metric a card shows links to its trend when one charts it, and is
// no link otherwise.
for (const m of Object.keys(COMPONENT_METRICS)) {
  const cell = componentCell(m, label(m), []);
  assert.equal(Boolean(cell.href), hasTrend(m, sectionOfMetric(m)), m);
  if (!cell.href) assert.ok(!cell.title.includes('Click'), m);
}
const card = componentBand(llama, llamaIndex, llamaAfter, NOW, llamaBefore).cards[0];
const cells = card.children[1].children;
const cellOf = text => cells.find(c => c.children[0].text === text);
// The DMA floor and the verified count have no trend: words, not a link.
assert.deepEqual(['Over the DMA floor', 'Tokens verified'].map(t => [cellOf(t).tag, cellOf(t).href]), [['div', undefined], ['div', undefined]]);
// Sizes stay as trends: the link opens the Compiled size tab.
assert.equal(cellOf('Compiled sizes').href, '#view=performance&metric=elf_bytes');
const data = collect(historiesOf('llama-chain-hw', [llamaBefore, llamaAfter]));
const box = trendBox(data, llama.trend, 'elf_bytes');
assert.equal(box.metric(), 'elf_bytes');
const sizes = trendConfig(data, COMPILE_TABS.find(t => t.id === 'elf_bytes'));
assert.deepEqual(sizes.data.datasets.filter(d => !d.band).map(d => d.label), ['full ELF', 'device image']);
assert.equal(sizes.options.scales.y.type, 'logarithmic');
// The decode rate opens the tab that draws it.
assert.equal(trendBox(data, llama.trend, 'decode_tok_s').metric(), 'tok_s');
// A link that names a metric opens Placement's Trends toggle and scrolls to it.
revealTrend('placement', $('placement-trends'));
assert.equal($('placement-more').open, true);
assert.ok($('placement-trends').scrolled);
// Performance has no toggle: its band's trend is scrolled to.
const slot = document.createElement('div');
revealTrend('performance', slot);
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


def test_static_dispatch_records_are_not_spliced_into_the_trend(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
// publish.py keeps an old static record's per-batch times under mobilenet/seed=S/batch=B.
const old = { ...hwBefore, id: '5', date: '2026-09-30T06:30:00Z', provenance: { dispatch: 'static' },
  rows: { 'mobilenet/seed=3/batch=1': { us_per_image: { value: 500, unit: 'us' } }, 'mobilenet/seed=3/batch=16': { us_per_image: { value: 150, unit: 'us' } } } };
const data = collect(historiesOf('sa-placer-hw', [old, hwBefore, hwAfter]));
const trend = trendConfig(data, hw.trend.find(t => t.id === 'us_per_image'));
assert.deepEqual(trend.data.datasets.filter(d => !d.band).map(d => [d.label, d.data]),
  [['batch 1', [295.7, 295.7]], ['batch 16', [98.5, 101.95]]]);
""")


def test_llama_decode_by_context_is_a_line_per_context(page):
    page(COMPONENT_RUNS + LLAMA_NIGHTS + """
// The context sweep: one compile, timed at each start position; a series per context.
const swept = (id, date, night) => {
  const r = llamaRecord(id, date, 4.37);
  for (const [ctx, ...ms] of [[1024, 52, 57.2], [8192, 70, 77], [131072, 390, 429]]) {
    r.rows[`${LLAMA}/ctx=${ctx}`] = { ms_per_token: { value: ms[night], unit: 'ms' }, decode_tok_s: { value: 1000 / ms[night], unit: 'tok/s' } };
  }
  return r;
};
const data = collect(historiesOf('llama-chain-hw', [swept('6', llamaBefore.date, 0), swept('7', llamaAfter.date, 1)]));
const tab = llama.trend.find(t => t.id === 'decode_by_context');
assert.equal(tab.name, 'Decode by context');
const config = trendConfig(data, tab);
// Decode tokens per second, higher is better, as mlir-air and FastFlowLM tabulate it.
const lines = config.data.datasets.filter(d => !d.band);
assert.deepEqual(lines.map(d => [d.label, d.data.map(v => Math.round(v * 100) / 100)]),
  [['1k context', [19.23, 17.48]], ['8k context', [14.29, 12.99]], ['128k context', [2.56, 2.33]]]);
assert.equal(config.options.scales.y.type, 'logarithmic');
assert.equal(config.options.scales.y.title.text, 'Decode tokens per second (median dispatch) · ↑ better');
// The time per token on hover.
assert.equal(config.options.plugins.tooltip.callbacks.label({ dataset: lines[2], dataIndex: 1, raw: 1000 / 429 }),
             '128k context: 2.33 tok/s, dispatch / token 429 ms');
// The plain decode tab charts the timed run alone, not the sweep summed in.
assert.deepEqual(trendConfig(data, llama.trend.find(t => t.id === 'ms_per_token')).data.datasets.filter(d => !d.band).map(d => d.data),
  [[1000 / 4.37, 1000 / 4.37]]);
assert.deepEqual(trendBox(data, llama.trend, '').children[0].children[0].children.map(b => b.textContent).slice(0, 3),
  ['Tokens / s', 'Dispatch / token', 'Decode by context']);
// The card and the summary stay on the timed run.
const card = componentBand(llama, llamaIndex, swept('7', llamaAfter.date, 1), NOW, swept('6', llamaBefore.date, 0));
assert.deepEqual(card.cards.map(c => c.kase), [LLAMA]);
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
// A flat series still spans its threshold: npu_us at 100, 10%.
const flat = yRange([{ data: [100, 100] }], 'npu_us');
assert.deepEqual(flat, { min: 87.5, max: 112.5 });
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
  assert.equal(ciVerdictBand(ci, NOW).className, 'verdict bad');
  const [tile] = overviewTiles(null, [], NOW, ci);
  assert.deepEqual([tile.level, tile.children[1].text], ['bad', 'Needs a look']);
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
// A Placement or Routing trend chart (draw).
const data = collect(historiesOf('sa-placer', [before, after]));
const holder = document.createElement('div');
const canvas = document.createElement('canvas');
holder.append(canvas);
document.createElement('div').append(holder);
const drawn = draw(canvas, chartGroups(data.series).find(g => g.metric === 'final_cost_max'), ['unknown'], data);
assert.ok(drawn.svg.some(p => p.startsWith('<polyline')));
assert.deepEqual(drawn.legend, ['sa-placer']);
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
    assert re.search(r"\.night\.warn \{ background: transparent; box-shadow: inset", css)
    assert ".ci-row.bad .dot::before" in css and ".k-row .k-bad .dot::before" in css
    # Headings nest: the overview has one, a section's blocks are h3s.
    assert '<h2 class="sr-only">Overview</h2>' in html
    assert "<h2>Needs a look</h2>" not in html and "<h3>Needs a look</h3>" in html
    # The theme button's accessible name is its visible text.
    assert re.search(r'<button id="theme" type="button" title="[^"]+">Dark</button>', html)
    page(COMPONENT_RUNS + HW_NIGHTS + """
// Placement's lanes: full effort dashed, in the line and its chip.
const card = componentBand(hwPlacement, hwIndex, hwAfter, NOW, hwBefore).cards[0];
assert.deepEqual(card.sparks.get('placement_cost').lines.map(([, key, dashed]) => [key.split('|')[1], Boolean(dashed)]), [
  ['mobilenet/seed=3/effort=default', false], ['mobilenet/seed=3/effort=full', true],
  ['mobilenet/seed=7/effort=default', false], ['mobilenet/seed=7/effort=full', true]]);
const chips = all(card.children[1].children[0], n => /^npu /.test(n.className || '')).map(n => n.className);
assert.deepEqual(chips, ['npu n-series-1', 'npu n-series-1 dashed', 'npu n-series-2', 'npu n-series-2 dashed']);
// A chart is an image named by what it charts and in which unit.
const data = collect(historiesOf('sa-placer-hw', [hwBefore, hwAfter]));
const box = trendBox(data, hw.trend, 'us_per_image');
const canvas = all(box, n => n.tag === 'canvas')[0];
assert.equal(canvas.role, 'img');
assert.equal(canvas.ariaLabel, 'Dispatch / image, Dispatch time per image (µs, fastest of 20), over recent nightlies');
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
        published |= set(re.findall(r'\(\s*"([a-z][a-z0-9_]*)",\s*"(?:bytes|cores|ms|s|us|tok/s|fps)"\s*\)', text))
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
    published |= {"us_per_image_b1", "us_per_image_b8", "fps_b1", "fps_b8"}
    # n_cores is dropped (the author's call): the page shows the layout as words.
    published -= {"n_cores"}
    assert {"max_ulp_float32", "not_correctly_rounded_bfloat16", "compile_s", "unrouted", "tok_s", "placement_cost",
            "failed_mutations", "npu_us_per_image", "host_overhead_us_per_image"} <= published
    page(f"""
const published = {json.dumps(sorted(published))};
for (const m of [...published, ...ALL_METRICS]) {{
  assert.ok(m in LABELS, `${{m}} has no label`);
  assert.ok(m in METRICS || m in COMPONENT_METRICS, `${{m}} has no description`);
}}
// A chart's tooltip formats its value as its ticks do.
db = fromRecords({{ npu1: [rec('1', 1, 'turbo', {{ 'a/1/i8/npu_us': ['us', 80.25] }}), rec('2', 2, 'turbo', {{ 'a/1/i8/npu_us': ['us', 81] }})] }});
draw(el0, groupOf(db), ['turbo'], db);
const tick = chart.options.scales.y.ticks.callback(81);
assert.equal(chart.options.plugins.tooltip.callbacks.label({{ dataset: chart.data.datasets[0], dataIndex: 1, raw: 81 }}), `npu1: ${{tick}}`);
// Information only: never worse or better.
THRESHOLDS.ort_matches = {{ ...THRESHOLDS.ort_matches, neutral: true }};
assert.equal(changeClass(8, 6, 'ort_matches'), '');
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
db = collect([history, { ...history, metric: 'npu_us', unit: 'us', series: { 'add/1024x16/bfloat16': { values: [1, 2] } } }]);
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
  let h = await loadHistories(['npu1'], ['cycles', 'npu_us']);
  assert.equal(h.series.length, 1);
  h = await loadHistories(['npu1'], ['cycles']);
  assert.deepEqual(fetched, ['../kernel-checks/npu1/history/cycles.json', '../kernel-checks/npu1/history/npu_us.json']);

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


def test_chart_draws_a_line_per_npu_and_mode_on_shared_nights(page):
    page("""
draw(el0, groupOf(db), ['turbo', 'performance'], db);
// npu1 and npu2 of one nightly (run t1) share a night.
// Dates only, each day once on the axis; the time is in the tooltip.
assert.deepEqual(chart.data.labels, Array(6).fill('1 Jan'));
assert.deepEqual([0, 1, 2].map(i => chart.options.scales.x.ticks.callback(null, i)), ['1 Jan', '', '']);
const lines = chart.data.datasets.filter(d => !d.band);
assert.deepEqual(lines.map(d => d.label), ['npu1 turbo', 'npu1 performance', 'npu2 turbo']);
assert.deepEqual(lines[0].data, [100, null, null, 110, null, 121]);
assert.deepEqual(lines[1].data, [null, 90, null, null, 95, null]);
assert.deepEqual(lines[2].data, [999, null, null, null, null, null]);
// Only the expected power mode is drawn solid.
assert.deepEqual(lines.map(d => d.borderDash.length > 0), [false, true, false]);
// npu2 sits ~10x above npu1, so it gets its own axis on the right.
assert.equal(lines[2].yAxisID, 'y1');
assert.equal(chart.options.scales.y1.position, 'right');
// The left axis says what it charts and which way is better.
assert.equal(chart.options.scales.y.title.text, 'Cycles per call (fastest traced call) · ↓ better');
assert.equal(chart.options.scales.y1.title.display, false);
const cb = chart.options.plugins.tooltip.callbacks;
assert.deepEqual(cb.title([{ dataIndex: 5 }]), ['1 Jan, 00:05 UTC']);
assert.equal(cb.afterTitle([{ dataIndex: 5 }]), 'abcdef1 same revision');
// The tooltip formats as the ticks do; the axis title names the unit.
assert.equal(cb.label({ dataset: lines[0], dataIndex: 5, raw: 121 }), 'npu1 turbo: 121');
assert.deepEqual(cb.footer([{ dataIndex: 0 }]), ['npu1: turbo', 'npu2: turbo']);
chart.options.onClick(null, [{ index: 5 }]);
assert.deepEqual(opened, [commit.url, '_blank', 'noopener']);
assert.deepEqual(chart.options.plugins.markers.at, []);
// Turbo alone: a line per NPU, labelled by the NPU, no mode.
draw(el0, groupOf(db), ['turbo'], db);
assert.deepEqual(chart.data.datasets.filter(d => !d.band).map(d => d.label), ['npu1', 'npu2']);
assert.equal(chart.data.labels.length, 4);
""")


def test_nearby_npus_share_one_axis(page):
    page("""
db = fromRecords({ npu1: [rec('1', 100000, 'turbo', sm(100))], npu2: [rec('1', 100000, 'turbo', sm(150))] });
draw(el0, groupOf(db), ['turbo'], db);
assert.ok(!chart.options.scales.y1);
assert.ok(chart.data.datasets.every(d => !d.yAxisID));
// At least +-5% of the middle, so noise is not blown up to fill the chart.
assert.ok(chart.options.scales.y.min <= 100 - 6 && chart.options.scales.y.max >= 150 + 6);
""")


def test_missing_npu_and_filtered_mode_preserve_repeated_runs(page):
    page("""
db = fromRecords({ npu1: [rec('1', 100000, 'turbo', sm(100)), rec('2', 200000, 'turbo', sm(110))] });
draw(el0, groupOf(db), ['performance', 'turbo'], db);
assert.equal(chart.data.datasets.length, 1);
assert.deepEqual(chart.data.datasets[0].data, [100, 110]);
assert.equal(latestMove(db.series[0], ['performance']), null);
""")


def test_npu_us_deviation_is_a_band_outside_the_legend_and_tooltip(page):
    page("""
const us = (v, range) => ({ 'softmax/1024x16/bfloat16/npu_us': ['us', v, range] });
db = fromRecords({ npu1: [
  rec('1', 100000, 'turbo', us(50, 'min 40.0 max 60.0 n=50')),
  rec('2', 200000, 'turbo', us(40, '± 2.5; min 38.0 max 45.0 n=50')),
]});
draw(el0, groupOf(db), ['turbo'], db);
const [line, low, high] = chart.data.datasets;
assert.equal(chart.data.datasets.length, 3);
assert.deepEqual(line.data, [50, 40]);
assert.deepEqual(low.data, [null, 37.5]);
assert.deepEqual(high.data, [null, 42.5]);
assert.equal(high.fill, '-1');
// The band takes its NPU's color.
assert.equal(low.backgroundColor, 'rgba(31, 111, 184, 0.2)');
const data = chart.data;
assert.deepEqual([0, 1, 2].map(i => chart.options.plugins.legend.labels.filter({ datasetIndex: i }, data)),
                 [true, false, false]);
assert.deepEqual([0, 1, 2].map(i => chart.options.plugins.tooltip.filter({ dataset: data.datasets[i] })),
                 [true, false, false]);
assert.equal(chart.options.plugins.tooltip.callbacks.label({ dataset: line, dataIndex: 1, raw: 40 }),
             'npu1: 40 µs (± 2.5; min 38.0 max 45.0 n=50)');
""")


def test_cycles_spread_is_a_band_from_the_min_to_the_median(page):
    page("""
const cy = (v, range) => ({ 'softmax/1024x16/bfloat16/cycles': ['cycles', v, range] });
db = fromRecords({ npu1: [
  rec('1', 100000, 'turbo', cy(100)),
  rec('2', 200000, 'turbo', cy(110, 'median 115 max 400 n=16; truncated')),
  rec('3', 300000, 'turbo', cy(105, 'median 105 max 105 n=16')),
]});
draw(el0, groupOf(db), ['turbo'], db);
const [line, low, high] = chart.data.datasets;
assert.deepEqual(line.data, [100, 110, 105]);
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
draw(el0, groupOf(db), ['performance'], db);
assert.deepEqual(chart.options.plugins.markers.at, [
  { index: 2, label: 'npu1: peano 22.0.0+bbbb, kernels k2' },
  { index: 3, label: 'npu1: host bench-2, runtime XRTHostRuntime, xrt 2.20.0, kernels k3' },
]);
const footer = chart.options.plugins.tooltip.callbacks.footer;
// What changed is in the tooltip, not drawn over the chart.
assert.equal(chart.options.plugins.tooltip.callbacks.title([{ dataIndex: 2 }])[1], 'changed: npu1: peano 22.0.0+bbbb, kernels k2');
assert.deepEqual(footer([{ dataIndex: 3 }]), ['npu1: Peano 22.0.0+bbbb, performance']);
assert.deepEqual(footer([{ dataIndex: 1 }]), ['npu1: performance']);
assert.equal(footer([]), '');
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
  'relu/1024/bf16/npu_us': ['us', 100], 'relu/1024/bf16/kernel_object_bytes': ['bytes', 4096],
  'gelu/1024/bf16/cycles': ['cycles', 2000], 'gone/1/i8/cycles': ['cycles', 10],
  'flat/1/i8/cycles': ['cycles', 500],
});
const r2 = rec('2', 2000, 'turbo', {
  'relu/1024/bf16/cycles': ['cycles', 1030], 'relu/1024/bf16/cycles_per_kop': ['cycles/1k-ops', 10.3],
  'relu/1024/bf16/npu_us': ['us', 150], 'relu/1024/bf16/kernel_object_bytes': ['bytes', 4096],
  'gelu/1024/bf16/cycles': ['cycles', 1500], 'flat/1/i8/cycles': ['cycles', 505],
  'fresh/1/i8/cycles': ['cycles', 5],
});
db = fromRecords({ npu1: [r1, r2] });
"""


def test_moved_series_use_the_thresholds_and_lead_with_gated_metrics(page):
    page(MOVED + """
const moved = movedSeries(db, 'npu1');
// cycles_per_kop is derived, an unchanged object is not a move, gone/ has no
// latest point, fresh/ no previous one, flat/ moved 1%.
assert.deepEqual(moved.regressed.map(x => [x.series.kase, x.series.metric, x.before, x.after]), [
  ['relu/1024/bf16', 'cycles', 1000, 1030],
  ['relu/1024/bf16', 'npu_us', 100, 150],
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


def test_latest_run_prefers_the_recorded_run_and_falls_back_to_the_series(page):
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
const now = 3600 * 1000;
assert.deepEqual(warningsFor('npu1', run, now).map(w => [w.level, w.text.split(':')[0]]), [
  ['warn', 'measured in power mode "default", not turbo'],
  ['warn', '1 test failed in this run'],
  ['warn', '1 case with a truncated trace'],
  ['info', 'peano changed since the previous run'],
  ['warn', 'host changed since the previous run'],
  ['warn', 'xrt changed since the previous run'],
]);
// A migrated run has no Actions link.
const migrated = latestRun('npu1', { runs: [{ ...index.runs[0], url: '' }] }, db, catalogue);
assert.equal(migrated.url, null);
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

// Without runs.json, the latest run's record stands in.
const fallback = latestRun('npu1', null, db, catalogue);
assert.equal(fallback.recorded, false);
assert.equal(fallback.url, null);
assert.equal(fallback.date, 2000);
assert.equal(fallback.commit, 'abcdef123456');
assert.equal(fallback.pmode, 'turbo');
assert.deepEqual(warningsFor('npu1', fallback, now), []);
assert.deepEqual(warningsFor('npu1', fallback, now + 48 * 3600 * 1000),
                 [{ level: 'bad', text: 'no run since 1 Jan, 00:00 UTC', stale: true }]);
// Only the catalogue (every run dropped): nothing published, nothing failed.
const empty = latestRun('npu1', { schema: 1, runs: [] }, fromRecords({}), catalogue);
assert.equal(empty.sane, null);
assert.deepEqual(warningsFor('npu1', empty, Date.parse(catalogue.date)).map(w => w.text),
                 ['no numbers have been published yet']);
// Nothing at all.
const none = latestRun('npu2', null, db, null);
assert.equal(none.date, null);
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
assert.equal(headline.text, '1 case slower vs 1 Jan, 1 failing correctness, 1 failed timing. 1 faster.');
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

// All kernels: a line each, those that need a look first; a state per NPU.
const list = renderKernelList(shownNights, db);
assert.deepEqual(list.map(r => [r.factory, r.className, r.children.slice(1, 3).map(c => c.text)]), [
  ['mm', 'k-row bad', ['npu11 failing', 'npu2not offered']],
  ['relu', 'k-row bad', ['npu1slower +3.0%1,030 cycles1 slower since 1 Jan', 'npu2not offered']],
  ['exp2f_vec', 'k-row none', ['npu1not offered', 'npu2not offered']],
]);
assert.equal(list[1].href, '#view=kernel&kernel=relu');
assert.equal($('kernels-status').text, '3 kernels of 3');
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
location.hash = '#view=night&changes=window';
const rows = renderWindowChanges(db, ['npu1', 'npu2']);
// One row per kernel and NPU; a column per measure, "then → now (change)".
assert.deepEqual(rows.map(r => r.children.map(c => c.text)),
  [['softmax/1024/bfloat16', 'npu1', '1,000 → 1,100 (+10.0%)', '—']]);
assert.equal(rows[0].children[0].children[0].children[0].href, '#view=kernel&kernel=softmax');
assert.ok($('window-about').text.startsWith("Each case's cycles and object size on its first nightly from 1 Jan against its latest"));
// Nothing moved: said so.
const flat = fromRecords({ npu1: [rec('1', 1, 'turbo', { 'a/1/i8/cycles': ['cycles', 100] }), rec('2', 2, 'turbo', { 'a/1/i8/cycles': ['cycles', 100] })] });
assert.deepEqual(renderWindowChanges(flat, ['npu1']), []);
assert.equal($('window-changes').text, 'Nothing moved past its threshold over these nightlies.');
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
$('kernels-filter').on.input();
assert.deepEqual(rows.map(r => r.hidden), [true, true, false]);
assert.equal($('kernels-status').text, '1 kernel of 3');
$('kernels-filter').value = '';
$('kernels-problems').checked = true;
$('kernels-problems').on.change();
assert.deepEqual(rows.map(r => r.hidden), [false, true, true]);
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


def test_case_titles_and_trend_lines(page):
    page("""
assert.deepEqual(caseParts('bn_conv2dk3/1792x8/int8_uint8/input_channels=8/stride=2/lut'),
  { factory: 'bn_conv2dk3', shape: '1792x8', dtype: 'int8_uint8', params: ['input channels 8', 'stride 2', 'lut'] });
assert.equal(caseTitle('mm/64x32x64x16/int16_int32').text, '64 × 32 × 64 × 16int16 int32');
// Each NPU relative to its own latest value: 10x apart, both lines end mid-axis.
const svg = sparkSvg([['npu1', [100, 110]], ['npu2', [1000, 1100]]], 'cycles');
assert.equal((svg.match(/<polyline/g) || []).length, 2);
assert.equal((svg.match(/cy="17.0"/g) || []).length, 2);
assert.equal(sparkSvg([['npu1', []]], 'cycles'), '');
// A dashed line (e.g. a warm compile against its cold lane) gets a dash
// pattern; its solid counterpart does not.
const dashedSvg = sparkSvg([['npu1', [100, 110]], ['npu1', [100, 90], true]], 'cycles');
assert.equal((dashedSvg.match(/<polyline/g) || []).length, 2);
assert.equal((dashedSvg.match(/stroke-dasharray/g) || []).length, 1);
""")


def test_a_kernels_chart_has_a_line_per_case_in_actual_cycles(page):
    page(KERNELS + """
const config = kernelTrendConfig(db, 'npu1', 'softmax', 'cycles');
assert.deepEqual(config.data.labels, ['1 Jan', '1 Jan']);
// Each case's actual cycles, night by night.
assert.deepEqual(config.data.datasets.map(d => [d.label, d.data]), [['64/bfloat16', [50, null]], ['1024/bfloat16', [1000, 1100]]]);
const tips = config.options.plugins.tooltip.callbacks;
assert.equal(tips.label({ dataset: config.data.datasets[1], dataIndex: 1, raw: 1100 }), '1024/bfloat16: 1,100');
assert.equal(tips.afterTitle([{ dataIndex: 1 }]), 'abcdef1 same revision');
assert.equal(config.options.scales.y.title.text, 'Cycles per call (fastest traced call) · ↓ better');
// 50 and 1,100 differ over ten times: a log axis, so both lines' moves show.
assert.equal(config.options.scales.y.type, 'logarithmic');
assert.equal(kernelTrendConfig(db, 'npu2', 'softmax', 'cycles').options.scales.y.type, 'linear');
// softmax_mask is another kernel, not a case of softmax.
assert.ok(!config.data.datasets.some(d => d.label.startsWith('mask')));
""")


def test_views_come_from_the_hash_and_old_links_keep_working(page):
    page("""
location.hash = '#view=catalogue';
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
// No hash is the overview; the kernels' last night was "dashboard".
location.hash = '';
assert.equal(applyView(), 'overview');
assert.equal($('overview').hidden, false);
assert.equal($('kernels-section').hidden, true);
assert.equal($('nav-overview').className, 'current');
location.hash = '#view=dashboard';
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
location.hash = '#view=routing';
assert.equal(applyView(), 'routing');
assert.equal($('routing').hidden, false);
""")


def test_npu_us_moves_only_past_its_noise(page):
    page("""
const us = (value, mad) => ({ value, unit: 'us', ...(mad === undefined ? {} : { range: `± ${mad}; min 1 max 2 n=50` }) });
// 20% moves: past 3x a 1 us MAD; not past 3x 10 us (either run's); no MAD, the percentage alone.
assert.equal(verdict('npu_us', us(100, 1), us(120, 1)), 'worse');
assert.equal(verdict('npu_us', us(100, 10), us(120, 1)), '');
assert.equal(verdict('npu_us', us(100, 1), us(120, 7)), '');
assert.equal(verdict('npu_us', us(100), us(120)), 'worse');
assert.equal(verdict('npu_us', us(120, 1), us(100, 1)), 'better');
// Many MADs but under 10%.
assert.equal(verdict('npu_us', us(100, 0.1), us(105, 0.1)), '');
// Other metrics ignore the range.
assert.equal(verdict('cycles', { value: 100, range: '± 50' }, { value: 103 }), 'worse');

db = fromRecords({ npu1: [
  rec('1', 1000, 'performance', { 'noisy/1/bf16/npu_us': ['us', 100, '± 10.0; min 1 max 2 n=50'],
                                   'steady/1/bf16/npu_us': ['us', 100, '± 1.0; min 1 max 2 n=50'] }),
  rec('2', 2000, 'performance', { 'noisy/1/bf16/npu_us': ['us', 125, '± 10.0; min 1 max 2 n=50'],
                                  'steady/1/bf16/npu_us': ['us', 125, '± 1.0; min 1 max 2 n=50'] }),
]});
assert.deepEqual(movedSeries(db, 'npu1').regressed.map(x => x.series.kase), ['steady/1/bf16']);
const { cases } = latestCases(db, 'npu1');
assert.equal(cases.get('noisy/1/bf16').metrics.get('npu_us').cls, '');
assert.equal(cases.get('steady/1/bf16').metrics.get('npu_us').cls, 'worse');
const move = latestMove(db.series.find(s => s.kase === 'noisy/1/bf16'), ['performance']);
assert.deepEqual(move, { change: 0.25, delta: 25, cls: '', since: 1000 });
""")


def test_charts_are_placed_a_box_each(page):
    page("""
db = fromRecords({ npu1: [
  rec('1', 1000, 'performance', { 'relu/1/bf16/cycles': ['cycles', 100], 'gelu/1/bf16/cycles': ['cycles', 100] }),
  rec('2', 2000, 'performance', { 'relu/1/bf16/cycles': ['cycles', 100], 'gelu/1/bf16/cycles': ['cycles', 110] }),
]});
const container = document.createElement('div');
placeCharts(container, chartGroups(db.series), ['performance'], db);
assert.deepEqual(container.children.map(d => d.className), ['chart', 'chart']);
// A move past its threshold is a chip on its box.
assert.ok(container.children.some(d => d.children[0].text.includes('npu1 +10.0%')));
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
const [sweep, hwPlacement, hw] = BANDS;
const night = (id, date, pmode, extra) => ({
  id, url: `https://example.com/runs/${id}`, date, commit: { ...commit }, pmode, provenance: {},
  sane: true, published: true, failed: [], ...(extra || {}),
});
const NOW = Date.parse('2026-10-02T07:00:00Z');
"""


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
// A card per fixture; a nonzero failed-seeds count is red.
assert.deepEqual(card.cards.map(c => c.kase), ['test_sa_effort', 'mobilenet']);
const [, grid] = card.cards[1].children;
assert.deepEqual(grid.children.map(c => c.children[0].text), ['Failed seeds', 'SA final cost', 'CPU time', label('peak_rss_mb_max')]);
assert.equal(grid.children[0].children[1].className, 'glance-row solo');
assert.equal(grid.children[0].children[1].children[0].className, 'vals worse');
assert.equal(grid.children[2].text, 'CPU timemean—max21,000 ms');
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
const seedRuns = (costs, cpu) => costs.map((final_cost, i) => ({ fixture: 'mobilenet', seed: i + 1,
  passed: final_cost !== null, final_cost, cpu_ms: cpu, peak_rss_mb: 200 }));
const sweepRecord = (id, date, costs, cpu) => {
  const passed = costs.filter(c => c !== null);
  const cell = (value, unit) => ({ value, unit });
  return { id, date, pmode: null, commit: { ...commit }, provenance: {}, detail: seedRuns(costs, cpu), rows: { mobilenet: {
    failed_seeds: cell(costs.length - passed.length, 'seeds'),
    final_cost_mean: cell(passed.reduce((a, b) => a + b, 0) / passed.length, 'cost'),
    final_cost_max: cell(Math.max(...passed), 'cost'),
    cpu_ms_mean: cell(cpu, 'ms'), cpu_ms_max: cell(cpu, 'ms'), peak_rss_mb_max: cell(200, 'MB'),
  } } };
};
const before = sweepRecord('6', '2026-10-01T06:30:00Z', [100, 100, 100, 100], 2000);
const after = sweepRecord('7', '2026-10-02T06:30:00Z', [100, 110, 100, null], 2100);
const sweepIndex = { target: 'sa-placer', runs: [night('6', before.date, null), night('7', after.date, null, { failed: ['mobilenet/seed4'] })] };
"""


def test_sweep_cards_and_seeds_chart_compare_with_the_nightly_before(page):
    page(COMPONENT_RUNS + SWEEP_NIGHTS + """
assert.equal(previousPublished(sweepIndex, after).id, '6');
const card = componentBand(sweep, sweepIndex, after, NOW, before);
assert.equal(one(card, n => n.className === 'headline').text, '1 seed failed: mobilenet/seed4.');
// The worst cost moved 10%, past its 1%; CPU time 5%, within its 25%.
const moves = one(card, n => n.className === 'moves');
// A seed that failed is a count from zero: its floor of 1 flags it.
assert.equal(moves.text, 'Since 1 Oct:mobilenet failed seeds +1mobilenet SA final cost (worst) +10.0%mobilenet SA final cost (mean) +3.3%');
const [header, grid] = card.cards[0].children;
assert.ok(header.text.includes('utils/component_checks/fixtures/mobilenet.mlir'));
assert.ok(header.text.includes('4 seeds'));
assert.equal(header.children.at(-1).text, 'worst ÷ mean cost 1.06×');
const cost = grid.children[1];
assert.equal(cost.text, 'SA final costmean103.3 +3.3%worst110 +10.0%');
assert.deepEqual(all(cost, n => /^chg /.test(n.className || '')).map(n => n.className), ['chg worse strong', 'chg worse strong']);
assert.equal(all(grid.children[2], n => /^chg /.test(n.className || ''))[0].className, 'chg flat');
assert.equal(cost.href, '#view=placement&metric=final_cost_mean');
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
// The trend lines: a line each for the mean and the worst cost.
fillSparks(card.cards, collect(historiesOf('sa-placer', [before, after])));
const spark = card.cards[0].sparks.get('final_cost_mean');
assert.ok(spark.innerHTML.includes('var(--series-1)') && spark.innerHTML.includes('var(--series-2)'));
""")


HW_NIGHTS = """
// Two hardware nightlies, seeds 3 and 7 at batch 1 and 16, each at the
// design's real default effort and a full-effort baseline: seed 3's batch
// 16 (default effort) is 7% slower in the latest, past its 5% and its MADs.
// The full-effort numbers are fixed (not parameterized by b16) and clearly
// distinct from default's, so a test can tell the two apart.
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
      for (const [batch, us] of [[1, 295.7], [16, effort === 'default' && seed === 3 ? b16 : (effort === 'default' ? 99 : 120)]]) {
        rows[`mobilenet/seed=${seed}/batch=${batch}/effort=${effort}`] = {
          us_per_image: { value: us, unit: 'us', range: `median ${us + 1}, MAD 0.20, p95 ${us + 2}` },
          npu_us_per_image: { value: Math.round((us - 47 / batch) * 10) / 10, unit: 'us', range: 'median 1, MAD 0.20, p95 2' },
          host_overhead_us_per_image: { value: 100, unit: 'us', range: 'median 101, MAD 0.40, p95 103' },
          e2e_us_per_image: { value: us + 100, unit: 'us' },
        };
      }
    }
  }
  rows.mobilenet = { failed_seeds: { value: 0, unit: 'seeds' } };
  const detail = [3, 7].flatMap(seed => ['default', 'full'].flatMap(effort_label =>
    [1, 16].map(batch => ({ seed, batch, effort_label, passed: true, placement: '0123456789ab' }))));
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
             "Every run passed, but 2 measures got worse vs 1 Oct. Seed 3's NPU time is 248.7 µs per image at batch 1, 102 µs at batch 16.");
const repro = one(card, n => n.className === 'repro');
assert.ok(!repro.open);
// Shown: the verdict, what moved, and the trend chart's slot. Behind
// Details: what the check does, the cards, the batch chart, how to run it.
const top = card.children.map(c => c.className);
assert.deepEqual(top, ['verdict-top', 'headline', 'moves', 'trend-slot', 'band-more']);
const more = card.children.at(-1);
assert.ok(!more.open);
assert.ok(all(more, n => n.className === 'glance').length === 1);
assert.ok(all(more, n => n.className === 'repro').length === 1);
assert.equal(card.trend.className, 'trend-slot');
// A quiet night: "nothing moved" is detail too.
const quiet = componentBand(hw, hwIndex, hwRecord('7', hwAfter.date, 98), NOW, hwBefore);
assert.deepEqual(quiet.children.map(c => c.className), ['verdict-top', 'headline', 'trend-slot', 'band-more']);
assert.equal(quiet.className, 'verdict ok');
assert.ok(one(repro, n => n.tag === 'pre').text.includes('python -m mobilenet.aie2_mobilenet_iron --sa-seed N --sa-effort 0.05 --dynamic 1 16 -w 20 -i 20'));
assert.ok(card.text.includes('turbo mode'));
// Nothing published yet.
const empty = componentBand(hw, null, null, NOW);
assert.equal(empty.className, 'verdict none');
assert.equal(one(empty, n => n.className === 'headline').text, 'No results published yet.');
""")


def test_hardware_seed_cards_show_each_batch_against_the_nightly_before(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
const card = componentBand(hw, hwIndex, hwAfter, NOW, hwBefore);
assert.equal(one(card, n => n.className === 'moves').text,
             'Since 1 Oct:seed 3, batch 16, default effort NPU time / image +7.3%seed 3, batch 16, default effort dispatch / image +7.0%');
assert.deepEqual(card.cards.map(c => c.kase), ['mobilenet/seed=3/effort=default', 'mobilenet/seed=3/effort=full', 'mobilenet/seed=7/effort=default', 'mobilenet/seed=7/effort=full']);
const [header, grid] = card.cards[0].children;
assert.equal(header.children[0].text, "seed 3the design's default");
// The placement, its cost under Placement.
assert.equal(header.children[1].text, 'placement 0123456789ab');
assert.equal(header.children[1].href, '#view=placement&metric=placement_cost');
assert.equal(header.children.at(-1).text, 'b16 ÷ b1 per image 0.35×');
assert.deepEqual(grid.children.map(c => c.children[0].text), ['NPU time / image', 'Host overhead / image', 'Dispatch / image', 'End-to-end / image']);
// A lane per batch, colored past the threshold only.
const perImage = grid.children[0];
assert.equal(perImage.text, 'NPU time / imageb1248.7 µs 0.0%b16102 µs +7.3%');
assert.deepEqual(perImage.children.slice(1, 3).map(r => r.children[0].className), ['npu n-series-1', 'npu n-series-2']);
assert.deepEqual(all(perImage, n => /^chg /.test(n.className || '')).map(n => n.className), ['chg flat', 'chg worse']);
// End-to-end too: its label, a lane per batch and its trend line.
assert.equal(grid.children[1].children.length, 4);
// One compile serves every batch: its block, below the batches, reads the
// seed/effort case. These runs have no stage profile, so each compile is
// its total and peak memory, with no bar and no key.
const block = card.cards[0].children[2];
assert.equal(block.className, 'stages');
const compiles = block.children.filter(r => r.className === 'stage-row');
assert.deepEqual(compiles.map(r => r.text), ['cold40 s 0.0% · 850 MiB largest process', 'warm4.6 s 0.0% · 700 MiB largest process']);
assert.equal(block.children[0].text, 'Compile');
assert.equal(all(block, n => /^stage-seg /.test(n.className || '')).length, 0);
assert.equal(all(block, n => n.className === 'stage-key').length, 0);
// Neutral rather than a batch's color, the warm one dashed on the trend.
assert.deepEqual(['compile_s', 'compile_s_warm'].map(m => card.cards[0].sparks.get(m).lines.map(([color, key, dashed]) => [color, key, dashed])), [
  [['muted', 'sa-placer-hw|mobilenet/seed=3/effort=default|compile_s', false]],
  [['muted', 'sa-placer-hw|mobilenet/seed=3/effort=default|compile_s_warm', true]]]);
assert.equal(perImage.href, '#view=performance&metric=npu_us_per_image');
// The trend lines, a line per batch.
fillSparks(card.cards, collect(historiesOf('sa-placer-hw', [hwBefore, hwAfter])));
const spark = card.cards[0].sparks.get('us_per_image');
assert.ok(spark.innerHTML.includes('var(--series-1)') && spark.innerHTML.includes('var(--series-2)'));
// The trend tabs are the view over nights: no batch-scaling snapshot chart.
assert.equal(all(card, n => n.className === 'charts').length, 0);
""")


def test_a_failed_run_is_named_by_seed_and_batch_with_its_repro(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
const failedAt = hwRecord('7', hwAfter.date, 104.9);
delete failedAt.rows['mobilenet/seed=7/batch=16/effort=default'];
failedAt.detail.find(d => d.seed === 7 && d.batch === 16 && d.effort_label === 'default').passed = false;
failedAt.failed = ['mobilenet/seed=7/batch=16/effort=default'];
const index = { target: 'sa-placer-hw', runs: [night('6', hwBefore.date, 'turbo'),
  night('7', hwAfter.date, 'turbo', { failed: ['mobilenet/seed=7/batch=16/effort=default'] })] };
const card = componentBand(hw, index, failedAt, NOW, hwBefore);
assert.equal(card.className, 'verdict bad');
assert.equal(one(card, n => n.className === 'headline').text, '1 run failed: seed 7 at batch 16, default effort.');
// A failure's repro is beside the verdict, outside Details.
assert.ok(card.children.some(c => c.className === 'repro'));
const seed7default = card.cards.find(c => c.kase === 'mobilenet/seed=7/effort=default');
const lanes = seed7default.children[1].children[0];
assert.equal(lanes.text, 'NPU time / imageb1248.7 µs 0.0%b16failed');
// No ratio from the failed batch's number of the night before.
assert.ok(!seed7default.children[0].text.includes('÷'));
const seed3default = card.cards.find(c => c.kase === 'mobilenet/seed=3/effort=default');
assert.ok(seed3default.children[0].text.includes('b16 ÷ b1'));
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


def test_a_compile_is_a_bar_of_its_stages_with_its_wall_clock_marked(page):
    page("""
// A real cold and warm compile of mobilenet, seed 3, default effort.
const latest = {
  compile_s: 20.6, compile_peak_rss_mb: 864.6, compile_place_s: 1.09, compile_route_s: 0.03, compile_lower_s: 0.45,
  compile_kernels_s: 15.85, compile_codegen_s: 0.19, compile_link_s: 0.24, compile_package_s: 0.07, compile_other_s: 0,
  compile_outside_aiecc_s: 2.79,
  compile_s_warm: 4.3, compile_peak_rss_mb_warm: 399.7, compile_place_s_warm: 0.73, compile_route_s_warm: 0.02,
  compile_lower_s_warm: 0.38, compile_kernels_s_warm: 0, compile_codegen_s_warm: 0.2, compile_link_s_warm: 0.21,
  compile_package_s_warm: 0.08, compile_other_s_warm: 0, compile_outside_aiecc_s_warm: 2.83,
};
const m = metric => metric in latest ? { value: latest[metric], unit: metric.includes('_rss_') ? 'MB' : 's' } : undefined;
const block = compileStages(m, metric => `hw|case|${metric}`, '#view=performance&metric=compile_s');
const [cold, warm, axis] = block.children.filter(r => r.className === 'stage-row');
const segs = row => row.children[1].children.filter(n => /^stage-seg /.test(n.className));
// Pipeline order; a stage that took no time has no segment.
assert.deepEqual(segs(cold).map(n => n.className), ['stage-seg st-place', 'stage-seg st-route', 'stage-seg st-lower', 'stage-seg st-kernels',
  'stage-seg st-codegen', 'stage-seg st-link', 'stage-seg st-package', 'stage-seg st-outside_aiecc']);
assert.deepEqual(segs(warm).map(n => n.className.slice(13)), ['place', 'route', 'lower', 'codegen', 'link', 'package', 'outside_aiecc']);
// One seconds axis for both bars, the longer compile's: 20.71 s of stages.
assert.equal(segs(cold)[3].style, 'width: max(1px, calc(76.533% - 2px))');
assert.equal(segs(warm)[0].style, 'width: max(1px, calc(3.525% - 2px))');
// Named in place only where the name fits; the textured time outside aiecc
// only in the key, since a label on the hatch would not read. Every segment
// can be hovered; none is a tab stop, since the key below tables every value.
assert.deepEqual(segs(cold).map(n => n.text), ['', '', '', 'waiting on kernels', '', '', '', '']);
assert.ok(segs(cold).every(n => n.tabIndex === undefined));
assert.equal(segs(cold)[0].title, 'place, cold compile: 1.09 s, 5.3% of its time per stage\\n' +
  'tile placement: aie-place-tiles, which runs the SA placer at its seed and effort');
assert.equal(segs(cold)[1].title.split('\\n')[0], 'route, cold compile: 0.03 s, 0.1% of its time per stage');
assert.equal(segs(cold)[6].title.split('\\n')[0], 'package, cold compile: 0.07 s, 0.3% of its time per stage');
// The whole compile's wall clock, marked on the same axis.
const wall = row => row.children[1].children.find(n => n.className === 'wall');
assert.equal(wall(cold).style, 'left: 99.469%');
assert.equal(wall(warm).title, 'the whole warm compile, wall clock: 4.30 s');
assert.equal(cold.children[2].text, '20.6 s · 864.6 MiB largest process');
assert.deepEqual(axis.children[1].children.map(n => n.text), ['0 s', '5 s', '10 s', '15 s', '20 s']);
// The key is the table of every stage's seconds, slivers included.
const key = block.children.find(n => n.className === 'stage-key');
const [head, body] = key.children;
assert.deepEqual(head.children[0].children.map(n => n.text),
  ['', 'place', 'route', 'lower', 'waiting on kernels', 'codegen', 'link', 'package', 'outside aiecc']);
assert.deepEqual(body.children.map(r => r.children.map(n => n.text)), [
  ['cold', '1.09 s', '0.03 s', '0.45 s', '15.8 s', '0.19 s', '0.24 s', '0.07 s', '2.79 s'],
  ['warm', '0.73 s', '0.02 s', '0.38 s', '0 s', '0.20 s', '0.21 s', '0.08 s', '2.83 s']]);
assert.equal(head.children[0].children[2].children[0].className, 'chip st-route');

// A compile without a stage profile keeps its total and has no bar, nor a wall line on nothing.
const coldOnly = metric => metric.endsWith('_warm') && metric !== 'compile_s_warm' ? undefined : m(metric);
const mixed = compileStages(coldOnly, metric => metric, '#');
const [, warmTotal] = mixed.children.filter(r => r.className === 'stage-row');
assert.deepEqual(warmTotal.children[1].children, []);
assert.deepEqual(mixed.children.find(n => n.className === 'stage-key').children[1].children[1].children.map(n => n.text),
  ['warm', '—', '—', '—', '—', '—', '—', '—', '—']);
// No compile at all: no block.
assert.equal(compileStages(() => undefined, metric => metric, '#'), null);
""")


def test_the_llama_card_reads_decode_rate_higher_as_better(page):
    page(COMPONENT_RUNS + LLAMA_NIGHTS + """
assert.equal(llama.section, 'performance');
const card = componentBand(llama, llamaIndex, llamaAfter, NOW, llamaBefore);
assert.equal(one(card, n => n.className === 'headline').text,
             'Every decoded token matched the oracle, but 1 measure got worse vs 1 Oct. 16 layers: a dispatch runs at 4.37 tok/s, 228.8 ms per token, 20.36× the DMA floor (compute-bound).');
// generate.py's end-to-end decode rate leads when the night has it; the dispatch rate follows.
assert.equal(llamaSummary({ [LLAMA]: { ...llamaAfter.rows[LLAMA], decode_tok_s: { value: 4.2, unit: 'tok/s' } } }),
             ' 16 layers decode at 4.2 tok/s end to end; a dispatch runs at 4.37 tok/s, 228.8 ms per token, 20.36× the DMA floor (compute-bound).');
// Fewer tokens per second is the regression; time per token repeats it and is not listed twice.
assert.equal(one(card, n => n.className === 'moves').text, 'Since 1 Oct:persist_grow, N 16 dispatch rate −7.0%');
assert.deepEqual(card.cards.map(c => c.kase), [LLAMA]);
const [header, grid] = card.cards[0].children;
// The layout is words on the card, not a metric.
assert.equal(header.text, '16 layerspersistent decode, 4 tokens per dispatchevery layer on one shared 8-core GEMV arraycompute-bound');
assert.deepEqual(grid.children.map(c => c.children[0].text), ['Decode tokens / s', 'Dispatch rate', 'Dispatch / token', 'Over the DMA floor',
  'Tokens verified', 'Compiled sizes']);
const chip = cell => all(cell, n => /^chg /.test(n.className || '')).map(n => n.className);
// Down is worse for tokens per second, up is worse for time per token.
assert.deepEqual(chip(grid.children[1]), ['chg worse']);
assert.deepEqual(chip(grid.children[2]), ['chg worse']);
assert.equal(grid.children[0].text, 'Decode tokens / s—');
assert.equal(grid.children[1].text, 'Dispatch rate4.37 tok/s −7.0%');
assert.equal(grid.children[5].text, 'Compiled sizesELF20.5 MiB ±0.0 KiBimage237.7 KiB ±0.0 KiB');
assert.deepEqual(card.cards[0].children[2].children.filter(r => r.className === 'stage-row').slice(0, 2).map(r => r.text),
  ['cold60.5 s 0.0% · 2,502.4 MiB largest process', 'warm1.2 s 0.0% · 410 MiB largest process']);
assert.equal(grid.children[1].href, '#view=performance&metric=tok_s');
fillSparks(card.cards, collect(historiesOf('llama-chain-hw', [llamaBefore, llamaAfter])));
assert.deepEqual(['compile_s', 'compile_s_warm'].map(m => card.cards[0].sparks.get(m).lines.map(([color, , dashed]) => [color, dashed])),
  [[['muted', false]], [['muted', true]]]);
assert.ok(card.cards[0].sparks.get('elf_bytes').innerHTML.includes('var(--series-2)'));
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
const verified = card.cards[0].children[1].children[4];
assert.equal(verified.children[1].children[0].className, 'vals worse');
const repro = one(card, n => n.className === 'repro');
assert.ok(!repro.open && card.children.includes(repro));
assert.deepEqual(one(repro, n => n.tag === 'pre').text.split('\\n'), [
  'git checkout abcdef123456',
  'cd programming_examples/ml/llama_3_2_1b && python aie2_decode.py --p-max=4 --kset=8 --n-layers=16 --t=128 --full-elf-path build/chain_persist_grow_N16_PT4.elf && python test_chain_persist_grow_mh.py --p-max=4 --kset=8 --n-layers=16 --t=128 --elf build/chain_persist_grow_N16_PT4.elf',
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


def test_the_yolo26n_card_shows_each_batch_and_what_it_verified(page):
    page(COMPONENT_RUNS + YOLO_NIGHTS + """
assert.equal(yolo.section, 'performance');
const card = componentBand(yolo, yoloIndex, yoloAfter, NOW, yoloBefore);
assert.equal(one(card, n => n.className === 'headline').text,
             'Every image matched the oracle, but 1 measure got worse vs 1 Oct. End to end, one image takes 5,400 µs (185.19 fps) at batch 1, 3,000 µs at batch 8.');
// The frame rate repeats the time per image and is not listed twice.
assert.equal(one(card, n => n.className === 'moves').text, 'Since 1 Oct:yolo26n end-to-end / image (b1) +8.0%');
assert.deepEqual(card.cards.map(c => c.kase), [YOLO]);
const [header, grid] = card.cards[0].children;
assert.equal(header.text, '640 × 6408 COCO images4 device configurations run in sequence in one full ELF, each fitting the 4×8 arrayb8 ÷ b1 per image 0.56×');
assert.deepEqual(grid.children.map(c => c.children[0].text), ['End-to-end / image', 'Images / s', 'Images verified', 'Match onnxruntime',
  'Compiled size']);
const chip = cell => all(cell, n => /^chg /.test(n.className || '')).map(n => n.className);
// A lane per batch; up is worse for time per image, down for images per second.
assert.equal(grid.children[0].text, 'End-to-end / imageb15,400 µs +8.0%b83,000 µs 0.0%');
assert.deepEqual(chip(grid.children[0]), ['chg worse', 'chg flat']);
assert.equal(grid.children[1].text, 'Images / sb1185.19 fps −7.4%b8333.33 fps 0.0%');
assert.deepEqual(chip(grid.children[1]), ['chg worse', 'chg flat']);
// A full-ELF build has one compiled size.
assert.equal(grid.children[4].text, 'Compiled sizeELF3.0 MiB ±0.0 KiB');
assert.equal(grid.children[0].href, '#view=performance&metric=us_per_image_b1');
assert.deepEqual(card.cards[0].children[2].children.filter(r => r.className === 'stage-row').slice(0, 2).map(r => r.text),
  ['cold95.5 s 0.0% · 3,100 MiB largest process', 'warm12.5 s 0.0% · 1,200 MiB largest process']);
fillSparks(card.cards, collect(historiesOf('yolo26n-hw', [yoloBefore, yoloAfter])));
const perImage = card.cards[0].sparks.get('us_per_image_b1').innerHTML;
assert.ok(perImage.includes('var(--series-1)') && perImage.includes('var(--series-2)'));
// One case: no batch chart.
assert.equal(all(card, n => n.className === 'charts').length, 0);
// A run that timed batch 1 alone says so, and its card has no ratio.
const b1Only = yoloRecord('7', yoloAfter.date, 78365.1);
for (const m of ['us_per_image_b8', 'fps_b8']) delete b1Only.rows[YOLO][m];
assert.equal(yoloSummary(b1Only.rows), ' End to end, one image takes 78,365.1 µs (12.76 fps) at batch 1.');
const alone = componentBand(yolo, yoloIndex, b1Only, NOW).cards[0];
assert.ok(!alone.children[0].text.includes('÷'));
assert.equal(alone.children[1].children[0].text, 'End-to-end / imageb178,365.1 µsb8—');
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
const verified = card.cards[0].children[1].children[2];
assert.equal(verified.children[1].children[0].className, 'vals worse');
const repro = one(card, n => n.className === 'repro');
assert.ok(!repro.open && card.children.includes(repro));
assert.deepEqual(one(repro, n => n.tag === 'pre').text.split('\\n'), [
  'git checkout abcdef123456',
  'cd programming_examples/ml/yolo26n && python yolo26n.py --images $IRON_EXAMPLE_WEIGHTS_DIR/yolo26n/coco128/images/train2017 --n 8',
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
assert.deepEqual(box.children[0].children[0].children.map(b => b.textContent),
                 ['Tokens / s', 'Dispatch / token', 'Compile time', 'Compile memory', 'Compile stages', 'Compiled size']);
const rate = trendConfig(llamaData, llama.trend[0]);
assert.deepEqual(rate.data.datasets.map(d => [d.label, d.data]), [['decode (generate.py)', [null, 4.2]], ['dispatch rate', [4.7, 4.37]]]);
// One line needs no legend; two do.
assert.equal(trendConfig(llamaData, llama.trend[1]).options.plugins.legend.display, false);
assert.equal(rate.options.plugins.legend.display, true);
// A tab's nights are those that published its metrics; the small stages fold into one line.
const stages = trendConfig(llamaData, COMPILE_TABS[2]);
assert.deepEqual(stages.data.datasets.map(d => [d.label, d.data]), [['place', [1]], ['route, package and other', [0.1]]]);
// Stages sit decades apart, so on a log axis; two batches' times do not.
assert.equal(stages.options.scales.y.type, 'logarithmic');
// YOLO26n: batch 1 beside batch 8, per image and per second, then the compile tabs.
const yoloData = collect(historiesOf('yolo26n-hw', [yoloBefore, yoloAfter]));
const perImage = trendConfig(yoloData, yolo.trend[0]);
assert.deepEqual(perImage.data.datasets.map(d => [d.label, d.data]), [['batch 1', [5000, 5400]], ['batch 8', [3000, 3000]]]);
assert.equal(perImage.options.scales.y.title.text, 'End-to-end time per image (µs, fastest call) · ↓ better');
assert.equal(trendConfig(yoloData, yolo.trend[1]).options.scales.y.title.text, 'Images per second (from the fastest call) · ↑ better');
assert.equal(perImage.options.scales.y.type, 'linear');
const yoloBox = trendBox(yoloData, yolo.trend, 'fps_b8');
assert.equal(yoloBox.metric(), 'fps_b1');
assert.ok(yoloBox.children[1].textContent.startsWith('Images / s: images per second at batch 1 and batch 8'), yoloBox.children[1].textContent);
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
assert.ok(card.cards[0].text.includes('295.7 µs'));
""")


def test_component_histories_are_charted_under_their_check(page):
    page(COMPONENT_RUNS + """
const fetched = [];
const runs = [{ id: '7', date: '2026-10-02T06:30:00Z', commit, pmode: null, provenance: { fixtures: 'abc' } }];
global.fetch = async url => {
  fetched.push(url);
  const check = url.split('/')[2];
  return { ok: true, json: async () => ({ schema: 1, target: check, metric: 'failed_seeds', unit: 'seeds', runs,
    series: check === 'sa-placer' ? { mobilenet: { values: [0] }, test_sa_effort: { values: [0] } } : { mobilenet: { values: [1] } } }) };
};
const loaded = [
  { check: sweep, index: { schema: 1, metrics: ['failed_seeds', 'final_cost_mean'] } },
  { check: hw, index: { schema: 1, metrics: ['failed_seeds', 'us_per_image'] } },
];
(async () => {
  let data = await componentHistories(loaded, 'us_per_image');
  assert.deepEqual(fetched, ['../component-checks/sa-placer-hw/history/us_per_image.json']);
  data = await componentHistories(loaded, 'failed_seeds');
  // Both checks have a mobilenet case; each keeps its own chart.
  assert.deepEqual(chartGroups(data.series).map(g => g.kase).sort(),
                   ['sa-placer-hw/mobilenet', 'sa-placer/mobilenet', 'sa-placer/test_sa_effort']);
  // No power mode recorded: one line, labelled by its check, drawn solid.
  const modes = modesOf(data);
  assert.deepEqual(modes, ['unknown']);
  draw(el0, chartGroups(data.series).find(g => g.kase === 'sa-placer-hw/mobilenet'), modes, data);
  assert.deepEqual(chart.data.datasets.map(d => [d.label, d.borderDash.length]), [['sa-placer-hw', 0]]);
  assert.equal(chart.options.plugins.legend.display, false);
  assert.deepEqual(chart.options.plugins.tooltip.callbacks.footer([{ dataIndex: 0 }]), ['sa-placer-hw: fixtures abc']);
  // A check's own histories, for its cards' trend lines, keep their cases.
  const own = await checkHistories(loaded[0]);
  assert.deepEqual(own.series.map(s => s.key).sort(), ['sa-placer|mobilenet|failed_seeds', 'sa-placer|test_sa_effort|failed_seeds']);
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
draw(el0, groupOf(across), ['turbo'], across);
assert.deepEqual(chart.data.labels, ['30 Dec 2025', '2 Jan 2026']);
""")


def test_the_placement_band_shows_each_seeds_placement_and_runs(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
assert.equal(hwPlacement.section, 'placement');
assert.equal(hw.section, 'performance');
const card = componentBand(hwPlacement, hwIndex, hwAfter, NOW, hwBefore);
assert.equal(card.className, 'verdict ok');
assert.equal(one(card, n => n.className === 'headline').text,
             'Every seed placed, compiled and verified. Placement cost 303 at seed 3 (default), 403 at seed 3 (full), 307 at seed 7 (default), 407 at seed 7 (full).');
// Only the section's metrics count as moves: seed 3's per-image regression is Performance's.
assert.equal(one(card, n => n.className === 'moves').text, 'Since 1 Oct:nothing moved past its threshold.');
assert.equal(card.cards.length, 1);
const [, grid] = card.cards[0].children;
assert.deepEqual(grid.children.map(c => c.children[0].text), ['Placement cost', 'Compile time (cold)', 'Placement', 'Runs', 'Failed seeds']);
assert.equal(grid.children[0].text, 'Placement costseed 3 (default)303 0.0%seed 3 (full)403 0.0%seed 7 (default)307 0.0%seed 7 (full)407 0.0%');
assert.equal(grid.children[2].text, 'Placementseed 3 (default)0123456789abseed 3 (full)0123456789abseed 7 (default)0123456789abseed 7 (full)0123456789ab');
assert.equal(grid.children[3].text, 'Runsseed 3 (default)2 of 2 passedseed 3 (full)2 of 2 passedseed 7 (default)2 of 2 passedseed 7 (full)2 of 2 passed');
assert.equal(grid.children[3].href, '#view=performance');
// Words-only cells have no trend line; the metrics do.
assert.deepEqual([...card.cards[0].sparks.keys()], ['placement_cost', 'compile_s', 'failed_seeds']);
// A failed run is named on its seed.
const failedAt = hwRecord('7', hwAfter.date, 104.9);
failedAt.detail.find(d => d.seed === 7 && d.batch === 16 && d.effort_label === 'default').passed = false;
failedAt.failed = ['mobilenet/seed=7/batch=16/effort=default'];
const bad = componentBand(hwPlacement, hwIndex, failedAt, NOW, hwBefore);
assert.equal(bad.cards[0].children[1].children[3].text,
             'Runsseed 3 (default)2 of 2 passedseed 3 (full)2 of 2 passedseed 7 (default)failed at b16seed 7 (full)2 of 2 passed');
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
assert.equal(router.section, 'routing');
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
// A card per device: its seeds, the verifier's kill rate, nonzero failures red.
assert.deepEqual(card.cards.map(c => c.kase), ['npu1', 'npu2']);
const [header, grid] = card.cards[0].children;
assert.ok(header.text.includes('seeds 20,370,000 + 2,000'));
assert.equal(header.children.at(-1).text, 'kill rate 1.000');
assert.deepEqual(grid.children.map(c => c.children[0].text), ['Failures', 'Unrouted / illegal', 'Missed corruptions', 'Route cost', 'Time / design']);
assert.equal(grid.children[0].text, 'Failuresproperties0mutations2');
assert.deepEqual(all(grid.children[0], n => /^vals/.test(n.className || '')).map(n => n.className), ['vals', 'vals worse']);
assert.equal(grid.children[3].text, 'Route costhops1.012×msels14.9');
assert.equal(grid.children[0].href, '#view=routing&metric=failed_properties');
const [, grid2] = card.cards[1].children;
assert.equal(all(grid2.children[1], n => /^vals/.test(n.className || ''))[1].className, 'vals worse');
// Run by hand with no failure: how to run the whole check.
const quiet = componentBand(router, { target: 'router', runs: [night('7', latest.date, null)] }, routerRecord('7', latest.date, 0), NOW);
assert.equal(one(quiet, n => n.className === 'headline').text,
             'Every property held and every mutation was answered right. npu2: 1.015× the constructed hops, 13.1 ms per design.');
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
const [header, grid] = card.cards[0].children;
assert.ok(header.text.includes('1 of 3 properties held'));
assert.deepEqual(grid.children.map(c => c.children[0].text), ['Failed properties', 'Unplaced / illegal', 'Quality', 'Wall time']);
assert.equal(grid.children[2].text, 'Qualitycrossings0sharing0.253 +4.1%');
// The counts both sections chart link to this one's trends.
assert.equal(grid.children[0].href, '#view=placement&metric=failed_properties');
const quiet = componentBand(allocator, { target: 'allocator', runs: [night('6', before.date, null)] }, before, NOW);
assert.ok(one(quiet, n => n.className === 'headline').text.endsWith('Bank sharing 0.243 (0 = most even split); the run took 103 s.'));
assert.equal(plural(2, 'property'), '2 properties');
assert.equal(plural(2, 'day'), '2 days');
""")


def test_router_and_allocator_cards_chart_every_metric(page):
    # Same spark mechanism as mobilenet's (placement_cost, us_per_image): one
    # per metric cell, filled from history by fillSparks. Here just that the
    # Map exists with the right keys; fillSparks itself is covered above.
    page(COMPONENT_RUNS + MODEL_NIGHTS + """
const routerIndex = { target: 'router', runs: [night('7', '2026-10-02T03:17:00Z', null)] };
const routerCard = componentBand(router, routerIndex, routerRecord('7', '2026-10-02T03:17:00Z', 0), NOW);
assert.deepEqual([...routerCard.cards[0].sparks.keys()],
                 ['failed_properties', 'unrouted', 'survived_corruptions', 'hop_ratio', 'ms_per_design']);
const allocatorIndex = { target: 'allocator', runs: [night('6', '2026-10-01T03:17:00Z', null)] };
const allocatorCard = componentBand(allocator, allocatorIndex, allocatorRecord('6', '2026-10-01T03:17:00Z', 0.2434), NOW);
assert.deepEqual([...allocatorCard.cards[0].sparks.keys()],
                 ['failed_properties', 'unsolved', 'avoidable_crossings', 'wall_s']);
""")


def test_the_overview_has_a_tile_per_section_and_says_when_one_is_late(page):
    page(MOVED + RUNS + COMPONENT_RUNS + HW_NIGHTS + """
const nightsNow = renderDashboard(['npu1'], db, new Map([['npu1', catalogue]]), new Map([['npu1', index]]), 3600 * 1000);
const loaded = [
  { check: hwPlacement, index: hwIndex, latest: hwAfter, previous: hwBefore },
  { check: hw, index: hwIndex, latest: hwAfter, previous: hwBefore },
];
const tiles = overviewTiles(nightsNow, loaded, NOW);
// In tab order: CI health, Kernels, then the sections, placement last.
assert.deepEqual(tiles.map(t => t.href), ['#view=ci', '#view=night', '#view=performance', '#view=routing', '#view=placement']);
// A row each: the name, one status word, one line, how long ago.
assert.deepEqual(tiles.map(t => t.children.map(c => c.className)[0]), ['area-name', 'area-name', 'area-name', 'area-name', 'area-name']);
assert.deepEqual(tiles.map(t => t.children[0].text), ['CI health', 'Kernels', 'Performance', 'Routing', 'Placement']);
const [ciTile, kernels, perf, routing, placement] = tiles;
// GitHub not asked yet.
assert.equal(ciTile.children[1].text, 'Loading…');
// A slower night is a warning on the overview too, not "All passing".
assert.equal(perf.level, 'warn');
assert.deepEqual(perf.children.slice(1, 3).map(c => c.text), ['Warnings', 'MobileNet: Every run passed, but 2 measures got worse vs 1 Oct.']);
// The whole of each line is in the row's tooltip.
assert.ok(perf.title.includes('MobileNet: Every run passed, but 2 measures got worse vs 1 Oct.'));
// A check that has not published yet: says so instead of a verdict.
assert.equal(routing.level, 'none');
assert.equal(routing.children[1].text, 'No results yet');
// Two days without a run: late and red, the line says so first.
const late = overviewTiles(null, loaded, NOW + 2 * 86400 * 1000);
assert.equal(late[2].level, 'bad');
assert.equal(late[2].children[2].text, 'MobileNet: No run for 2 days 1 h. Every run passed, but 2 measures got worse vs 1 Oct.');
assert.equal(late[1].children[1].text, 'Loading…');
// The band says it above its verdict.
const lateBand = componentBand(hw, hwIndex, hwAfter, NOW + 2 * 86400 * 1000, hwBefore);
assert.equal(one(lateBand, n => n.className === 'stale-line').text,
             'No run for 2 days 1 h: the numbers below are from 2 Oct, 06:30 UTC.');
assert.equal(one(lateBand, n => n.className === 'headline').text.slice(0, 16), 'Every run passed');
""")


def test_old_links_land_on_their_new_views(page):
    page("""
for (const [hash, view] of [['', 'overview'], ['#view=dashboard', 'night'], ['#view=catalogue', 'kernels'],
    ['#view=components', 'placement'], ['#view=components&metric=compile_s', 'performance'],
    ['#view=design', 'performance'], ['#view=nonsense', 'overview']]) {
  location.hash = hash;
  assert.equal(currentView(), view, hash);
}
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
// The trend chart's compile tab draws that series cold beside warm.
const trend = trendConfig(read, COMPILE_TABS[0]);
assert.deepEqual(trend.data.datasets.filter(d => !d.band).map(d => [d.label, d.data]), [['cold', [40, 40]], ['warm', [4.6, 4.6]]]);
// Charted like a kernel case: one chart per batch, the mean moved +3.5%, under 5%.
const groups = chartGroups(perImage);
assert.equal(groups.length, 2);
draw(el0, groups.find(g => g.kase === 'mobilenet/batch=16/effort=default'), ['turbo'], data);
assert.equal(chart.data.datasets.filter(d => d.band).length, 2);
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
assert.equal(summary, " Seed 3's NPU time is 248.7 µs per image at batch 1, 117.1 µs at batch 16 (full effort).");
// The trend chart draws the full-effort lines.
const trend = trendConfig(collect(historiesOf('sa-placer-hw', [before, after])), hw.trend.find(t => t.id === 'us_per_image'));
assert.deepEqual(trend.data.datasets.filter(d => !d.band).map(d => d.label), ['batch 1', 'batch 16']);
""")


def test_every_power_of_two_batch_gets_its_own_color(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
const batches = [1, 2, 4, 8, 16, 32, 64];
const rows = { mobilenet: { failed_seeds: { value: 0, unit: 'seeds' } } };
for (const b of batches) rows[`mobilenet/seed=3/batch=${b}/effort=default`] = { us_per_image: { value: 90 + 300 / b, unit: 'us' } };
const powers = { ...hwAfter, rows, detail: batches.map(batch => ({ seed: 3, batch, effort_label: 'default', passed: true, placement: '0123456789ab' })) };
const card = componentBand(hw, hwIndex, powers, NOW, hwBefore).cards[0];
// Seven lanes, seven colors: none wraps back onto batch 1's.
const perImage = card.children[1].children[0];
assert.deepEqual(perImage.children.slice(1, 8).map(r => r.children[0].className), batches.map((_, i) => `npu n-series-${i + 1}`));
// The trend draws batch 1 and batch 64 only, each in its card lane's color.
const powersOn = id => ({ ...powers, id, date: id === '6' ? hwBefore.date : hwAfter.date });
const trend = trendConfig(collect(historiesOf('sa-placer-hw', [powersOn('6'), powersOn('7')])), hw.trend.find(t => t.id === 'us_per_image'));
const lines = trend.data.datasets.filter(d => !d.band);
assert.deepEqual(lines.map(d => d.label), ['batch 1', 'batch 64']);
assert.deepEqual(lines.map(d => d.borderColor), [cssVar('--series-1', '#5b6475'), cssVar('--series-7', '#5b6475')]);
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
assert.deepEqual(ciRunLevel(run(1, 2, 'completed', 'cancelled'), NOW), { level: 'warn', text: 'cancelled' });
assert.deepEqual(ciRunLevel(run(1, 2, 'in_progress', null), NOW), { level: 'running', text: 'running' });
assert.deepEqual(ciRunLevel(run(1, 2, 'queued', null), NOW), { level: 'running', text: 'queued' });
// Queued past six hours: no runner with its labels is online.
assert.deepEqual(ciRunLevel(run(1, 30, 'queued', null), NOW), { level: 'warn', text: 'waiting for a runner for 1 day 6 h' });
// A nightly still going is judged by the run before it.
let v = ciVerdict(nightly, [run(2, 1, 'in_progress', null), run(1, 25, 'completed', 'failure')], [], NOW);
assert.deepEqual([v.level, v.headline], ['bad', 'Running now; the run before failed.']);
// A leg that never found a runner is named, and warns even when the run is still going.
v = ciVerdict(nightly, [run(2, 30, 'in_progress', null), run(1, 54, 'completed', 'success')],
  [job('Kernel checks (aie2-4col)', 30, 'completed', 'success'), job('Kernel checks (aie2p-8col)', 30, 'queued', null)], NOW);
assert.equal(v.level, 'warn');
assert.deepEqual(v.legs.map(l => [l.name, l.text]), [['Kernel checks (aie2p-8col)', 'waiting for a runner for 1 day 6 h']]);
// A job queued a short while is not listed: it is about to start.
assert.deepEqual(ciLegs([job('soon', 1, 'queued', null)], NOW), []);
// Late: no scheduled run for 1.5 of its period.
v = ciVerdict(nightly, [run(1, 40, 'completed', 'success')], [], NOW);
assert.deepEqual([v.level, v.late, v.headline], ['bad', true, 'No scheduled run for 1 day 16 h. Its last passed.']);
// Nothing on record.
assert.equal(ciVerdict(nightly, [], [], NOW).level, 'none');
""")


def test_nightly_jobs_are_a_row_each_with_their_recent_runs(page):
    page(CI_RUNS + """
const runs = [run(3, 6, 'completed', 'failure'), run(2, 30, 'completed', 'success'), run(1, 54, 'completed', 'failure')];
const row = nightlyRow({ wf: nightly, runs, jobs: [] }, NOW);
const [dot, name, state, strip] = row.children;
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
""")


def test_critical_workflows_judge_main_and_flakiness_not_pull_requests(page):
    page(CI_RUNS + """
// Healthy: 90% on main is the line, a third of PRs failing is not judged.
assert.deepEqual(criticalVerdict(crit(), NOW), { level: 'ok', notes: [], failedJobs: [], late: false });
// Main's latest run failed: the jobs that failed are named.
let v = criticalVerdict(crit({ latest_main: run(30, 3, 'completed', 'failure'),
  latest_main_jobs: [job('Linux hardware tests (aie2p-8col)', 3, 'completed', 'failure'), job('lint', 3, 'completed', 'success')] }), NOW);
assert.deepEqual(v, { level: 'bad', notes: ['its latest run on main failed (aie2p-8col)'], failedJobs: ['aie2p-8col'], late: false });
assert.equal(criticalTile(crit({ latest_main: run(30, 3, 'completed', 'failure'),
  latest_main_jobs: [job('build', 3, 'completed', 'failure')] }), NOW).text, 'Failed on main 3 h ago: build');
// Main passing less, and one commit in ten re-run to pass.
v = criticalVerdict(crit({ main: { passed: 8, failed: 2, rate: 0.8 }, flaky: { commits: 20, flaky: 3, rate: 0.15 } }), NOW);
assert.deepEqual(v.notes, ['passed 80% of its runs on main', '3 of 20 commits needed a re-run to pass']);
assert.equal(v.level, 'warn');
// Main runs it nightly: no run for 1.5 days means the nightly stopped.
v = criticalVerdict(crit({ every: 24, latest_main: run(30, 40, 'completed', 'success') }), NOW);
assert.deepEqual([v.level, v.late, v.notes[0]], ['bad', true, 'no run on main for 1 day 16 h']);
assert.equal(criticalTile(crit({ every: 24, latest_main: run(30, 40, 'completed', 'success') }), NOW).text, 'No run on main for 1 day 16 h');
// A run still going counts as recent.
assert.equal(criticalVerdict(crit({ every: 24, latest_main: run(30, 40, 'completed', 'success'), newest_main: run(31, 2, 'in_progress', null) }), NOW).late, false);
assert.equal(criticalVerdict({ file: 'x.yml', error: 'HTTP 404' }).level, 'none');
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
  ['none', 'The rest', '1 of 2 merge-gating tests healthy; 1 of 2 other scheduled jobs passed their last run.'],
  ['none', 'Merging', 'Median 26.4 h to merge; 42 open.'],
]);
const [tile] = overviewTiles(null, [], NOW, loaded);
assert.equal(tile.level, 'bad');
assert.equal(tile.children.at(-1).text, 'read 30 min ago');
// One line: who needs a look.
assert.equal(tile.children[2].text, '2 need a look: Linux hardware tests and Kernel checks.');
assert.equal(ciSummary(loadedOf(status()), NOW), 'All clear. 1 of 1 merge-gating test healthy; 1 of 1 other scheduled job passed their last run.');
// All healthy.
const fine = ciLines(loadedOf(status()), NOW);
assert.deepEqual(fine[0], { level: 'ok', who: 'All workflows', text: '1 of 1 merge-gating test healthy; 1 of 1 other scheduled job passed their last run.' });
// Nothing published: says so rather than "no results".
assert.equal(overviewTiles(null, [], NOW, loadedOf(null))[0].children[1].text, 'Not published yet');
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
assert.deepEqual([v.level, v.notes.at(-1)], ['warn', 'its newest run on main is waiting for a runner for 1 day 2 h']);
// "Failed on main" counts from the run's end, not its start.
const ended = { ...run(30, 3, 'completed', 'failure'), updated_at: at(1) };
assert.equal(criticalTile(crit({ latest_main: ended }), NOW).text, 'Failed on main 1 h ago');
// The run-time chart: a failed run red, a cancelled one amber.
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
assert.deepEqual([v.level, v.notes[0]], ['warn', 'no runner took its latest run on main']);
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


def test_ci_health_reads_at_a_glance(page):
    page(CI_RUNS + """
// All clear: one sentence, the counts beside it, how fresh.
let verdictBox = ciVerdictBand(loadedOf(status()), NOW);
assert.equal(verdictBox.className, 'verdict ok');
assert.equal(verdictBox.children[0].text, 'All clear. 1 of 1 merge-gating test healthy; 1 of 1 other scheduled job passed their last run.');
assert.equal(verdictBox.children.at(-1).text, 'Read from GitHub 6 Oct, 11:30 UTC by publishCiHealth.yml, which runs every six hours.');
// Problems: how many, each a line linked to its run, worst first, then the rest.
const loaded = loadedOf(status({
  critical: [crit({ latest_main: run(30, 3, 'completed', 'failure') }), crit({ file: 'py.yml', title: 'Python versions', main: { passed: 8, failed: 2, rate: 0.8 } })],
  workflows: [{ ...nightly, runs: [run(2, 30, 'in_progress', null)], jobs: [job('Kernel checks (aie2p-8col)', 30, 'queued', null)] }],
}));
verdictBox = ciVerdictBand(loaded, NOW);
assert.equal(verdictBox.className, 'verdict bad');
assert.equal(verdictBox.children[0].text, '3 workflows need a look.');
assert.deepEqual(verdictBox.children[1].children.map(li => [li.className, li.text]), [
  ['bad', 'Linux hardware testsIts latest run on main failed.'],
  ['warn', 'Python versionsPassed 80% of its runs on main.'],
  ['warn', 'Kernel checksRunning now. aie2p-8col: waiting for a runner for 1 day 6 h.'],
]);
assert.equal(verdictBox.children[1].children[0].children[1].children[0].children[0].href, 'https://example.com/runs/30');
// The rest is one line under the verdict, not in it.
assert.equal(ciHealthyLine(loaded, NOW), '0 of 1 other scheduled job passed their last run.');
assert.equal(ciHealthyLine(loadedOf(status()), NOW), '');
// Nothing published.
assert.equal(ciVerdictBand(loadedOf(null), NOW).children[0].text, 'No CI health published yet.');
// The board: a tile a workflow, one state each.
const groups = ciBoard(loaded, NOW);
// The board is the merge-gating tests; the scheduled jobs are rows below it.
assert.deepEqual(groups.map(g => g.children[0].text), ['Tests that gate a merge']);
const tiles = groups.flatMap(g => g.children[1].children);
assert.deepEqual(tiles.map(t => [t.className, t.text]), [
  ['ci-tile bad', 'Linux hardware testsFailed on main 3 h ago'],
  ['ci-tile warn', 'Python versionsPassed 80% of its runs on main'],
]);
assert.equal(nightlyRow(loaded.results[0], NOW).children[2].text, 'Waiting for a runner: aie2p-8col');
assert.equal(boardTile(criticalTile(crit(), NOW)).text, 'Linux hardware testsPassing: 90% on main, 1 h 24 min a run');
assert.equal(scheduledTile({ wf: nightly, runs: [run(5, 6, 'completed', 'success')], jobs: [] }, NOW).text, 'Passed 6 h ago');
assert.equal(scheduledTile({ wf: nightly, runs: [run(5, 40, 'completed', 'success')], jobs: [] }, NOW).text, 'No run for 1 day 16 h');
assert.equal(scheduledTile({ wf: nightly, runs: null, error: 'HTTP 502' }, NOW).text, 'GitHub did not answer');
assert.equal(jobName('Kernel checks (aie2p-8col)', 'Kernel checks'), 'aie2p-8col');
assert.equal(jobName('build-and-test (aie2p-8col)', 'Linux hardware tests'), 'build-and-test (aie2p-8col)');
""")


def test_performance_is_one_chart_a_line_per_batch(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
const data = collect(historiesOf('sa-placer-hw', [hwBefore, hwAfter]));
const config = trendConfig(data, hw.trend.find(t => t.id === 'us_per_image'));
// A line per batch, the mean over the seeds, each with its seed range shaded.
const lines = config.data.datasets.filter(d => !d.band);
// The caption says so; a compile tab's lines are each the one compile that serves every batch.
const caption = m => trendBox(data, hw.trend, m).children[1].textContent;
assert.ok(caption('us_per_image').includes('. A line for the smallest and the largest batch, each the mean over the seeds at default effort;'), caption('us_per_image'));
assert.ok(caption('compile_s').startsWith('Compile time: wall time of the whole compile'), caption('compile_s'));
assert.ok(caption('compile_s').includes('. A single compile serves every batch: the mean over the seeds'), caption('compile_s'));
assert.deepEqual(lines.map(d => d.label), ['batch 1', 'batch 16']);
assert.deepEqual(lines[1].data, [98.5, 101.95]);
assert.equal(config.data.datasets.filter(d => d.band).length, 4);
// One point per nightly, by date.
assert.deepEqual(config.data.labels, ['1 Oct', '2 Oct']);
// The legend names the batches, not the shading.
const legend = config.options.plugins.legend.labels.filter;
assert.deepEqual(config.data.datasets.map((d, datasetIndex) => legend({ datasetIndex })), config.data.datasets.map(d => !d.band));
// Per-image times on a log axis; the tooltip gives the seed range.
assert.equal(config.options.scales.y.type, 'logarithmic');
assert.equal(config.options.plugins.tooltip.callbacks.label({ dataset: lines[1], dataIndex: 1, raw: 101.95 }),
             'batch 16: 102 µs (seeds 99 µs to 104.9 µs)');
// Both axes say what they are, the y axis with its unit.
assert.equal(config.options.scales.y.title.text, 'Dispatch time per image (µs, fastest of 20) · ↓ better');
// Runs older than 90 days are thinned to weekly: the axis says so.
assert.equal(config.options.scales.x.title.text, 'Nightly run (weekly past 90 days)');
// Hover names the run's commit; a click opens it.
const tips = config.options.plugins.tooltip.callbacks;
assert.equal(tips.afterTitle([{ dataIndex: 1 }]), 'abcdef1 same revision');
assert.equal(tips.footer([{ dataIndex: 1 }]), 'Click for the commit');
config.options.onClick(null, [{ index: 1 }]);
assert.deepEqual(opened, ['https://github.com/Xilinx/mlir-aie/commit/abcdef123456', '_blank', 'noopener']);
assert.equal(commitLine({ id: '0123456789abcdef', message: 'x'.repeat(100) + '\\nbody' }), `0123456 ${'x'.repeat(71)}…`);
// Compile time, seconds, on a plain axis, cold beside warm, each shaded over the seeds.
const compile = trendConfig(data, COMPILE_TABS[0]);
assert.equal(compile.options.scales.y.type, 'linear');
assert.equal(compile.options.scales.y.title.text, 'Compile time (s, one compile each) · ↓ better');
assert.deepEqual(compile.data.datasets.filter(d => !d.band).map(d => d.label), ['cold', 'warm']);
// Compile peak memory, MB, also on a plain axis.
const memory = trendConfig(data, COMPILE_TABS[1]);
assert.equal(memory.options.scales.y.type, 'linear');
assert.equal(memory.options.scales.y.title.text, 'Largest process peak (MiB, one compile each) · ↓ better');
// The box: a switch for the tab, drawn once on screen, redrawn on a switch.
// A link to a warm metric opens the tab that draws it; a tab with no history is left out.
const box = trendBox(data, hw.trend, 'compile_s_warm');
assert.equal(box.metric(), 'compile_s');
const seg = box.children[0].children[0];
// NPU time and host overhead lead (the author's view), then the dispatch they
// split and the whole call, then the compile.
assert.deepEqual(seg.children.map(b => b.value),
  ['npu_us_per_image', 'host_overhead_us_per_image', 'us_per_image', 'e2e_us_per_image', 'compile_s', 'compile_peak_rss_mb']);
assert.deepEqual(seg.children.map(b => b.textContent),
  ['NPU time', 'Host overhead', 'Dispatch / image', 'End-to-end / image', 'Compile time', 'Compile memory']);
assert.equal(chart.options.scales.y.type, 'linear');
seg.children[0].on.click();
assert.equal(box.metric(), 'npu_us_per_image');
assert.deepEqual(chart.data.datasets.filter(d => !d.band).map(d => d.label), ['batch 1', 'batch 16']);
assert.equal(chart.options.scales.y.title.text, 'NPU time per image (µs, fastest of 20 less the empty dispatch) · ↓ better');
assert.equal(chart.options.scales.y.type, 'logarithmic');
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
const drawn = svgChart(canvas, trendConfig(data, hw.trend.find(t => t.id === 'us_per_image')));
// A line and a shaded band per batch, the legend naming the batches.
assert.deepEqual(drawn.legend, ['batch 1', 'batch 16']);
assert.equal(drawn.svg.filter(p => p.startsWith('<polyline')).length, 2);
assert.equal(drawn.svg.filter(p => p.startsWith('<polygon')).length, 2);
// The axis titles, as the Chart.js chart has them.
assert.ok(drawn.svg.some(p => p.includes('rotate(-90)') && p.includes('Dispatch time per image (µs, fastest of 20) · ↓ better')));
assert.ok(drawn.svg.some(p => p.includes('>Nightly run (weekly past 90 days)<')));
// Each point: hover text with the commit, and a link to it.
assert.ok(drawn.svg.some(p => p.includes('<title>2 Oct, 06:30 UTC\\nabcdef1 same revision\\nbatch 16: ') && p.includes('href="https://github.com/Xilinx/mlir-aie/commit/abcdef123456"')));
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


def test_the_change_table_compares_last_night_by_default_or_the_window(page):
    page(RUNS + """
// Case a moved last night only (100, 90, 100); case b over the window only
// (100, 120, 121: +0.8% last night, under the 2% threshold); case c moved,
// then was not measured last night, so it has no last-night change.
const day = d => 86400000 * d;
const db3 = fromRecords({ npu1: [
  rec('1', day(10), 'turbo', { 'a/1/i8/cycles': ['cycles', 100], 'b/1/i8/cycles': ['cycles', 100], 'c/1/i8/cycles': ['cycles', 100] }),
  rec('2', day(20), 'turbo', { 'a/1/i8/cycles': ['cycles', 90], 'b/1/i8/cycles': ['cycles', 120], 'c/1/i8/cycles': ['cycles', 150] }),
  rec('3', day(21), 'turbo', { 'a/1/i8/cycles': ['cycles', 100], 'b/1/i8/cycles': ['cycles', 121] }),
] });
const shown = () => $('window-changes').children[0].children[0];
const header = () => shown().children[0].children[0].children[2].text;
const rowsOf = rendered => rendered.map(r => r.children.map(c => c.text));
// Last night, by default: the latest nightly against the one before it.
location.hash = '#view=night';
assert.deepEqual(rowsOf(renderWindowChanges(db3, ['npu1'])), [['a/1/i8', 'npu1', '90 → 100 (+11.1%)', '—']]);
assert.equal(header(), 'Cycles21 Jan → 22 Jan (change)');
assert.ok($('window-about').text.startsWith("Each case's cycles and object size on its latest nightly, up to 22 Jan, against the nightly before it, 21 Jan for most cases."));
const buttons = $('window-compare').children;
assert.deepEqual(buttons.map(b => [b.textContent, b.className]), [['Last night', 'on'], ['Window', '']]);
// The window: its first nightly against the latest, the choice in the hash.
buttons[1].on.click();
assert.equal(location.hash, '#view=night&changes=window');
assert.deepEqual(rowsOf(renderWindowChanges(db3, ['npu1'])), [['c/1/i8', 'npu1', '100 → 150 (+50.0%)', '—'], ['b/1/i8', 'npu1', '100 → 121 (+21.0%)', '—']]);
assert.equal(header(), 'Cycles11 Jan → 22 Jan (change)');
assert.ok($('window-about').text.startsWith("Each case's cycles and object size on its first nightly from 11 Jan against its latest, up to 22 Jan: 3 nightlies."));
assert.deepEqual($('window-compare').children.map(b => b.className), ['', 'on']);
// Back from the hash, and an unknown value is last night.
location.hash = '#view=night&changes=last';
assert.deepEqual(rowsOf(renderWindowChanges(db3, ['npu1'])).map(r => r[0]), ['a/1/i8']);
location.hash = '#view=night&changes=bogus';
assert.deepEqual(rowsOf(renderWindowChanges(db3, ['npu1'])).map(r => r[0]), ['a/1/i8']);
// One night only: nothing to compare yet, in either mode.
const one = fromRecords({ npu1: [rec('1', day(10), 'turbo', { 'a/1/i8/cycles': ['cycles', 100] })] });
assert.deepEqual(renderWindowChanges(one, ['npu1']), []);
assert.equal($('window-about').text, "Each case's latest nightly will be compared with an earlier one once there are two.");
// Nothing moved last night: said so in its terms.
const quiet = fromRecords({ npu1: [rec('1', day(10), 'turbo', { 'a/1/i8/cycles': ['cycles', 100] }),
                                   rec('2', day(11), 'turbo', { 'a/1/i8/cycles': ['cycles', 100] })] });
location.hash = '';
assert.deepEqual(renderWindowChanges(quiet, ['npu1']), []);
assert.equal($('window-changes').text, 'Nothing moved past its threshold since the nightly before.');
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
assert.deepEqual(head.children.map(c => c.text), ['Case', 'NPU', 'Cycles1 Jan → 1 Jan (change)', 'Object size1 Jan → 1 Jan (change)']);
assert.deepEqual(rows.map(r => r.children.map(c => c.text)), [
  ['mm 2 cases', 'npu2', '—', '2.0 KiB → 4.3 KiB (+115.0%)2 of 2 cases moved'],
  ['swiglu/1/bf16', 'npu2', '—', '8.8 KiB → 16.3 KiB (+85.6%)'],
]);
// Hover gives each case.
assert.equal(rows[0].children[3].title, 'mm/1/f32/gelu: 2.0 KiB → 4.3 KiB (+115.0%)\\nmm/1/f32/relu: 2.0 KiB → 4.2 KiB (+110.0%)');
// A measure that did not move is greyed: size grew, cycles flat.
const both = fromRecords({ npu2: [
  rec('1', 1, 'turbo', { 'k/1/i8/cycles': ['cycles', 100], 'k/1/i8/kernel_object_bytes': ['bytes', 2048] }),
  rec('2', 2, 'turbo', { 'k/1/i8/cycles': ['cycles', 101], 'k/1/i8/kernel_object_bytes': ['bytes', 4096] }),
] });
const [row] = renderWindowChanges(both, ['npu2']);
assert.deepEqual(row.children.slice(2).map(c => [c.className, c.text]),
  [['num quiet', '100 → 101 (+1.0%)'], ['num', '2.0 KiB → 4.0 KiB (+100.0%)']]);
assert.equal(rows[0].children[0].children[0].title, 'mm/1/f32/gelu\\nmm/1/f32/relu');
""")
