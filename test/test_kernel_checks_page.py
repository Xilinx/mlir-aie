# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Exercise the page's real data, chart and table code under node, without a browser."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

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
global.fetch = async () => ({ ok: false });
// Just enough DOM for the page to build its cards and tables.
class El {
  constructor(tag) { this.tag = tag; this.children = []; this.on = {}; this.hidden = false; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  addEventListener(type, f) { this.on[type] = f; }
  querySelectorAll() { return []; }
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
        subprocess.run(
            [node, "-"], input=source, encoding="utf-8", check=True, env=env
        )

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
assert.equal(changeClass(0.019, 'cycles'), '');
assert.equal(changeClass(0.02, 'cycles'), 'worse');
assert.equal(changeClass(-0.02, 'kernel_object_bytes'), 'better');
assert.equal(changeClass(0.09, 'npu_us'), '');
assert.equal(changeClass(0.1, 'npu_us'), 'worse');
assert.equal(changeClass(0.14, 'compile_s'), '');
assert.equal(changeClass(0.15, 'compile_s'), 'worse');
assert.equal(changeClass(0.009, 'final_cost_mean'), '');
assert.equal(changeClass(0.01, 'final_cost_mean'), 'worse');
assert.deepEqual(GATED, ['cycles', 'kernel_object_bytes']);
""")


def test_narrow_screens_hide_no_kernels_table_column():
    # Each NPU's header spans its columns; hiding one slides the headers.
    css = (
        PAGE.read_text(encoding="utf-8").split("<style>", 1)[1].split("</style>", 1)[0]
    )
    rules = re.findall(r"#kernels[^{]*\{[^}]*display:\s*none", css)
    assert all(".summary" in r for r in rules), rules


def test_repeated_commits_stay_separate_runs(page):
    page("""
assert.equal(db.order.length, 7);
assert.equal(new Set(db.order.map(p => p.id)).size, 7);
assert.deepEqual(db.order.filter(o => o.npu === 'npu1').map(o => o.run), ['t1', 'p1', 't2', 't25', 'p25', 't3']);
assert.deepEqual(s.byMode.get('turbo').map(p => p.row.value), [100, 110, 121]);
assert.equal(s.factory, 'softmax');
assert.equal(latestChange(s, ['turbo']), 0.1);
assert.equal(latestChange(s, ['turbo', 'performance']), 0.1);
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
const cb = chart.options.plugins.tooltip.callbacks;
assert.deepEqual(cb.title([{ dataIndex: 5 }]), ['1970-01-01 00:05 UTC']);
assert.equal(cb.afterTitle([{ dataIndex: 5 }]), 'abcdef1 same revision');
assert.equal(cb.label({ dataset: lines[0], dataIndex: 5, raw: 121 }), 'npu1 turbo: 121 cycles');
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
assert.equal(latestChange(db.series[0], ['performance']), null);
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
             'npu1: 40 us (± 2.5; min 38.0 max 45.0 n=50)');
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
// The run without provenance is skipped; kernel digests are not marked.
assert.deepEqual(markersFor(db.order), [
  { index: 2, label: 'peano 22.0.0+bbbb' },
  { index: 3, label: 'host bench-2, runtime XRTHostRuntime, xrt 2.20.0' },
]);
draw(el0, groupOf(db), ['performance'], db);
assert.deepEqual(chart.options.plugins.markers.at, [
  { index: 2, label: 'npu1: peano 22.0.0+bbbb' },
  { index: 3, label: 'npu1: host bench-2, runtime XRTHostRuntime, xrt 2.20.0' },
]);
const footer = chart.options.plugins.tooltip.callbacks.footer;
// What changed is in the tooltip, not drawn over the chart.
assert.equal(chart.options.plugins.tooltip.callbacks.title([{ dataIndex: 2 }])[1], 'changed: npu1: peano 22.0.0+bbbb');
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
assert.deepEqual(big.metrics.get('cycles'), { value: 110, unit: 'cycles', change: 0.1, delta: 10, cls: 'worse', range: 'median 112 max 130 n=16' });
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
                 [{ level: 'bad', text: 'no run since 1970-01-01 00:00 UTC', stale: true }]);
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
assert.equal(headline.text, '1 case slower, 1 failing correctness, 1 failed timing. 1 faster.');
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
  '1 Jan, 00:00 UTC', 'abcdef1 same revision', 'Peano 22.0.0+bbbb', 'XRT 2.20.0, driver 2.20.0_1',
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
assert.equal(items[2].children[1].text, 'No results have been published.');

// A regression is named in Needs a look with how far it moved; its kernel a click away.
const worse = items[1];
assert.ok(worse.text.startsWith('npu11 case got slower or bigger past its threshold:'));
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
// The Summary's table: case, NPU, measure, then and now with their dates, the change.
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
assert.deepEqual(kernelWindow(db, 'npu1', 'softmax'), { cycles: [50, 1100], since: 1, worse: 1, better: 0 });
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
assert.equal(config.options.scales.y.title.text, 'Cycles per call');
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
assert.deepEqual(move, { change: 0.25, delta: 25, cls: '' });
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
assert.deepEqual(grid.children.map(c => c.children[0].text), ['Failed seeds', 'Placement cost', 'CPU time', label('peak_rss_mb_max')]);
assert.equal(grid.children[0].children[1].children[1].className, 'vals worse');
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
assert.equal(moves.text, 'Since 1 Oct:mobilenet cost (worst) +10.0%mobilenet cost (mean) +3.3%');
const [header, grid] = card.cards[0].children;
assert.ok(header.text.includes('utils/component_checks/fixtures/mobilenet.mlir'));
assert.ok(header.text.includes('4 seeds'));
assert.equal(header.children.at(-1).text, 'worst ÷ mean cost 1.06×');
const cost = grid.children[1];
assert.equal(cost.text, 'Placement costmean103.3 +3.3%worst110 +10.0%');
assert.deepEqual(all(cost, n => /^chg /.test(n.className || '')).map(n => n.className), ['chg worse strong', 'chg worse strong']);
assert.equal(all(grid.children[2], n => /^chg /.test(n.className || ''))[0].className, 'chg flat');
assert.equal(cost.href, '#view=placement&metric=final_cost_mean');
// The Seeds chart: a bar per seed, the nightly before as ticks, seed 4 a ✕.
const [bars, ticks, crosses] = chart.data.datasets;
assert.deepEqual(chart.data.labels, ['1', '2', '3', '4']);
assert.deepEqual(bars.data, [100, 110, 100, null]);
assert.deepEqual(ticks.data, [100, 100, 100, 100]);
assert.deepEqual(crosses.data.map(v => v !== null), [false, false, false, true]);
assert.equal(crosses.data[3], chart.options.scales.y.min);
assert.equal(chart.options.plugins.tooltip.callbacks.afterBody([{ dataIndex: 1 }]), 'CPU 2,100 ms, peak 200 MB');
assert.equal(chart.options.plugins.tooltip.callbacks.label({ dataset: crosses, raw: 0 }), 'did not place');
// The trend lines: a line each for the mean and the worst cost.
fillSparks(card.cards, collect(historiesOf('sa-placer', [before, after])));
const spark = card.cards[0].sparks.get('final_cost_mean');
assert.ok(spark.innerHTML.includes('var(--series-1)') && spark.innerHTML.includes('var(--series-2)'));
""")


HW_NIGHTS = """
// Two hardware nightlies, seeds 3 and 7 at batch 1 and 16: seed 3's
// batch 16 is 7% slower in the latest, past its 5% and its MADs.
const hwRecord = (id, date, b16, extra) => {
  const rows = {};
  for (const seed of [3, 7]) {
    rows[`mobilenet/seed=${seed}`] = { placement_cost: { value: 300 + seed, unit: 'cost', range: `placement 0123456789ab` } };
    for (const [batch, us] of [[1, 295.7], [16, seed === 3 ? b16 : 99]]) {
      rows[`mobilenet/seed=${seed}/batch=${batch}`] = {
        us_per_image: { value: us, unit: 'us', range: `median ${us + 1}, MAD 0.20, p95 ${us + 2}` },
        e2e_us_per_image: { value: us + 100, unit: 'us' },
        compile_s: { value: 40, unit: 's' },
        ...(batch > 1 ? { streaming_us: { value: (us * 16 - 295.7) / 15, unit: 'us' } } : {}),
      };
    }
  }
  rows.mobilenet = { failed_seeds: { value: 0, unit: 'seeds' } };
  const detail = [3, 7].flatMap(seed => [1, 16].map(batch => ({ seed, batch, passed: true, placement: '0123456789ab' })));
  return { id, date, pmode: 'turbo', commit: { ...commit }, provenance: {}, rows, detail, ...(extra || {}) };
};
const hwBefore = hwRecord('6', '2026-10-01T06:30:00Z', 98);
const hwAfter = hwRecord('7', '2026-10-02T06:30:00Z', 104.9);
const hwIndex = { target: 'sa-placer-hw', runs: [night('6', hwBefore.date, 'turbo'), night('7', hwAfter.date, 'turbo')] };
"""


def test_a_passing_check_says_so_and_how_to_run_it(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
const card = componentBand(hw, hwIndex, hwAfter, NOW, hwBefore);
assert.equal(card.className, 'verdict ok');
assert.equal(one(card, n => n.className === 'headline').text,
             'Every run passed. Seed 3 streams at 92.2 µs per image; one image takes 295.7 µs.');
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
assert.ok(one(repro, n => n.tag === 'pre').text.includes('python -m mobilenet.aie2_mobilenet_iron --sa-seed N --sa-effort 1.0 --batch B'));
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
             'Since 1 Oct:seed 3, batch 16 streaming +8.7%seed 3, batch 16 per image +7.0%');
assert.deepEqual(card.cards.map(c => c.kase), ['mobilenet/seed=3', 'mobilenet/seed=7']);
const [header, grid] = card.cards[0].children;
assert.equal(header.children[0].text, "seed 3the design's default");
// The placement, its cost under Placement.
assert.equal(header.children[1].text, 'placement 0123456789ab');
assert.equal(header.children[1].href, '#view=placement&metric=placement_cost');
assert.equal(header.children.at(-1).text, 'b16 ÷ b1 per image 0.35×');
assert.deepEqual(grid.children.map(c => c.children[0].text), ['Per image', 'Streaming', 'End-to-end / image', 'Compile time']);
// A lane per batch, colored past the threshold only; streaming has no b1.
const perImage = grid.children[0];
assert.equal(perImage.text, 'Per imageb1295.7 µs 0.0%b16104.9 µs +7.0%');
assert.deepEqual(perImage.children.slice(1, 3).map(r => r.children[0].className), ['npu n-series-1', 'npu n-series-2']);
assert.deepEqual(all(perImage, n => /^chg /.test(n.className || '')).map(n => n.className), ['chg flat', 'chg worse']);
assert.equal(grid.children[1].children.length, 3);
assert.equal(grid.children[3].text, 'Compile timeb140 s 0.0%b1640 s 0.0%');
assert.equal(perImage.href, '#view=performance&metric=us_per_image');
// The trend lines, a line per batch.
fillSparks(card.cards, collect(historiesOf('sa-placer-hw', [hwBefore, hwAfter])));
const spark = card.cards[0].sparks.get('us_per_image');
assert.ok(spark.innerHTML.includes('var(--series-1)') && spark.innerHTML.includes('var(--series-2)'));
// The batch scaling chart: a line per seed, the nightly before dashed and out of the legend.
const sets = chart.data.datasets;
assert.deepEqual(sets.map(d => [d.label, d.borderDash ? d.borderDash.length : 0]), [
  ['seed 3', 0], ['seed 3, the nightly before', 2], ['seed 7', 0], ['seed 7, the nightly before', 2]]);
assert.deepEqual(sets[0].data, [{ x: 1, y: 295.7 }, { x: 16, y: 104.9 }]);
assert.equal(chart.options.scales.x.type, 'logarithmic');
const axis = {};
chart.options.scales.x.afterBuildTicks(axis);
assert.deepEqual(axis.ticks.map(t => t.value), [1, 16]);
assert.equal(chart.options.scales.x.ticks.callback(16), 'b16');
assert.deepEqual(sets.map((d, datasetIndex) => chart.options.plugins.legend.labels.filter({ datasetIndex })), [true, false, true, false]);
assert.equal(chart.options.plugins.tooltip.callbacks.label({ dataset: sets[0], parsed: { x: 16, y: 104.9 } }),
             'seed 3: 104.9 µs per image, 9,533 images/s');
// A batch placed apart from the first is noted on its seed's card.
const apart = hwRecord('7', hwAfter.date, 104.9);
apart.detail[1].placement = 'fedcba987654';
const noted = componentBand(hw, hwIndex, apart, NOW, hwBefore).cards[0].children[0];
assert.ok(noted.text.includes('placed differently at batch 16'));
""")


def test_a_failed_run_is_named_by_seed_and_batch_with_its_repro(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
const failedAt = hwRecord('7', hwAfter.date, 104.9);
delete failedAt.rows['mobilenet/seed=7/batch=16'];
failedAt.detail[3].passed = false;
failedAt.failed = ['mobilenet/seed=7/batch=16'];
const index = { target: 'sa-placer-hw', runs: [night('6', hwBefore.date, 'turbo'),
  night('7', hwAfter.date, 'turbo', { failed: ['mobilenet/seed=7/batch=16'] })] };
const card = componentBand(hw, index, failedAt, NOW, hwBefore);
assert.equal(card.className, 'verdict bad');
assert.equal(one(card, n => n.className === 'headline').text, '1 run failed: seed 7 at batch 16.');
// A failure's repro is beside the verdict, outside Details.
assert.ok(card.children.some(c => c.className === 'repro'));
const lanes = card.cards[1].children[1].children[0];
assert.equal(lanes.text, 'Per imageb1295.7 µs 0.0%b16failed');
// No ratio from the failed batch's number of the night before.
assert.ok(!card.cards[1].children[0].text.includes('÷'));
assert.ok(card.cards[0].children[0].text.includes('b16 ÷ b1'));
const repro = one(card, n => n.className === 'repro');
// Beside the verdict, shut: a click away.
assert.ok(!repro.open && card.children.includes(repro));
assert.deepEqual(one(repro, n => n.tag === 'pre').text.split('\\n').slice(0, 3), [
  'git checkout abcdef123456',
  'cd programming_examples/ml && python -m mobilenet.aie2_mobilenet_iron --sa-seed 7 --sa-effort 1.0 --batch 16',
  '# every run, as the nightly runs them:',
]);
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


def test_component_values_read_as_counts_and_costs(page):
    page("""
assert.equal(formatValue(1, 'seeds'), '1 seed');
assert.equal(formatValue(0, 'seeds'), '0 seeds');
assert.equal(formatValue(12345.678, 'cost'), '12,345.68');
assert.equal(formatValue(295.7, 'us'), '295.7 us');
""")


def test_the_placement_band_shows_each_seeds_placement_and_runs(page):
    page(COMPONENT_RUNS + HW_NIGHTS + """
assert.equal(hwPlacement.section, 'placement');
assert.equal(hw.section, 'performance');
const card = componentBand(hwPlacement, hwIndex, hwAfter, NOW, hwBefore);
assert.equal(card.className, 'verdict ok');
assert.equal(one(card, n => n.className === 'headline').text,
             'Every seed placed, compiled and verified. Placement cost 303 at seed 3, 307 at seed 7.');
// Only the section's metrics count as moves: seed 3's per-image regression is Performance's.
assert.equal(one(card, n => n.className === 'moves').text, 'Since 1 Oct:nothing moved past its threshold.');
assert.equal(card.cards.length, 1);
const [, grid] = card.cards[0].children;
assert.deepEqual(grid.children.map(c => c.children[0].text), ['Placement cost', 'Placement', 'Runs', 'Failed runs']);
assert.equal(grid.children[0].text, 'Placement costseed 3303 0.0%seed 7307 0.0%');
assert.equal(grid.children[1].text, 'Placementseed 30123456789abseed 70123456789ab');
assert.equal(grid.children[2].text, 'Runsseed 32 of 2 passedseed 72 of 2 passed');
assert.equal(grid.children[2].href, '#view=performance');
// Words-only cells have no trend line; the metrics do.
assert.deepEqual([...card.cards[0].sparks.keys()], ['placement_cost', 'failed_seeds']);
// A failed run is named on its seed.
const failedAt = hwRecord('7', hwAfter.date, 104.9);
failedAt.detail[3].passed = false;
failedAt.failed = ['mobilenet/seed=7/batch=16'];
const bad = componentBand(hwPlacement, hwIndex, failedAt, NOW, hwBefore);
assert.equal(bad.cards[0].children[1].children[2].text, 'Runsseed 32 of 2 passedseed 7failed at b16');
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
assert.equal(perf.level, 'ok');
assert.deepEqual(perf.children.slice(1, 3).map(c => c.text), ['All passing', 'Every run passed.']);
// The whole of each line is in the row's tooltip.
assert.ok(perf.title.includes('MobileNet: Every run passed.'));
// No checks: says so instead of a verdict.
assert.equal(routing.level, 'none');
assert.equal(routing.children[1].text, 'Not tracked yet');
// Two days without a run: late and red, the line says so first.
const late = overviewTiles(null, loaded, NOW + 2 * 86400 * 1000);
assert.equal(late[2].level, 'bad');
assert.equal(late[2].children[2].text, 'MobileNet: No run for 2 days 1 h. Every run passed.');
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
const perImage = data.series.filter(s => s.metric === 'us_per_image');
// A series per batch, not per seed and batch.
assert.deepEqual(perImage.map(s => s.kase).sort(), ['mobilenet/batch=1', 'mobilenet/batch=16']);
const b16 = perImage.find(s => s.kase === 'mobilenet/batch=16');
// Seeds 3 and 7 at batch 16: (98 + 99) / 2, then (104.9 + 99) / 2; one point per run.
assert.deepEqual(b16.byMode.get('turbo').map(p => p.row.value), [98.5, 101.95]);
assert.equal(b16.byMode.get('turbo')[1].row.range, 'seeds 99 to 104.9, n=2');
// The band runs from the fastest seed to the slowest.
assert.deepEqual(spreadOf(101.95, b16.byMode.get('turbo')[1].row), [99, 104.9]);
// Series that are not per seed and batch pass through.
assert.ok(data.series.some(s => s.kase === 'mobilenet/seed=3' && s.metric === 'placement_cost'));
// Charted like a kernel case: one chart per batch, the mean moved +3.5%, under 5%.
const groups = chartGroups(perImage);
assert.equal(groups.length, 2);
draw(el0, groups.find(g => g.kase === 'mobilenet/batch=16'), ['turbo'], data);
assert.equal(chart.data.datasets.filter(d => d.band).length, 2);
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
assert.equal(config.data.datasets[0].pointBackgroundColor[2], cssVar('--bad', '#c62f3a'));
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
const config = seedTrendConfig(data, 'us_per_image');
// A line per batch, the mean over the seeds, each with its seed range shaded.
const lines = config.data.datasets.filter(d => !d.band);
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
assert.equal(config.options.scales.y.title.text, 'Time per image (µs)');
assert.equal(config.options.scales.x.title.text, 'Nightly run');
// Hover names the run's commit; a click opens it.
const tips = config.options.plugins.tooltip.callbacks;
assert.equal(tips.afterTitle([{ dataIndex: 1 }]), 'abcdef1 same revision');
assert.equal(tips.footer([{ dataIndex: 1 }]), 'Click for the commit');
config.options.onClick(null, [{ index: 1 }]);
assert.deepEqual(opened, ['https://github.com/Xilinx/mlir-aie/commit/abcdef123456', '_blank', 'noopener']);
assert.equal(commitLine({ id: '0123456789abcdef', message: 'x'.repeat(100) + '\\nbody' }), `0123456 ${'x'.repeat(71)}…`);
// Compile time, seconds, on a plain axis.
assert.equal(seedTrendConfig(data, 'compile_s').options.scales.y.type, 'linear');
assert.equal(seedTrendConfig(data, 'compile_s').options.scales.y.title.text, 'Compile time (s)');
// The box: a switch for the measure, drawn once on screen, redrawn on a switch.
const box = seedTrendBox(data, 'compile_s');
assert.equal(box.metric(), 'compile_s');
const seg = box.children[0].children[0];
assert.deepEqual(seg.children.map(b => b.value), ['us_per_image', 'streaming_us', 'e2e_us_per_image', 'compile_s']);
assert.equal(chart.options.scales.y.type, 'linear');
seg.children[0].on.click();
assert.equal(box.metric(), 'us_per_image');
assert.equal(chart.options.scales.y.type, 'logarithmic');
assert.equal(seedTrendBox(collect([]), '').text, 'No history published yet.');
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
const drawn = svgChart(canvas, seedTrendConfig(data, 'us_per_image'));
// A line and a shaded band per batch, the legend naming the batches.
assert.deepEqual(drawn.legend, ['batch 1', 'batch 16']);
assert.equal(drawn.svg.filter(p => p.startsWith('<polyline')).length, 2);
assert.equal(drawn.svg.filter(p => p.startsWith('<polygon')).length, 2);
// The axis titles, as the Chart.js chart has them.
assert.ok(drawn.svg.some(p => p.includes('rotate(-90)') && p.includes('Time per image (µs)')));
assert.ok(drawn.svg.some(p => p.includes('>Nightly run<')));
// Each point: hover text with the commit, and a link to it.
assert.ok(drawn.svg.some(p => p.includes('<title>2 Oct, 06:30 UTC\\nabcdef1 same revision\\nbatch 16: ') && p.includes('href="https://github.com/Xilinx/mlir-aie/commit/abcdef123456"')));
assert.ok(canvas.hidden);
assert.equal(outer.children[0].className, 'note svg-legend');
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
