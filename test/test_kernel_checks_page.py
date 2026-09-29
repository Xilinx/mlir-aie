# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Exercise the page's real data, chart and table code under node, without a browser."""

import json
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
        pytest.skip("node is required to test the kernel checks page")
    script = PAGE.read_text().split("<script>", 1)[1].split("</script>", 1)[0]
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
  getElementById: id => byId.get(id) || byId.set(id, new El()).get(id),
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
"""
    data = """
const commit = {
  id: 'abcdef123456', message: 'same revision\\nbody',
  timestamp: '2020-01-01T00:00:00Z', url: 'https://github.com/Xilinx/mlir-aie/commit/abcdef123456',
};
const entry = (date, value, extra) => ({
  commit, date,
  benches: value === null ? [] : [
    { name: 'softmax/1024x16/bfloat16/cycles', unit: 'cycles', value, ...(extra ? { extra } : {}) },
  ],
});
db = collect([
  ['npu1', { entries: {
    'aie_kernels (npu1, turbo)': [
      entry(300000, 121), entry(100000, 100), entry(200000, null), entry(250000, 110),
    ],
    'aie_kernels (npu1, performance)': [entry(100000, 90), entry(250000, 95)],
  }}],
  ['npu2', { entries: { 'aie_kernels (npu2, turbo)': [entry(100000, 999)] } }],
]);
modeColor = new Map([['turbo', '#0969da'], ['performance', '#cf222e']]);
const s = db.series.find(s => s.npu === 'npu1');
const el0 = {};
const row = (name, unit, value, range, extra) => ({ name, unit, value, ...(range ? { range } : {}), ...(extra ? { extra } : {}) });
"""

    def run(checks):
        subprocess.run([node, "-e", setup + script + data + checks], check=True)

    return run


def test_thresholds_match_the_report_and_color_only_past_them(page):
    html = PAGE.read_text()
    match = re.search(r"^\s*const THRESHOLDS = (\{.*\});$", html, re.MULTILINE)
    assert match, "the page carries no THRESHOLDS constant"
    shared = json.loads((ROOT / "utils/kernel_checks/thresholds.json").read_text())
    expected = {m: s for m, s in shared.items() if not m.startswith("_")}
    assert json.loads(match[1]) == expected
    page("""
assert.equal(changeClass(0.019, 'cycles'), '');
assert.equal(changeClass(0.02, 'cycles'), 'worse');
assert.equal(changeClass(-0.02, 'core_elf_bytes'), 'better');
assert.equal(changeClass(0.09, 'npu_us'), '');
assert.equal(changeClass(0.1, 'npu_us'), 'worse');
assert.equal(changeClass(0.05, 'compile_s'), '');
assert.equal(changeClass(0.1, 'compile_s'), 'worse');
assert.deepEqual(GATED, ['cycles', 'core_elf_bytes']);
""")


def test_repeated_sha_observations_are_not_deduplicated(page):
    page("""
assert.equal(db.order.length, 7);
assert.equal(new Set(db.order.map(p => p.id)).size, 7);
assert.deepEqual(s.byMode.get('turbo').map(p => p.row.value), [100, 110, 121]);
assert.equal(s.factory, 'softmax');
assert.equal(latestChange(s, ['turbo']), 0.1);
assert.equal(latestChange(s, ['turbo', 'performance']), 0.1);
assert.deepEqual(provenanceOf('commit abc | peano 22.0.0+e1 | host bench 3'),
                 { commit: 'abc', peano: '22.0.0+e1', host: 'bench 3' });
assert.deepEqual(provenanceOf(undefined), {});
""")


def test_chart_preserves_history_gaps_and_separate_npus(page):
    page("""
draw(el0, s, ['turbo', 'performance']);
assert.deepEqual(chart.data.labels, Array(6).fill('01-01'));
assert.deepEqual(chart.data.datasets[0].data, [100, null, null, 110, null, 121]);
assert.deepEqual(chart.data.datasets[1].data, [null, 90, null, null, 95, null]);
const cb = chart.options.plugins.tooltip.callbacks;
assert.equal(cb.title([{ dataIndex: 5 }]), '1970-01-01 00:05 UTC');
assert.equal(cb.afterTitle([{ dataIndex: 5 }]), 'abcdef1 same revision');
assert.equal(cb.label({ dataset: chart.data.datasets[0], dataIndex: 5, formattedValue: '121' }), 'turbo: 121 cycles');
chart.options.onClick(null, [{ index: 5 }]);
assert.deepEqual(opened, [commit.url, '_blank', 'noopener']);
assert.equal(chart.options.scales.y.title.text, 'cycles');
assert.deepEqual(chart.options.plugins.markers.at, []);
draw(el0, db.series.find(s => s.npu === 'npu2'), ['turbo']);
assert.deepEqual(chart.data.datasets[0].data, [999]);
""")


def test_missing_npu_and_filtered_mode_preserve_repeated_runs(page):
    page("""
db = collect([['npu1', { entries: {
  'aie_kernels (npu1, turbo)': [entry(100000, 100), entry(200000, 110)],
}}]]);
draw(el0, db.series[0], ['performance', 'turbo']);
assert.equal(chart.data.datasets.length, 1);
assert.deepEqual(chart.data.datasets[0].data, [100, 110]);
assert.equal(latestChange(db.series[0], ['performance']), null);
""")


def test_npu_us_deviation_is_a_band_outside_the_legend_and_tooltip(page):
    page("""
const timed = (date, value, range) => ({ commit, date, benches: [
  row('softmax/1024x16/bfloat16/npu_us', 'us', value, range),
]});
db = collect([['npu1', { entries: { 'aie_kernels (npu1, turbo)': [
  timed(100000, 50, 'min 40.0 max 60.0 n=50'),
  timed(200000, 40, '± 2.5; min 38.0 max 45.0 n=50'),
]}}]]);
draw(el0, db.series[0], ['turbo']);
const [line, low, high] = chart.data.datasets;
assert.equal(chart.data.datasets.length, 3);
assert.deepEqual(line.data, [50, 40]);
assert.deepEqual(low.data, [null, 37.5]);
assert.deepEqual(high.data, [null, 42.5]);
assert.equal(high.fill, '-1');
assert.equal(low.backgroundColor, 'rgba(9, 105, 218, 0.2)');
const data = chart.data;
assert.deepEqual([0, 1, 2].map(i => chart.options.plugins.legend.labels.filter({ datasetIndex: i }, data)),
                 [true, false, false]);
assert.deepEqual([0, 1, 2].map(i => chart.options.plugins.tooltip.filter({ dataset: data.datasets[i] })),
                 [true, false, false]);
assert.equal(chart.options.plugins.tooltip.callbacks.label({ dataset: line, dataIndex: 1, formattedValue: '40' }),
             'turbo: 40 us (± 2.5; min 38.0 max 45.0 n=50)');
""")


def test_provenance_changes_are_marked_and_shown_in_the_footer(page):
    page("""
const a = 'commit 111 | peano 22.0.0+aaaa | kernels k1 | host bench-1 | pmode performance';
const b = 'commit 222 | peano 22.0.0+bbbb | kernels k2 | host bench-1 | pmode performance';
const c = 'commit 333 | peano 22.0.0+bbbb | kernels k3 | host bench-2 | xrt 2.20.0 | pmode performance';
db = collect([['npu1', { entries: { 'aie_kernels (npu1, performance)': [
  entry(100000, 5, a), entry(200000, null), entry(300000, 6, b), entry(400000, 7, c),
]}}]]);
// The empty run has no provenance and is skipped; kernel digests are not marked.
assert.deepEqual(markersFor(db.order), [
  { index: 2, label: 'peano 22.0.0+bbbb' },
  { index: 3, label: 'host bench-2, xrt 2.20.0' },
]);
draw(el0, db.series[0], ['performance']);
assert.deepEqual(chart.options.plugins.markers.at.map(m => m.index), [2, 3]);
const footer = chart.options.plugins.tooltip.callbacks.footer;
assert.deepEqual(footer([{ dataset: chart.data.datasets[0], dataIndex: 3 }]), c.split(' | '));
assert.equal(footer([{ dataset: chart.data.datasets[0], dataIndex: 1 }]), '');
""")


def test_latest_cases_follow_the_latest_nightly(page):
    page("""
db = collect([['npu1', { entries: {
  'aie_kernels (npu1, turbo)': [
    { commit, date: 1, benches: [
      row('softmax/1024/bfloat16/cycles', 'cycles', 100),
      row('softmax/64/bfloat16/cycles', 'cycles', 50),
    ]},
    { commit, date: 3, benches: [
      row('softmax/1024/bfloat16/cycles', 'cycles', 110, 'median 112 max 130 n=16'),
      row('softmax/1024/bfloat16/core_elf_bytes', 'bytes', 4096),
    ]},
  ],
  'aie_kernels (npu1, performance)': [
    { commit, date: 2, benches: [row('softmax/1024/bfloat16/cycles', 'cycles', 1)] },
  ],
}}]]);
const { last, cases } = latestCases(db, 'npu1');
assert.equal(last.mode, 'turbo');
const big = cases.get('softmax/1024/bfloat16');
assert.ok(big.current);
assert.deepEqual(big.metrics.get('cycles'), { value: 110, unit: 'cycles', change: 0.1, range: 'median 112 max 130 n=16' });
assert.equal(big.metrics.get('core_elf_bytes').change, null);
assert.ok(!cases.get('softmax/64/bfloat16').current);
assert.equal(latestCases(db, 'npu2').cases.size, 0);
""")


def test_cycle_spread_and_truncation_are_read_from_the_range(page):
    page("""
assert.deepEqual(cyclesSpread('median 2939 max 2990 n=84; truncated'),
                 { median: 2939, max: 2990, n: 84, truncated: true });
assert.deepEqual(cyclesSpread('median 264 max 264 n=16'),
                 { median: 264, max: 264, n: 16, truncated: false });
assert.deepEqual(cyclesSpread('median 5580 max 5580 n=3; init[2] min 16; truncated'),
                 { median: 5580, max: 5580, n: 3, truncated: true });
assert.equal(cyclesSpread('± 1.6; min 172.8 max 202.2 n=50'), null);
assert.equal(cyclesSpread(undefined), null);

db = collect([['npu1', { entries: { 'aie_kernels (npu1, performance)': [
  { commit, date: 1, benches: [
    row('swiglu/1024x256/bfloat16/cycles', 'cycles', 2875, 'median 2939 max 2990 n=84; truncated'),
    row('add/1024x16/bfloat16/cycles', 'cycles', 78, 'median 87 max 131 n=16'),
  ]},
]}}]]);
const { cases } = latestCases(db, 'npu1');
const cell = caseCell(undefined, cases.get('swiglu/1024x256/bfloat16'));
assert.equal(cell.text, '2,875 cycles truncated');
assert.equal(cell.children[0].children[1].className, 'fail');
assert.ok(cell.children[0].title.includes('min 2875, median 2939, max 2990 over 84 calls'));
const plain = caseCell(undefined, cases.get('add/1024x16/bfloat16'));
assert.equal(plain.text, '78 cycles');
assert.ok(plain.children[0].title.includes('median 87, max 131 over 16 calls'));
assert.equal(caseCell('failing', undefined).text, 'failing');
assert.equal(caseCell('timing failed', undefined).className, 'fail');
assert.equal(caseCell('untimed', undefined).text, 'passed, not timed');
assert.equal(caseCell(undefined, undefined).text, '—');
""")


MOVED = """
db = collect([['npu1', { entries: { 'aie_kernels (npu1, performance)': [
  { commit, date: 1, benches: [
    row('relu/1024/bf16/cycles', 'cycles', 1000), row('relu/1024/bf16/cycles_per_kop', 'cycles/1k-ops', 10),
    row('relu/1024/bf16/npu_us', 'us', 100), row('relu/1024/bf16/core_elf_bytes', 'bytes', 4096),
    row('gelu/1024/bf16/cycles', 'cycles', 2000), row('gone/1/i8/cycles', 'cycles', 10),
    row('flat/1/i8/cycles', 'cycles', 500),
  ]},
  { commit, date: 2, benches: [
    row('relu/1024/bf16/cycles', 'cycles', 1030), row('relu/1024/bf16/cycles_per_kop', 'cycles/1k-ops', 10.3),
    row('relu/1024/bf16/npu_us', 'us', 150), row('relu/1024/bf16/core_elf_bytes', 'bytes', 4096),
    row('gelu/1024/bf16/cycles', 'cycles', 1500), row('flat/1/i8/cycles', 'cycles', 505),
    row('fresh/1/i8/cycles', 'cycles', 5),
  ]},
]}}]]);
"""


def test_moved_series_use_the_thresholds_and_lead_with_gated_metrics(page):
    page(MOVED + """
const moved = movedSeries(db, 'npu1');
// cycles_per_kop is derived, a 1% ELF change is under threshold, gone/ has
// no latest point, fresh/ no previous one, flat/ moved 1%.
assert.deepEqual(moved.regressed.map(x => [x.series.kase, x.series.metric, x.before, x.after]), [
  ['relu/1024/bf16', 'cycles', 1000, 1030],
  ['relu/1024/bf16', 'npu_us', 100, 150],
]);
assert.deepEqual(moved.improved.map(x => [x.series.kase, x.series.metric]), [['gelu/1024/bf16', 'cycles']]);
assert.equal(movedSeries(db, 'npu2').regressed.length, 0);
""")


def test_latest_run_prefers_the_recorded_run_and_falls_back_to_the_series(page):
    page(MOVED + """
const catalogue = { npu: 'npu1', arch: 'aie2', commit: 'abcdef123456', date: '2020-01-01T00:00:03Z', kernels: [] };
const index = { npu: 'npu1', runs: [
  { id: '1', url: 'https://example.com/runs/1', date: '2020-01-01T00:00:01Z', commit: 'abcdef123456', pmode: 'performance',
    device: 'NPU Strix', provenance: { peano: '22.0.0+aaaa', host: 'bench-1' }, sane: true, published: true, failed: [], truncated: [] },
  { id: '2', url: 'https://example.com/runs/2', date: '2020-01-01T00:00:02Z', commit: 'abcdef123456', pmode: 'default',
    device: 'NPU Strix', provenance: { peano: '22.0.0+bbbb', host: 'bench-2', xrt: '2.20.0' }, sane: true, published: true,
    failed: ['test_kernels_perf.py::test_kernel_perf[mm/64/i8]'], truncated: ['swiglu/1024x256/bfloat16'] },
]};
const run = latestRun('npu1', index, db, catalogue);
assert.equal(run.recorded, true);
assert.equal(run.url, 'https://example.com/runs/2');
assert.equal(run.date, Date.parse('2020-01-01T00:00:02Z'));
assert.equal(run.commit, 'abcdef123456');
assert.equal(run.commitUrl, commit.url);
assert.equal(run.message, 'same revision');
assert.equal(run.pmode, 'default');
assert.equal(run.device, 'NPU Strix');
assert.equal(run.previous.id, '1');
const now = Date.parse('2020-01-01T01:00:00Z');
assert.deepEqual(warningsFor('npu1', run, now).map(w => [w.level, w.text.split(':')[0]]), [
  ['warn', 'measured in power mode "default", not performance'],
  ['warn', '1 test failed in this run'],
  ['warn', '1 case with a truncated trace'],
  ['info', 'peano changed since the previous run'],
  ['warn', 'host changed since the previous run'],
  ['warn', 'xrt changed since the previous run'],
]);
// Stale, unsane runs say so first.
const bad = latestRun('npu1', { npu: 'npu1', runs: [{ ...index.runs[0], sane: false, published: false }] }, db, catalogue);
assert.deepEqual(warningsFor('npu1', bad, now + 48 * 3600 * 1000).map(w => w.level), ['bad', 'bad']);

// Without runs.json, the series' latest observation stands in.
const fallback = latestRun('npu1', null, db, catalogue);
assert.equal(fallback.recorded, false);
assert.equal(fallback.url, null);
assert.equal(fallback.date, 2);
assert.equal(fallback.commit, 'abcdef123456');
assert.equal(fallback.pmode, 'performance');
assert.deepEqual(warningsFor('npu1', fallback, now), [{ level: 'bad', text: 'no run since 1970-01-01 00:00 UTC' }]);
// Nothing at all.
const none = latestRun('npu2', null, db, null);
assert.equal(none.date, null);
assert.deepEqual(warningsFor('npu2', none, now).map(w => w.level), ['bad']);
""")


def test_dashboard_cards_and_regression_rows(page):
    page(MOVED + """
const catalogue = { npu: 'npu1', arch: 'aie2', commit: 'abcdef123456', date: '2020-01-01T00:00:03Z', kernels: [
  { factory: 'relu', family: 'activation', summary: 'ReLU', sources: ['activation/relu.cc'], builds: ['relu'],
    passed: 3, failed: [], timed: 1, timing_failed: ['relu/2048/bf16'], untimed: ['relu/64/bf16'] },
  { factory: 'mm', family: 'linalg', summary: 'mm', sources: [], builds: ['mm'], passed: 0, failed: ['mm/64/i8'], timed: 0 },
  { factory: 'exp2f_vec', family: 'activation', summary: 'npu2 only', sources: [], builds: [], passed: 0, failed: [], timed: 0 },
]};
const index = { npu: 'npu1', runs: [
  { id: '2', url: 'https://example.com/runs/2', date: '2020-01-01T00:00:02Z', commit: 'abcdef123456', pmode: 'performance',
    device: 'NPU Strix', provenance: { peano: '22.0.0+bbbb', host: 'bench-2', xrt: '2.20.0', xdna: '2.20.0_1', kernels: 'k9' },
    sane: true, published: true, failed: [], truncated: [] },
]};
renderDashboard(['npu1', 'npu2'], db, new Map([['npu1', catalogue]]), new Map([['npu1', index]]), Date.parse('2020-01-01T01:00:00Z'));
const [card1, card2] = $('cards').children;
assert.equal(card1.children[0].text, 'npu1 · aie2 · NPU Strix');
const dl = card1.children.find(c => c.tag === 'dl');
const terms = dl.children.filter((_, i) => i % 2 === 0).map(c => c.text);
assert.deepEqual(terms, ['Run', 'Commit', 'Power mode', 'Peano', 'XRT / driver', 'Host', 'Kernel sources']);
const values = dl.children.filter((_, i) => i % 2 === 1).map(c => c.text);
assert.deepEqual(values, ['2020-01-01 00:00 UTC', 'abcdef1 same revision', 'performance', '22.0.0+bbbb', '2.20.0 / 2.20.0_1', 'bench-2', 'k9']);
assert.equal(dl.children[1].children[0].href, 'https://example.com/runs/2');
const counts = card1.children.find(c => c.className === 'counts');
assert.equal(counts.children[0].text, 'Cases: 3 passed · 1 failing · 1 timed · 1 timing failed · 1 correctness only');
assert.equal(counts.children[1].text, 'Kernels: 2 of 2 offered checked on hardware · kernels');
assert.equal(counts.children[2].text,
  'Since the previous nightly: 1 regression in cycles, core_elf_bytes, 1 other series worse, 1 improved · charts, worst first');
assert.equal(counts.children[2].children.find(c => c.tag === 'a').href, '#view=charts&npu=npu1&metric=all&kernel=&sort=worse');
// No warnings on a clean run.
assert.ok(!card1.children.some(c => c.className === 'warnings'));
// npu2 has nothing yet.
assert.equal(card2.children[0].text, 'npu2');
assert.equal(card2.children[1].children[0].text, 'npu2: no results have been published');

const rows = $('regressions').children;
assert.deepEqual(rows.map(r => r.children.map(c => c.text)), [
  ['npu1', 'relu/1024/bf16 chart', 'cycles', '1,000 cycles', '1,030 cycles', '+3.0%'],
  ['npu1', 'relu/1024/bf16 chart', 'npu_us', '100 us', '150 us', '+50.0%'],
]);
assert.equal(rows[0].children[1].children[0].href, '#view=kernel&kernel=relu');
assert.equal(rows[0].children[1].children[3].href, '#view=charts&npu=npu1&metric=cycles&kernel=relu%2F1024%2Fbf16');
assert.equal(rows[0].children[5].className, 'worse');
assert.equal($('regressions-about').textContent, "2 series moved past its threshold in the latest nightly's power mode; cycles and core_elf_bytes first.");
assert.equal($('improvements-summary').textContent, 'Improvements (1)');
assert.equal($('improvements-box').hidden, false);
""")


def test_kernels_view_groups_cases_under_their_factory(page):
    page("""
db = collect([['npu1', { entries: { 'aie_kernels (npu1, turbo)': [
  { commit, date: 1, benches: [
    row('softmax/1024/bfloat16/cycles', 'cycles', 1000),
    row('softmax/64/bfloat16/cycles', 'cycles', 50),
  ]},
  { commit, date: 2, benches: [
    row('softmax/1024/bfloat16/cycles', 'cycles', 1100),
    row('softmax/1024/bfloat16/core_elf_bytes', 'bytes', 4096),
    row('softmax_mask/8/bfloat16/cycles', 'cycles', 7),
  ]},
]}}]]);
const kernel = (factory, extra) => ({
  factory, family: 'activation', summary: factory, sources: [`activation/${factory}.cc`],
  builds: [factory], passed: 1, failed: [], timed: 1, ...extra,
});
renderKernels([{ npu: 'npu1', arch: 'aie2', commit: 'abcdef1', date: '2020-01-01T00:00:00Z', kernels: [
  kernel('softmax', { failed: ['softmax/2048/bfloat16'], timing_failed: ['softmax/16/bfloat16'],
                      untimed: ['softmax/32/bfloat16'] }),
  kernel('relu', { builds: [], passed: 0, timed: 0 }),
  kernel('cascade_mm', { passed: 0, timed: 0, reason: 'no case' }),
]}], db);
assert.equal($('kernels-status').textContent, 'npu1 (aie2): abcdef1, 2020-01-01 00:00 UTC, turbo mode');
const rows = $('kernels-rows').children;
const name = r => r.children[r.className === 'case' ? 0 : 1];
assert.deepEqual(rows.map(r => name(r).text), [
  'cascade_mm', 'relu', 'softmax5 cases',
  'softmax/1024/bfloat16', 'softmax/16/bfloat16', 'softmax/2048/bfloat16',
  'softmax/32/bfloat16', 'softmax/64/bfloat16',
]);
assert.equal(name(rows[2]).children[0].href, '#view=kernel&kernel=softmax');
assert.equal(rows[2].children[3].children[0].href, 'https://github.com/Xilinx/mlir-aie/blob/abcdef1/aie_kernels/activation/softmax.cc');
const npu = r => r.children[r.children.length - 1];
assert.equal(npu(rows[0]).text, '1 build · not checked on hardware');
assert.equal(npu(rows[0]).children[1].title, 'no case');
assert.equal(npu(rows[1]).text, '—');
assert.equal(npu(rows[2]).text, '1 build · 1 failing · 1 timing failed · charts');
assert.equal(npu(rows[2]).children[2].title, 'softmax/16/bfloat16');
assert.equal(npu(rows[3]).text, '1,100 cycles (+10.0%) · 4.0 KiB');
assert.equal(npu(rows[3]).children[0].children[1].className, 'worse');
assert.equal(npu(rows[4]).text, 'timing failed');
assert.equal(npu(rows[5]).text, 'failing');
assert.equal(npu(rows[6]).text, 'passed, not timed');
assert.equal(npu(rows[7]).className, 'stale');
assert.equal(npu(rows[7]).text, '50 cycles');
assert.equal(rows[3].children[0].children[0].href,
  '#view=charts&npu=npu1&metric=all&kernel=softmax%2F1024%2Fbfloat16');

assert.ok(rows.slice(3).every(r => r.hidden));
rows[2].children[1].children[2].on.click();
assert.ok(rows.slice(3).every(r => !r.hidden));
rows[2].children[1].children[2].on.click();
assert.ok(rows.slice(3).every(r => r.hidden));

$('kernels-filter').value = '2048';
$('kernels-filter').on.input();
assert.deepEqual(rows.map(r => r.hidden), [true, true, false, true, true, false, true, true]);
$('kernels-filter').value = '';
$('kernels-filter').on.input();
assert.deepEqual(rows.map(r => r.hidden), [false, false, false, true, true, true, true, true]);
""")


def test_kernels_view_counts_untimed_cases(page):
    page("""
renderKernels([{ npu: 'npu1', arch: 'aie2', commit: 'abcdef1', date: '2020-01-01T00:00:00Z', kernels: [
  { factory: 'zero', family: 'zero', summary: 'zero', sources: ['common/zero.h'], builds: ['zero'],
    passed: 4, failed: [], timed: 0, untimed: ['zero/64/int32', 'zero/64/bfloat16'] },
]}], null);
const first = $('kernels-rows').children[0];
const cell = first.children[first.children.length - 1];
assert.equal(cell.text, '1 build · 4 cases pass');
assert.equal(cell.children[1].title, '2 cases not timed by design');
""")


def test_kernel_detail_lists_every_metric_and_draws_its_charts(page):
    page(MOVED + """
const catalogues = [{ npu: 'npu1', arch: 'aie2', commit: 'abcdef123456', date: '2020-01-01T00:00:03Z', kernels: [
  { factory: 'relu', family: 'activation', summary: 'ReLU', sources: ['activation/relu.cc'], builds: ['relu', 'relu/dtype=int8'],
    passed: 3, failed: [], timed: 1, timing_failed: ['relu/2048/bf16'], untimed: ['relu/64/bf16'] },
]}];
renderKernel('relu', catalogues, db);
const head = $('kernel-head');
assert.equal(head.children[0].text, 'relu activation');
assert.equal(head.children[1].text, 'ReLU');
const links = head.children[2].children.filter(c => c.tag === 'a').map(c => c.href);
assert.deepEqual(links, [
  'https://github.com/Xilinx/mlir-aie/blob/abcdef123456/aie_kernels/activation/relu.cc',
  'https://github.com/Xilinx/mlir-aie/blob/abcdef123456/python/iron/kernels/activation.py',
  'https://github.com/Xilinx/mlir-aie/blob/abcdef123456/test/python/npu/kernel_cases.py',
]);
assert.equal(head.children[3].children[1].text, '2 builds · 3 cases pass · 1 timing failed · charts');
assert.deepEqual($('kernel-cases-head').children.map(c => c.text), ['npu1']);
const rows = $('kernel-cases').children;
assert.deepEqual(rows.map(r => r.children[0].text), ['relu/1024/bf16', 'relu/2048/bf16', 'relu/64/bf16']);
assert.equal(rows[0].children[1].text, '1,030 cycles (+3.0%) · 10.3 cycles/1k-ops (+3.0%) · 150 us (+50.0%) · 4.0 KiB (0.0%)');
assert.equal(rows[1].children[1].text, 'timing failed');
assert.equal(rows[2].children[1].text, 'passed, not timed');
// Its four series, drawn: the stubbed observer intersects at once.
const boxes = $('kernel-charts').children;
assert.deepEqual(boxes.map(b => b.children[0].text), [
  'npu1 · relu/1024/bf16 · cycles +3.0%', 'npu1 · relu/1024/bf16 · cycles_per_kop +3.0%',
  'npu1 · relu/1024/bf16 · npu_us +50.0%', 'npu1 · relu/1024/bf16 · core_elf_bytes 0.0%',
]);
assert.equal(boxes[0].children[0].children[1].href, '#view=kernel&kernel=relu');
assert.equal(boxes[0].children[0].children[3].className, 'delta worse');
assert.equal(boxes[3].children[0].children[3].className, 'delta ');
assert.equal(chart.options.scales.y.title.text, 'bytes');
renderKernel('nope', catalogues, db);
assert.ok($('kernel-head').children.some(c => c.text.includes('No kernel of that name')));
""")


def test_views_come_from_the_hash_and_old_links_keep_working(page):
    page("""
location.hash = '#view=catalogue';
assert.equal(applyView(), 'kernels');
assert.equal($('kernels').hidden, false);
assert.equal($('charts-view').hidden, true);
assert.equal($('tab-kernels').className, 'current');
location.hash = '#view=kernel&kernel=softmax';
assert.equal(applyView(), 'kernel');
assert.equal($('tab-kernel').hidden, false);
assert.equal($('kernel').hidden, false);
location.hash = '';
assert.equal(applyView(), 'dashboard');
assert.equal($('dashboard').hidden, false);
assert.equal($('tab-kernel').hidden, true);
assert.equal($('status').hidden, true);
location.hash = '#view=charts&metric=cycles';
assert.equal(applyView(), 'charts');
assert.equal($('status').hidden, false);
""")
