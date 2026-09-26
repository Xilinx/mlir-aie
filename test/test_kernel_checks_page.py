# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Exercise the page's real collection and chart code without network access."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture
def page():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required to test the kernel checks page")
    path = Path(__file__).resolve().parents[1] / "utils/kernel_checks/index.html"
    script = path.read_text().split("<script>", 1)[1].split("</script>", 1)[0]
    setup = """
const assert = require('node:assert/strict');
// Disable the automatic data fetch and capture the Chart.js configuration.
global.fetch = async () => ({ ok: false });
// Just enough DOM for the page to build its tables.
class El {
  constructor(tag) { this.tag = tag; this.children = []; this.on = {}; this.hidden = false; }
  append(...nodes) { this.children.push(...nodes); }
  addEventListener(type, f) { this.on[type] = f; }
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
global.window = { open: (...args) => { opened = args; }, addEventListener: () => {} };
"""
    data = """
const commit = {
  id: 'abcdef123456', message: 'same revision\\nbody',
  timestamp: '2020-01-01T00:00:00Z', url: 'https://github.com/Xilinx/mlir-aie/commit/abcdef123456',
};
const entry = (date, value) => ({
  commit, date,
  benches: value === null ? [] : [
    { name: 'softmax/1024x16/bfloat16/cycles', unit: 'cycles', value },
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
modeColor = new Map([['turbo', 'red'], ['performance', 'blue']]);
const s = db.series.find(s => s.npu === 'npu1');
const el = { querySelector: () => ({}) };
"""

    def run(checks):
        subprocess.run([node, "-e", setup + script + data + checks], check=True)

    return run


def test_repeated_sha_observations_are_not_deduplicated(page):
    page("""
assert.equal(db.order.length, 7);
assert.equal(new Set(db.order.map(p => p.id)).size, 7);
assert.deepEqual(s.byMode.get('turbo').map(p => p.row.value), [100, 110, 121]);
assert.equal(latestChange(s, ['turbo']), 0.1);
assert.equal(latestChange(s, ['turbo', 'performance']), 0.1);
""")


def test_chart_preserves_history_gaps_and_separate_npus(page):
    page("""
draw(el, s, ['turbo', 'performance']);
assert.deepEqual(chart.data.labels, Array(6).fill('abcdef1'));
assert.deepEqual(chart.data.datasets[0].data, [100, null, null, 110, null, 121]);
assert.deepEqual(chart.data.datasets[1].data, [null, 90, null, null, 95, null]);
const tooltip = chart.options.tooltips.callbacks.afterTitle([{index: 5}]);
assert.ok(tooltip.includes(new Date(300000).toString()));
assert.ok(!tooltip.includes(commit.timestamp));
chart.options.onClick(null, [{_index: 5}]);
assert.deepEqual(opened, [commit.url, '_blank', 'noopener']);
draw(el, db.series.find(s => s.npu === 'npu2'), ['turbo']);
assert.deepEqual(chart.data.datasets[0].data, [999]);
""")


def test_missing_npu_and_filtered_mode_preserve_repeated_runs(page):
    page("""
db = collect([['npu1', { entries: {
  'aie_kernels (npu1, turbo)': [entry(100000, 100), entry(200000, 110)],
}}]]);
draw(el, db.series[0], ['performance', 'turbo']);
assert.equal(chart.data.datasets.length, 1);
assert.deepEqual(chart.data.datasets[0].data, [100, 110]);
assert.equal(latestChange(db.series[0], ['performance']), null);
""")


def test_npu_us_deviation_is_a_band_outside_the_legend_and_tooltip(page):
    page("""
const timed = (date, value, range) => ({ commit, date, benches: [
  { name: 'softmax/1024x16/bfloat16/npu_us', unit: 'us', value, range },
]});
db = collect([['npu1', { entries: { 'aie_kernels (npu1, turbo)': [
  timed(100000, 50, 'min 40.0 max 60.0 n=50'),
  timed(200000, 40, '± 2.5; min 38.0 max 45.0 n=50'),
]}}]]);
modeColor = new Map([['turbo', '#0969da']]);
draw(el, db.series[0], ['turbo']);
const [line, low, high] = chart.data.datasets;
assert.equal(chart.data.datasets.length, 3);
assert.deepEqual(line.data, [50, 40]);
assert.deepEqual(low.data, [null, 37.5]);
assert.deepEqual(high.data, [null, 42.5]);
assert.equal(high.fill, '-1');
assert.equal(low.backgroundColor, 'rgba(9, 105, 218, 0.2)');
const data = chart.data;
assert.deepEqual([0, 1, 2].map(i => chart.options.legend.labels.filter({ datasetIndex: i }, data)),
                 [true, false, false]);
assert.deepEqual([0, 1, 2].map(i => chart.options.tooltips.filter({ datasetIndex: i }, data)),
                 [true, false, false]);
""")


def test_latest_cases_follow_the_latest_nightly(page):
    page("""
const row = (name, unit, value) => ({ name, unit, value });
db = collect([['npu1', { entries: {
  'aie_kernels (npu1, turbo)': [
    { commit, date: 1, benches: [
      row('softmax/1024/bfloat16/cycles', 'cycles', 100),
      row('softmax/64/bfloat16/cycles', 'cycles', 50),
    ]},
    { commit, date: 3, benches: [
      row('softmax/1024/bfloat16/cycles', 'cycles', 110),
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
assert.deepEqual(big.metrics.get('cycles'), { value: 110, unit: 'cycles', change: 0.1 });
assert.equal(big.metrics.get('core_elf_bytes').change, null);
assert.ok(!cases.get('softmax/64/bfloat16').current);
assert.equal(latestCases(db, 'npu2').cases.size, 0);
""")


def test_kernels_view_groups_cases_under_their_factory(page):
    page("""
const row = (name, unit, value) => ({ name, unit, value });
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
renderKernels([{ npu: 'npu1', arch: 'aie2', commit: 'abcdef1', date: 0, kernels: [
  kernel('softmax', { failed: ['softmax/2048/bfloat16'] }),
  kernel('relu', { builds: [], passed: 0, timed: 0 }),
]}], db);
assert.ok($('kernels-status').textContent.endsWith(', turbo mode'));
const rows = $('kernels-rows').children;
const name = r => r.children[r.className === 'case' ? 0 : 1];
assert.deepEqual(rows.map(r => name(r).text), [
  'relu', 'softmax3 cases',
  'softmax/1024/bfloat16', 'softmax/2048/bfloat16', 'softmax/64/bfloat16',
]);
const npu = r => r.children[r.children.length - 1];
assert.equal(npu(rows[0]).text, '—');
assert.equal(npu(rows[1]).text, '1 build · 1 failing · charts');
assert.equal(npu(rows[2]).text, '1,100 cycles (+10.0%) · 4.0 KiB');
assert.equal(npu(rows[2]).children[0].children[1].className, 'worse');
assert.equal(npu(rows[3]).text, 'failing');
assert.equal(npu(rows[4]).className, 'stale');
assert.equal(npu(rows[4]).text, '50 cycles');
assert.equal(rows[2].children[0].children[0].href,
  '#view=charts&npu=npu1&metric=all&kernel=softmax%2F1024%2Fbfloat16');

assert.ok(rows.slice(2).every(r => r.hidden));
rows[1].children[1].children[1].on.click();
assert.ok(rows.slice(2).every(r => !r.hidden));
rows[1].children[1].children[1].on.click();
assert.ok(rows.slice(2).every(r => r.hidden));

$('kernels-filter').value = '2048';
$('kernels-filter').on.input();
assert.deepEqual(rows.map(r => r.hidden), [true, false, true, false, true]);
$('kernels-filter').value = '';
$('kernels-filter').on.input();
assert.deepEqual(rows.map(r => r.hidden), [false, false, true, true, true]);
""")


def test_old_catalogue_links_open_the_kernels_view(page):
    page("""
location.hash = '#view=catalogue';
applyView();
assert.equal($('kernels').hidden, false);
assert.equal($('charts').hidden, true);
""")
