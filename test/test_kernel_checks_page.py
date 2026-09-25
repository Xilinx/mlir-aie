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
global.document = {
  getElementById: () => ({}), querySelector: () => ({}), querySelectorAll: () => [],
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
