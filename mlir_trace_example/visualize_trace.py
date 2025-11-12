#!/usr/bin/env python3
##===- visualize_trace.py -----------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##
#
# Script to visualize trace JSON data as a PNG timeline
#
##===----------------------------------------------------------------------===##

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

def parse_trace_json(trace_file):
    """Parse the trace JSON file and extract events."""
    with open(trace_file, 'r') as f:
        data = json.load(f)
    
    # Organize data by process and thread
    processes = {}
    threads = {}
    events = []
    
    for entry in data:
        if entry['ph'] == 'M':  # Metadata
            if entry['name'] == 'process_name':
                processes[entry['pid']] = entry['args']['name']
            elif entry['name'] == 'thread_name':
                threads[(entry['pid'], entry['tid'])] = entry['args']['name']
        elif entry['ph'] in ['B', 'E']:  # Begin or End events
            events.append(entry)
    
    return processes, threads, events

def create_timeline(processes, threads, events, output_file, title="Trace Timeline"):
    """Create a timeline visualization of trace events."""
    
    # Group events into intervals (B -> E pairs)
    active_events = {}
    intervals = []
    
    for event in events:
        key = (event['pid'], event['tid'], event['name'])
        
        if event['ph'] == 'B':  # Begin
            active_events[key] = event['ts']
        elif event['ph'] == 'E':  # End
            if key in active_events:
                start_ts = active_events[key]
                end_ts = event['ts']
                intervals.append({
                    'pid': event['pid'],
                    'tid': event['tid'],
                    'name': event['name'],
                    'start': start_ts,
                    'end': end_ts,
                    'duration': end_ts - start_ts
                })
                del active_events[key]
    
    if not intervals:
        print("No trace intervals found!")
        return
    
    # Create unique lanes for each (pid, tid) combination
    lanes = {}
    lane_idx = 0
    for pid in sorted(set(i['pid'] for i in intervals)):
        for tid in sorted(set(i['tid'] for i in intervals if i['pid'] == pid)):
            lanes[(pid, tid)] = lane_idx
            lane_idx += 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, max(8, min(lane_idx * 0.5, 20))))
    
    # Calculate time range
    if intervals:
        min_time = min(i['start'] for i in intervals)
        max_time = max(i['end'] for i in intervals)
        time_range = max_time - min_time
    else:
        min_time = 0
        max_time = 1
        time_range = 1
    
    ax.set_xlim(min_time - time_range * 0.02, max_time + time_range * 0.02)
    
    # Color map for different event types
    event_colors = {}
    color_palette = plt.cm.tab20.colors
    color_idx = 0
    
    # Plot intervals
    for interval in intervals:
        lane = lanes[(interval['pid'], interval['tid'])]
        event_name = interval['name']
        
        # Assign color to event type
        if event_name not in event_colors:
            event_colors[event_name] = color_palette[color_idx % len(color_palette)]
            color_idx += 1
        
        color = event_colors[event_name]
        
        # Draw rectangle for the interval
        rect = mpatches.Rectangle(
            (interval['start'], lane - 0.4),
            interval['duration'],
            0.8,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add text label if duration is large enough (relative to visible range)
        label_threshold = time_range / 100  # Show label if event is > 1% of total time
        if interval['duration'] > label_threshold:
            ax.text(
                interval['start'] + interval['duration'] / 2,
                lane,
                event_name.replace('INSTR_', '').replace('DMA_', '').replace('_', ' '),
                ha='center',
                va='center',
                fontsize=7,
                weight='bold',
                clip_on=True
            )
    
    # Set up axes
    ax.set_ylim(-0.5, lane_idx - 0.5)
    ax.set_yticks(range(lane_idx))
    
    # Create lane labels
    lane_labels = []
    for (pid, tid), lane in sorted(lanes.items(), key=lambda x: x[1]):
        proc_name = processes.get(pid, f"Process {pid}")
        thread_name = threads.get((pid, tid), f"Thread {tid}")
        lane_labels.append(f"{proc_name}\n{thread_name}")
    
    ax.set_yticklabels(lane_labels, fontsize=8)
    ax.set_xlabel('Time (cycles)', fontsize=10)
    ax.set_title(title, fontsize=12, weight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend for event types (limit to most common)
    event_counts = defaultdict(int)
    for interval in intervals:
        event_counts[interval['name']] += 1
    
    # Show legend for top events
    top_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    legend_patches = [
        mpatches.Patch(color=event_colors[name], label=f"{name} ({count})")
        for name, count in top_events
        if name in event_colors
    ]
    
    if legend_patches:
        ax.legend(
            handles=legend_patches,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
            framealpha=0.9
        )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Trace visualization saved to: {output_file}")
    
    # Print statistics
    print("\nTrace Statistics:")
    print(f"  Total intervals: {len(intervals)}")
    print(f"  Total lanes: {lane_idx}")
    print(f"  Time range: {min(i['start'] for i in intervals)} - {max(i['end'] for i in intervals)} cycles")
    print(f"  Duration: {max(i['end'] for i in intervals) - min(i['start'] for i in intervals)} cycles")
    
    print("\nTop Events:")
    for name, count in top_events[:10]:
        print(f"  {name}: {count}")

def main():
    parser = argparse.ArgumentParser(
        description='Visualize trace JSON data as a PNG timeline'
    )
    parser.add_argument(
        '-i', '--input',
        default='trace_mlir.json',
        help='Input trace JSON file (default: trace_mlir.json)'
    )
    parser.add_argument(
        '-o', '--output',
        default='trace_timeline.png',
        help='Output PNG file (default: trace_timeline.png)'
    )
    parser.add_argument(
        '-t', '--title',
        default='MLIR Trace Timeline',
        help='Chart title (default: MLIR Trace Timeline)'
    )
    
    args = parser.parse_args()
    
    print(f"Reading trace data from: {args.input}")
    processes, threads, events = parse_trace_json(args.input)
    
    print(f"Found {len(processes)} processes, {len(threads)} threads, {len(events)} events")
    
    create_timeline(processes, threads, events, args.output, args.title)

if __name__ == '__main__':
    main()
