import argparse
import csv
import os
import re
import subprocess
import time

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("test_dir", type=str, help="Directory containing routing tests")
args = parser.parse_args()


# Regular expression pattern to match the end iteration message
pattern = re.compile(
    r"---End findPaths iteration #(\d+) , illegal edges count = (\d+), total path length = (\d+)---"
)
results = {}
# Iterate over all files in the given directory
files = sorted(os.listdir(args.test_dir))
for file in files:
    filepath = os.path.join(args.test_dir, file)
    if os.path.isfile(filepath) and file.endswith(".mlir"):
        # without the extension
        test = file.split(".")[0]
        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("// RUN: aie-opt"):
                    # Extract the command after // RUN:
                    command = line[len("// RUN:") :].strip()
                    # Replace %s with the file path
                    command = command.replace("%s", filepath)
                    # Split the command by pipe to insert --debug appropriately
                    parts = command.split("|")
                    parts[0] = parts[0].strip() + " --debug"
                    debug_command = parts[0]

                    # Execute the command
                    print(f"Executing command: {debug_command}")
                    start_time = time.time()
                    try:
                        result = subprocess.run(
                            debug_command,
                            shell=True,
                            check=True,
                            capture_output=True,
                            text=True,
                            timeout=1200,
                        )
                        status = "SUCCESS"
                    except subprocess.CalledProcessError as e:
                        result = e
                        status = "FAILED"
                    except subprocess.TimeoutExpired as e:
                        result = e
                        status = "FAILED"
                    end_time = time.time()

                    iteration_count = illegal_edges_count = total_path_length = -1
                    if result.stderr and status != "FAILED":
                        matches = list(pattern.finditer(result.stderr))
                        if matches:
                            # Get the last match
                            last_match = matches[-1]
                            iteration_count = last_match.group(1)
                            illegal_edges_count = last_match.group(2)
                            total_path_length = last_match.group(3)

                    results[test] = {
                        "iteration_count": iteration_count,
                        "illegal_edges_count": illegal_edges_count,
                        "total_path_length": total_path_length,
                        "status": status,
                        "execution_time": end_time - start_time,
                    }
print(results)
# Write the results to a CSV file
csv_file = os.path.join(args.test_dir, "routing_performance_results.csv")
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Test",
            "Iterations Count",
            "Illegal Edges Count",
            "Total Path Length",
            "Status",
            "Execution Time",
        ]
    )
    for test, data in results.items():
        writer.writerow(
            [
                test,
                data["iteration_count"],
                data["illegal_edges_count"],
                data["total_path_length"],
                data["status"],
                data["execution_time"],
            ]
        )

print(f"Results have been written to {csv_file}")
