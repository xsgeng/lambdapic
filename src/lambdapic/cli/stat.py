import re
from collections import defaultdict
from typing import Dict, Tuple

import typer


def parse_log_file(filename: str) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Parse log file and return dictionaries of category times and counts"""
    category_times = defaultdict(float)
    category_counts = defaultdict(int)
    pattern = r'Rank \d+ (.*?) took ([\d.]+)ms'

    try:
        with open(filename, 'r') as file:
            for line in file:
                if 'TIMER' not in line:
                    continue

                # Extract TIMER information part
                timer_info = line.split('|')[-1].strip()

                # Use regex to match category and time
                match = re.search(pattern, timer_info)
                if match:
                    category = match.group(1)
                    time_val = float(match.group(2))
                    category_times[category] += time_val
                    category_counts[category] += 1

    except FileNotFoundError:
        typer.echo(f"Error: File '{filename}' not found", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error processing file: {str(e)}", err=True)
        raise typer.Exit(code=1)

    return category_times, category_counts


def calculate_percentages(category_times: Dict[str, float], category_counts: Dict[str, int]) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Calculate percentages and average times for each category and total time"""
    total_time = sum(category_times.values())
    category_percentages = {}
    category_averages = {}

    for category, time in category_times.items():
        if total_time > 0:
            percentage = (time / total_time) * 100
        else:
            percentage = 0.0
        category_percentages[category] = percentage

        # Calculate average time
        count = category_counts[category]
        if count > 0:
            average = time / count
        else:
            average = 0.0
        category_averages[category] = average

    return category_percentages, category_averages, total_time