import importlib
import os
import re
import time
from pathlib import Path
import sys
from collections import defaultdict
from typing import Dict, Tuple

import typer

from .core.utils.logger import logger
from .simulation import Simulation

class AutoReload:
    def __init__(self, script: Path, exit_on_error=False, exit_on_finish=False):
        self.script_path = script
        self.script = script.stem
        self.last_modified = os.path.getmtime(self.script_path)
        self.modified = False

        sys.path.insert(0, str(script.parent))
        self.module = importlib.import_module(self.script)
        
        self.exit_on_error = exit_on_error
        self.exit_on_finish = exit_on_finish

    @property
    def sim(self):
        for local in dir(self.module):
            if isinstance(getattr(self.module, local), Simulation):
                sim: Simulation = getattr(self.module, local)
                break
        else:
            raise ValueError("No simulation class found in the input file.")
        
        return sim
    
    @property
    def callbacks(self):
        if hasattr(self.module, 'callbacks'):
             return self.module.callbacks
        else:
            logger.warning("No 'callbacks' attribute found in the module.")
            return []
    
    def reload(self):
        try:
            importlib.reload(self.module)
            logger.info(f"Reloaded {self.script}")
        except SyntaxError as e:
            logger.error(f"Syntax error in {self.script}: {e}")
        except Exception as e:
            logger.error(f"Error reloading {self.script}: {e}")

    def check_modification(self):
        current_time = os.path.getmtime(self.script_path)
        if current_time > self.last_modified:
            self.modified = True
            self.last_modified = current_time
            return True
        return False

    def run(self):
        
        finished = False
        while True:
            if self.modified:
                self.reload()
                self.modified = False
                finished = False
            try:
                if not finished:
                    ret = self.sim.initialized = False
                    ret = self.sim.run(callbacks=self.callbacks, stop_callback=self.check_modification)
                    if ret is None:
                        finished = True
                        if self.exit_on_finish:
                            break
                    elif ret == "stop by callback":
                        logger.info("Modification detected. Restarting simulation...")
                        finished = False

            except Exception as e:
                if not self.exit_on_error:
                    logger.info(f"Error: {e}. Restarting...")
                    time.sleep(3)
                    self.modified = True
                else:
                    raise e
            
            # if not self.exit_on_finish:
            self.check_modification()
            time.sleep(3)

app = typer.Typer(
    name="lambdapic",
    help="LambdaPIC Particle-in-Cell Simulation CLI",
    add_completion=False
)

@app.command()
def batch(input_files: list[Path] = typer.Argument(
        ...,
        help="Input simulation scripts",
        exists=True,
        dir_okay=True,
        readable=True
    )
):
    """TODO: batch run simulations"""

@app.command("autoreload")
def auto_reload(
    input_file: Path = typer.Argument(
        ...,
        help="Input simulation script",
        exists=True,
        dir_okay=False,
        readable=True
    ),
    exit_on_error: bool = typer.Option(False, help="Exit on error"),
    exit_on_finish: bool = typer.Option(True, help="Exit on finish")
):
    """
    Auto-reload on modification.
    Useful in clusters with job schedulers like slurm.
    No need to re-queue the job.
    """

    if not input_file.is_file():
        raise typer.Exit(code=1)
    
    ar = AutoReload(input_file, exit_on_error, exit_on_finish)
    ar.run()

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

@app.command("timer-stat")
def analyze_timers(
    filename: str = typer.Argument(..., help="Path to the log file"),
    sort_by: str = typer.Option("time", "--sort", "-s",
                                help="Sort by 'time' or 'name'",
                                case_sensitive=False)
):
    """
    Analyze TIMER information in log files, calculate total time and percentages by category
    
    Examples:
        $ lambdapic timer-stat log.txt
        $ lambdapic timer-stat log.txt --sort name
    """
    # Parse log file
    category_times, category_counts = parse_log_file(filename)
    
    # Calculate percentages and averages
    category_percentages, category_averages, total_time = calculate_percentages(category_times, category_counts)
    
    # Prepare sorting
    if sort_by.lower() == "name":
        sorted_items = sorted(category_times.items(), key=lambda x: x[0])
    else:  # Default sort by time
        sorted_items = sorted(category_times.items(), key=lambda x: x[1], reverse=True)
    
    # Print results
    typer.echo("\\nTIMER CATEGORY ANALYSIS")
    typer.echo(f"Log file: {filename}")
    typer.echo(f"Total TIMER events: {len(category_times)}")
    typer.echo(f"Total time: {total_time:.1f} ms\
")
    
    typer.echo("{:<45} {:>12} {:>15} {:>15}".format("CATEGORY", "TIME (ms)", "AVERAGE (ms)", "PERCENTAGE"))
    typer.echo("-" * 90)
    
    for category, time in sorted_items:
        percentage = category_percentages[category]
        average = category_averages[category]
        typer.echo("{:<45} {:>12.1f} {:>15.1f} {:>14.1f}%".format(category, time, average, percentage))
    
    typer.echo("-" * 90)
    typer.echo(f"{'TOTAL':<45} {total_time:>12.1f} {'':>15} {'100.0%':>15}")

if __name__ == "__main__":
    app()
