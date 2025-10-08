from pathlib import Path

import typer

from lambdapic.cli.auto_reload import AutoReload
from lambdapic.cli.stat import calculate_percentages, parse_log_file

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


@app.command("mcp")
def mcp() -> None:
    """Start the LambdaPIC Model Context Protocol server."""
    try:
        from .mcp import run as run_mcp_server
    except ModuleNotFoundError as e:
        typer.secho(
            "MCP support requires optional dependencies. "
            "Install them with `pip install \"lambdapic[mcp]\"`.",
            fg="red",
            err=True,
        )
        raise e

    run_mcp_server()


if __name__ == "__main__":
    app()
