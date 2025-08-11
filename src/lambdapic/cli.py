import importlib
import os
import time
from pathlib import Path
import sys

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

if __name__ == "__main__":
    app()
