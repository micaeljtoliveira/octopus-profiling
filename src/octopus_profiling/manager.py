# Copyright 2026 Micael Oliveira
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Profiling of Octopus calculations run locally with mpirun.

Each experiment is a directory inside the working directory holding a copy of the control calculation, with the input
file changed to set the layout, and a run.sh script that runs Octopus with the right number of MPI processes and
OpenMP threads. Experiments are run one after the other, so that they do not compete for the same cores.
"""

import logging
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from access.profiling.control import ControlSource, ExistingDirectoryControlSource, GitControlSource
from access.profiling.experiment import (
    ExperimentPlan,
    ProfilingExperiment,
    ProfilingExperimentStatus,
    ProfilingLog,
)
from access.profiling.manager import ProfilingManager

from octopus_profiling.application import OctopusApplication
from octopus_profiling.inp import set_variables

logger = logging.getLogger(__name__)

# Script running an experiment, written to each experiment directory
RUN_SCRIPT = "run.sh"
# File the run script writes the exit code of mpirun to
EXIT_CODE_FILE = "exit_code"
# File the run script writes the standard error of Octopus to
STDERR_FILE = "err"

# Status files Octopus writes to the exec directory, and what they mean
_OCTOPUS_STATUSES = {
    "finished": ProfilingExperimentStatus.DONE,
    "aborted": ProfilingExperimentStatus.FAILED,
    "walltimer-aborted": ProfilingExperimentStatus.FAILED,
    "running": ProfilingExperimentStatus.RUNNING,
}


class OctopusManager(ProfilingManager[OctopusApplication]):
    """Profiling of Octopus calculations run locally with mpirun.

    Args:
        work_dir (Path): Working directory where profiling experiments will be generated and run.
        archive_dir (Path): Directory where completed experiments will be archived.
        application (OctopusApplication): The calculation being profiled.
        control (ControlSource | None): Where the control calculation comes from: a directory holding the input file
            and anything else the calculation needs, e.g. a restart directory. A git repository is cloned into the
            working directory, without prompting for credentials. None (the default) is enough to read and plot
            experiments that already exist.
        mpirun (str): Command used to launch Octopus with MPI. Defaults to "mpirun".
        mpirun_args (Sequence[str]): Further arguments to pass to mpirun, e.g. ["--bind-to", "core"].
        octopus (str): The Octopus executable. Defaults to "octopus".
        preamble (str): Shell commands to run before Octopus in each experiment, e.g. to load modules.
    """

    def __init__(
        self,
        work_dir: Path,
        archive_dir: Path,
        application: OctopusApplication,
        control: ControlSource | None = None,
        *,
        mpirun: str = "mpirun",
        mpirun_args: Sequence[str] = (),
        octopus: str = "octopus",
        preamble: str = "",
    ):
        super().__init__(work_dir, archive_dir, application, control)
        self.mpirun = mpirun
        self.mpirun_args = tuple(mpirun_args)
        self.octopus = octopus
        self.preamble = preamble

    def profiling_logs(self, path: Path, run_path: Path | None = None) -> dict[str, dict[int, ProfilingLog]]:
        """Returns all profiling logs of an experiment.

        Args:
            path (Path): Path to the experiment directory.
            run_path (Path | None): Unused, the logs being inside the experiment directory.

        Returns:
            dict[str, dict[int, ProfilingLog]]: Dictionary mapping log names to their logs. Experiments are run once,
                so every log is returned as the single run 0.

        Raises:
            FileNotFoundError: If the experiment did not write a flat profile.
        """
        logs = self.application.component_logs(path)
        if "time" not in logs:
            raise FileNotFoundError(
                f"No profiling data found in {path / 'profiling'}. Was the experiment run with ProfilingMode set?"
            )
        return {name: {0: log} for name, log in logs.items()}

    def parse_ncpus(self, path: Path, run_path: Path | None = None) -> int:
        """Parses the number of CPUs an experiment used, from what Octopus reported in its standard output.

        Args:
            path (Path): Path to the experiment directory.
            run_path (Path | None): Unused.

        Returns:
            int: Number of MPI processes times the number of OpenMP threads per process. 1 for serial runs.

        Raises:
            FileNotFoundError: If the standard output of Octopus is not found.
        """
        text = (path / self.application.stdout).read_text(errors="replace")
        processes = re.search(r"Number of processes\s*:\s*(\d+)", text)
        threads = re.search(r"Number of threads per process\s*:\s*(\d+)", text)
        if processes is None:
            # Serial runs do not report the number of processes
            return 1
        return int(processes.group(1)) * (int(threads.group(1)) if threads is not None else 1)

    def parse_status(self, path: Path, run_path: Path | None = None) -> ProfilingExperimentStatus | None:
        """Parses the state an experiment reached.

        Octopus keeps a single oct-status-* file in its exec directory saying whether it is running, finished or
        aborted. A run killed before Octopus could say it aborted, e.g. by the walltime limit, leaves the "running"
        status behind, so the exit code the run script records is checked as well.

        Args:
            path (Path): Path to the experiment directory.
            run_path (Path | None): Unused.

        Returns:
            ProfilingExperimentStatus | None: The state of the experiment, or None if it has not run.
        """
        statuses = [status for status in _OCTOPUS_STATUSES if (path / "exec" / f"oct-status-{status}").is_file()]
        status = _OCTOPUS_STATUSES[statuses[0]] if statuses else None

        exit_code_file = path / EXIT_CODE_FILE
        if status != ProfilingExperimentStatus.DONE and exit_code_file.is_file():
            try:
                exit_code = int(exit_code_file.read_text().strip())
            except ValueError:
                logger.debug(f"Could not read the exit code in {exit_code_file}.")
            else:
                if exit_code != 0:
                    return ProfilingExperimentStatus.FAILED

        if status is None:
            logger.debug(f"No Octopus status file in {path / 'exec'}.")
        return status

    def _control_directory(self) -> Path:
        """Returns the directory of the control calculation, cloning it first if needed.

        Raises:
            ValueError: If the control is of an unsupported kind or the directory does not exist.
            subprocess.CalledProcessError: If cloning the control fails.
        """
        control = self.control
        if isinstance(control, ExistingDirectoryControlSource):
            directory = control.path
        elif isinstance(control, GitControlSource):
            directory = self.work_dir / control.directory
            if not directory.exists():
                logger.info(f"Cloning control {control.origin} into {directory}.")
                env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
                subprocess.run(["git", "clone", control.origin, str(directory)], check=True, env=env)
                subprocess.run(["git", "checkout", control.start_point], cwd=directory, check=True, env=env)
        else:
            raise ValueError(f"Unsupported control source: {control!r}.")

        if not directory.is_dir():
            raise ValueError(f"Control directory '{directory}' does not exist.")
        return directory

    def _run_script(self, plan: ExperimentPlan) -> str:
        """Returns the script running an experiment."""
        n_ranks = plan.layout.n_ranks
        n_threads = plan.layout.threads_per_rank or 1
        walltime_seconds = max(1, round(plan.walltime_hours * 3600))
        command = shlex.join([self.mpirun, "-np", str(n_ranks), *self.mpirun_args, self.octopus])
        lines = ["#!/usr/bin/env bash"]
        if self.preamble:
            lines.append(self.preamble.rstrip("\n"))
        lines += [
            f"export OMP_NUM_THREADS={n_threads}",
            f"timeout {walltime_seconds} {command} > {shlex.quote(self.application.stdout)} 2> {STDERR_FILE}",
            f"echo $? > {EXIT_CODE_FILE}",
        ]
        return "\n".join(lines) + "\n"

    def _create_experiments(self, plans: list[ExperimentPlan], **runner_options) -> dict[str, ProfilingExperiment]:
        """Creates the planned experiments as copies of the control calculation.

        Outputs of a previous run of the control are not copied, so that the experiments start afresh. An experiment
        whose directory already exists is left alone, which is how experiments generated in a previous session are
        picked up again.

        Args:
            plans (list[ExperimentPlan]): What to create.
            **runner_options: Ignored. Accepted so that this stays substitutable for the base method.

        Returns:
            dict[str, ProfilingExperiment]: The new experiments, keyed by name.

        Raises:
            ValueError: If the control is unsupported or missing, or if a plan changes a file other than the input
                file.
        """
        control_dir = self._control_directory()
        ignore = shutil.ignore_patterns(
            ".git", "exec", "profiling", self.application.stdout, STDERR_FILE, RUN_SCRIPT, EXIT_CODE_FILE
        )

        created = {}
        for plan in plans:
            exp_path = self.work_dir / plan.name
            if exp_path.exists() and not plan.regenerating:
                logger.info(f"Experiment directory {exp_path} already exists. Leaving it as it is.")
            else:
                if not exp_path.exists():
                    logger.info(f"Creating experiment {plan.name} in {exp_path}.")
                    shutil.copytree(control_dir, exp_path, symlinks=True, ignore=ignore)

                for file_name, changes in plan.changes.items():
                    if file_name != "inp":
                        raise ValueError(f"Cannot change '{file_name}': only the Octopus input file can be changed.")
                    inp = exp_path / file_name
                    inp.write_text(set_variables(inp.read_text(), changes))

                run_script = exp_path / RUN_SCRIPT
                run_script.write_text(self._run_script(plan))
                run_script.chmod(0o755)

            if not plan.regenerating:
                created[plan.name] = ProfilingExperiment(path=exp_path, layout=plan.layout)

        return created

    @staticmethod
    def _clear_run(path: Path) -> None:
        """Removes the outputs of a previous run of an experiment."""
        for status_file in (path / "exec").glob("oct-status-*"):
            status_file.unlink()
        (path / EXIT_CODE_FILE).unlink(missing_ok=True)
        shutil.rmtree(path / "profiling", ignore_errors=True)

    def run_experiments(self, retry_failed: bool = False) -> None:
        """Runs the experiments not run yet, one after the other.

        This blocks until all the experiments have finished. Only experiments generated by this manager can be run,
        since they need a run script.

        Args:
            retry_failed (bool): Whether to run the experiments that failed as well as the new ones. False (the
                default) runs only the new ones.
        """
        # A status does not outlive the session, so ask what each experiment reached before deciding what to run
        self.update_statuses()

        wanted = {ProfilingExperimentStatus.NEW}
        if retry_failed:
            wanted.add(ProfilingExperimentStatus.FAILED)

        to_run = {name: exp for name, exp in self.experiments.items() if exp.status in wanted}
        if not to_run:
            logger.info("No new experiments to run. Will skip execution.")
            return

        for name, exp in to_run.items():
            if not (exp.path / RUN_SCRIPT).is_file():
                logger.warning(f"Experiment '{name}' has no {RUN_SCRIPT}, so it cannot be run. Skipping it.")
                continue

            if exp.status == ProfilingExperimentStatus.FAILED:
                self._clear_run(exp.path)

            logger.info(f"Running experiment '{name}' in '{exp.path}'.")
            exp.status = ProfilingExperimentStatus.RUNNING
            subprocess.run(["bash", RUN_SCRIPT], cwd=exp.path, check=False)

            status = self.parse_status(exp.path)
            exp.status = status if status is not None else ProfilingExperimentStatus.FAILED
            logger.info(f"Experiment '{name}' is {exp.status.name}.")

    def _delete_experiment(self, name: str, dry_run: bool, **kwargs) -> None:
        """Deletes the directory of a single experiment.

        Args:
            name (str): Name of the experiment to delete.
            dry_run (bool): If True, logs what would be deleted without making any changes.
            **kwargs: Ignored. Accepted so that this stays substitutable for the base method.
        """
        exp_path = self.experiments[name].path
        if not exp_path.is_dir():
            logger.warning(f"Experiment directory '{exp_path}' does not exist. Skipping deletion.")
            return
        if not exp_path.resolve().is_relative_to(self.work_dir.resolve()):
            logger.warning(f"Experiment directory '{exp_path}' is outside the working directory. Skipping deletion.")
            return
        if dry_run:
            logger.info(f"Dry run: would delete experiment directory '{exp_path}'.")
            return
        logger.info(f"Deleting experiment directory '{exp_path}'.")
        shutil.rmtree(exp_path)

    def archive_experiments(
        self,
        exclude_dirs: list[str] | None = None,
        exclude_files: list[str] | None = None,
        follow_symlinks: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Archives completed experiments to the archive directory.

        Args:
            exclude_dirs (list[str] | None): Directory patterns to exclude when archiving. Defaults to ["restart"].
            exclude_files (list[str] | None): File patterns to exclude when archiving.
            follow_symlinks (bool): Whether to follow symlinks when archiving. Defaults to False.
            overwrite (bool): Whether to overwrite existing archives. Defaults to False.
        """
        if exclude_dirs is None:
            exclude_dirs = ["restart"]
        super().archive_experiments(
            exclude_dirs=exclude_dirs,
            exclude_files=exclude_files,
            follow_symlinks=follow_symlinks,
            overwrite=overwrite,
        )
