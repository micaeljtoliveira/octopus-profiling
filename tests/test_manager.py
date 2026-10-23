# Copyright 2026 Micael Oliveira
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import subprocess
from pathlib import Path
from unittest import mock

import pytest
from access.config.parallel_allocation_strategies import RootAllocation
from access.profiling import RegionGroup
from access.profiling.control import ExistingDirectoryControlSource, GitControlSource
from access.profiling.experiment import ProfilingExperiment, ProfilingExperimentStatus
from access.profiling.metrics import tavg
from conftest import PARALLEL_STDOUT, SERIAL_STDOUT, write_experiment

from octopus_profiling.application import OctopusApplication
from octopus_profiling.inp import OctopusInputParser
from octopus_profiling.manager import OctopusManager
from octopus_profiling.metrics import ttot


@pytest.fixture
def app():
    return OctopusApplication(name="ch4", n_states=8)


@pytest.fixture
def control(tmp_path):
    """A control calculation that was itself run, so that its outputs should not be copied."""
    path = write_experiment(tmp_path / "control", stdout=PARALLEL_STDOUT, status="finished", exit_code=0)
    (path / "restart" / "gs").mkdir(parents=True)
    (path / "err").touch()
    (path / "run.sh").touch()
    return ExistingDirectoryControlSource(path)


@pytest.fixture
def manager(tmp_path, app, control):
    return OctopusManager(
        tmp_path / "work",
        tmp_path / "archive",
        app,
        control,
        mpirun="srun",
        mpirun_args=["--cpu-bind", "cores"],
        octopus="/opt/octopus/bin/octopus",
        preamble="module load octopus\n",
    )


def generate(manager, regenerate_failed=False):
    manager.generate_scaling_experiments(
        [1],
        cores_per_node=2,
        walltime=0.5,
        allocations=RootAllocation(thread_range=(1, 2)),
        regenerate_failed=regenerate_failed,
    )


class TestParseStatus:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("finished", ProfilingExperimentStatus.DONE),
            ("aborted", ProfilingExperimentStatus.FAILED),
            ("walltimer-aborted", ProfilingExperimentStatus.FAILED),
            ("running", ProfilingExperimentStatus.RUNNING),
            (None, None),
        ],
    )
    def test_status_files(self, manager, tmp_path, status, expected):
        path = write_experiment(tmp_path / "exp", status=status)
        assert manager.parse_status(path) == expected

    @pytest.mark.parametrize(
        ("status", "exit_code", "expected"),
        [
            ("running", 124, ProfilingExperimentStatus.FAILED),
            (None, 1, ProfilingExperimentStatus.FAILED),
            ("finished", 1, ProfilingExperimentStatus.DONE),
            ("running", 0, ProfilingExperimentStatus.RUNNING),
            (None, "garbage", None),
        ],
    )
    def test_exit_code(self, manager, tmp_path, status, exit_code, expected):
        path = write_experiment(tmp_path / "exp", status=status, exit_code=exit_code)
        assert manager.parse_status(path) == expected


class TestParseNcpus:
    def test_parallel(self, manager, tmp_path):
        assert manager.parse_ncpus(write_experiment(tmp_path / "exp", stdout=PARALLEL_STDOUT)) == 8

    def test_without_threads(self, manager, tmp_path):
        stdout = PARALLEL_STDOUT.replace("Number of threads per process :       2", "")
        assert manager.parse_ncpus(write_experiment(tmp_path / "exp", stdout=stdout)) == 4

    def test_serial(self, manager, tmp_path):
        assert manager.parse_ncpus(write_experiment(tmp_path / "exp", stdout=SERIAL_STDOUT)) == 1

    def test_missing_stdout(self, manager, tmp_path):
        with pytest.raises(FileNotFoundError):
            manager.parse_ncpus(write_experiment(tmp_path / "exp"))


class TestProfilingLogs:
    def test_logs(self, manager, tmp_path):
        logs = manager.profiling_logs(write_experiment(tmp_path / "exp", ranks=2))

        assert set(logs) == {"time", "time_tree", "time_ranks"}
        assert all(list(runs) == [0] for runs in logs.values())
        assert logs["time"][0].filepath == tmp_path / "exp" / "profiling" / "time.000000"

    def test_no_profile(self, manager, tmp_path):
        with pytest.raises(FileNotFoundError, match="ProfilingMode"):
            manager.profiling_logs(write_experiment(tmp_path / "exp", profile=False))

    def test_parse_profiling_data(self, manager, tmp_path):
        """Tests the whole path from an experiment directory to the parsed data."""
        manager.work_dir.mkdir()
        write_experiment(manager.work_dir / "exp", stdout=PARALLEL_STDOUT, status="finished", ranks=2)
        manager.add_experiment_from_directory("exp", Path("exp"))

        manager.parse_profiling_data()

        assert set(manager.data["exp"]) == {"time", "time_tree", "time_ranks"}
        assert manager.data["exp"]["time_ranks"].sizes == {"region": 4, "pe": 2}

    def test_plots(self, manager):
        """Tests that the parsed data can be plotted with the plotting methods of access-profiling."""
        manager.work_dir.mkdir()
        write_experiment(manager.work_dir / "serial", stdout=SERIAL_STDOUT, status="finished")
        write_experiment(manager.work_dir / "parallel", stdout=PARALLEL_STDOUT, status="finished")
        manager.add_experiment_from_directory("serial", Path("serial"))
        manager.add_experiment_from_directory("parallel", Path("parallel"))
        manager.parse_profiling_data()

        groups = [RegionGroup("time", ["COMPLETE_RUN", "DVLPSI"], relabel={"COMPLETE_RUN": "Total"})]
        manager.plot_scaling_data(groups, tavg, show=False)
        manager.plot_bar_chart(groups, tavg, show=False)
        # The call tree has metrics of its own
        manager.plot_bar_chart([RegionGroup("time_tree", ["COMPLETE_RUN", "IONS_CONSTRUCTOR"])], ttot, show=False)
        # The cores of each experiment are read back from what Octopus reported: 1, and 4 ranks of 2 threads
        figure = manager.plot_component_scaling_data(groups, tavg, show=False)
        assert sorted(figure.axes[0].lines[0].get_xdata()) == [1, 8]
        assert manager.select_best_experiments("time", "COMPLETE_RUN", tavg) == ["serial", "parallel"]


class TestCreateExperiments:
    def test_generate(self, manager):
        generate(manager)

        work_dir = manager.work_dir
        names = {
            "octopus-layout_ch4_dom1_st2_kp1_omp1",
            "octopus-layout_ch4_dom2_st1_kp1_omp1",
            "octopus-layout_ch4_dom1_st1_kp1_omp2",
        }
        assert set(manager.experiments) == names
        assert {path.name for path in work_dir.iterdir()} == names

        exp = manager.experiments["octopus-layout_ch4_dom1_st2_kp1_omp1"]
        assert exp.status == ProfilingExperimentStatus.NEW
        assert exp.layout.decomposition.grid.shape == (1, 2, 1)

        # Outputs of the control are not copied, but everything else is
        assert {path.name for path in exp.path.iterdir()} == {"inp", "restart", "run.sh"}

        variables = OctopusInputParser().parse((exp.path / "inp").read_text())
        assert variables["pardomains"] == 1
        assert variables["parstates"] == 2
        assert variables["parkpoints"] == 1
        assert variables["profilingmode"] == "prof_time"
        assert variables["profilingallnodes"] == "yes"

        assert (exp.path / "run.sh").read_text() == (
            "#!/usr/bin/env bash\n"
            "module load octopus\n"
            "export OMP_NUM_THREADS=1\n"
            "timeout 1800 srun -np 2 --cpu-bind cores /opt/octopus/bin/octopus > out 2> err\n"
            "echo $? > exit_code\n"
        )
        threaded = manager.experiments["octopus-layout_ch4_dom1_st1_kp1_omp2"]
        assert "export OMP_NUM_THREADS=2\n" in (threaded.path / "run.sh").read_text()
        assert manager.parse_status(exp.path) is None

    def test_existing_directories_are_kept(self, manager, tmp_path, app, control):
        generate(manager)
        inp = manager.work_dir / "octopus-layout_ch4_dom2_st1_kp1_omp1" / "inp"
        inp.write_text("edited by hand\n")

        # A new session generating the same experiments picks up the ones already there
        new_manager = OctopusManager(manager.work_dir, tmp_path / "archive", app, control)
        generate(new_manager)

        assert set(new_manager.experiments) == set(manager.experiments)
        assert inp.read_text() == "edited by hand\n"

    def test_regenerate_failed(self, manager):
        generate(manager)
        exp = manager.experiments["octopus-layout_ch4_dom2_st1_kp1_omp1"]
        # The value the control had is of another type than the one written over it
        (exp.path / "inp").write_text("ParDomains = auto\nRadius = 4\n")
        (exp.path / "exit_code").write_text("1\n")
        manager.update_statuses()
        assert exp.status == ProfilingExperimentStatus.FAILED

        generate(manager, regenerate_failed=True)

        assert (exp.path / "inp").read_text().startswith("Radius = 4\nParDomains = 2\n")
        assert manager.experiments["octopus-layout_ch4_dom2_st1_kp1_omp1"] is exp

    def test_only_inp_can_change(self, manager):
        with (
            mock.patch.object(manager.application.__class__, "config_changes", return_value={"other": {}}),
            pytest.raises(ValueError, match="other"),
        ):
            generate(manager)
        assert manager.experiments == {}

    def test_missing_control(self, tmp_path, app):
        manager = OctopusManager(
            tmp_path / "work", tmp_path / "archive", app, ExistingDirectoryControlSource(tmp_path / "nowhere")
        )
        with pytest.raises(ValueError, match="does not exist"):
            generate(manager)

    def test_git_control(self, tmp_path, app, control):
        git_control = GitControlSource(repository="https://example.com/octopus-inputs.git", start_point="v1")
        manager = OctopusManager(tmp_path / "work", tmp_path / "archive", app, git_control)

        def fake_git(args, **kwargs):
            if args[1] == "clone":
                write_experiment(Path(args[3]))
            assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
            return subprocess.CompletedProcess(args, 0)

        with mock.patch("octopus_profiling.manager.subprocess.run", side_effect=fake_git) as run:
            generate(manager)

        assert [call.args[0][:2] for call in run.call_args_list] == [["git", "clone"], ["git", "checkout"]]
        assert run.call_args_list[1].args[0] == ["git", "checkout", "v1"]
        assert len(manager.experiments) == 3


class TestRunExperiments:
    def fake_run(self, outcome):
        """Returns a stand-in for running run.sh that leaves the given Octopus status behind."""

        def run(args, cwd, **kwargs):
            assert args == ["bash", "run.sh"]
            write_experiment(Path(cwd), stdout=PARALLEL_STDOUT, status=outcome, exit_code=0)
            return subprocess.CompletedProcess(args, 0)

        return run

    def test_run(self, manager):
        generate(manager)

        with mock.patch("octopus_profiling.manager.subprocess.run", side_effect=self.fake_run("finished")) as run:
            manager.run_experiments()

        assert run.call_count == 3
        assert {exp.status for exp in manager.experiments.values()} == {ProfilingExperimentStatus.DONE}

    def test_failed_runs_are_retried_on_request(self, manager):
        generate(manager)
        with mock.patch("octopus_profiling.manager.subprocess.run", side_effect=self.fake_run("aborted")):
            manager.run_experiments()
        assert {exp.status for exp in manager.experiments.values()} == {ProfilingExperimentStatus.FAILED}

        with mock.patch("octopus_profiling.manager.subprocess.run") as run:
            manager.run_experiments()
        run.assert_not_called()

        exp = next(iter(manager.experiments.values()))
        with (
            mock.patch("octopus_profiling.manager.subprocess.run", side_effect=self.fake_run("finished")) as run,
            mock.patch.object(manager, "_clear_run", wraps=manager._clear_run) as clear,
        ):
            manager.run_experiments(retry_failed=True)
        assert run.call_count == 3
        assert clear.call_count == 3
        assert exp.status == ProfilingExperimentStatus.DONE
        assert not (exp.path / "exec" / "oct-status-aborted").exists()

    def test_run_without_status(self, manager):
        """Tests that a run leaving no status behind is considered failed."""
        generate(manager)

        def run(args, cwd, **kwargs):
            return subprocess.CompletedProcess(args, 0)

        with mock.patch("octopus_profiling.manager.subprocess.run", side_effect=run):
            manager.run_experiments()

        assert {exp.status for exp in manager.experiments.values()} == {ProfilingExperimentStatus.FAILED}

    def test_experiments_without_run_script_are_skipped(self, manager, caplog):
        manager.work_dir.mkdir()
        manager.experiments["by_hand"] = ProfilingExperiment(write_experiment(manager.work_dir / "by_hand"))

        with mock.patch("octopus_profiling.manager.subprocess.run") as run:
            manager.run_experiments()

        run.assert_not_called()
        assert "no run.sh" in caplog.text

    def test_nothing_to_run(self, manager, caplog):
        caplog.set_level("INFO")
        manager.run_experiments()
        assert "No new experiments to run" in caplog.text


class TestDeleteExperiments:
    def test_delete(self, manager):
        generate(manager)
        name = "octopus-layout_ch4_dom2_st1_kp1_omp1"
        path = manager.experiments[name].path

        manager.delete_experiments([name], dry_run=True)
        assert path.is_dir()
        assert name in manager.experiments

        manager.delete_experiments([name])
        assert not path.exists()
        assert name not in manager.experiments

    def test_outside_work_dir(self, manager, tmp_path, caplog):
        outside = write_experiment(tmp_path / "outside")
        manager.experiments["outside"] = ProfilingExperiment(outside)

        manager.delete_experiments(["outside"])

        assert outside.is_dir()
        assert "outside the working directory" in caplog.text

    def test_missing_directory(self, manager, tmp_path, caplog):
        manager.experiments["gone"] = ProfilingExperiment(tmp_path / "work" / "gone")

        manager.delete_experiments(["gone"])

        assert "does not exist" in caplog.text


def test_archive_excludes_restart(manager):
    with mock.patch("access.profiling.manager.ProfilingManager.archive_experiments") as archive:
        manager.archive_experiments()
    archive.assert_called_once_with(
        exclude_dirs=["restart"], exclude_files=None, follow_symlinks=False, overwrite=False
    )
