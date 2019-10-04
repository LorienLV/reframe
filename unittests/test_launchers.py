import abc
import unittest

import reframe.core.launchers as launchers
from reframe.core.launchers.registry import getlauncher
from reframe.core.schedulers import Job


class FakeJob(Job):
    def emit_preamble(self):
        pass

    def submit(self):
        pass

    def wait(self):
        pass

    def cancel(self):
        pass

    def finished(self):
        pass

    def get_all_nodes(self):
        pass

    def filter_nodes(self, nodes):
        pass


class _TestLauncher(abc.ABC):
    '''Base class for launcher tests.'''

    def setUp(self):
        self.job = FakeJob(name='fake_job',
                           launcher=self.launcher,
                           num_tasks=4,
                           num_tasks_per_node=2,
                           num_tasks_per_core=1,
                           num_tasks_per_socket=1,
                           num_cpus_per_task=2,
                           use_smt=True,
                           time_limit=(0, 10, 0),
                           script_filename='fake_script',
                           stdout='fake_stdout',
                           stderr='fake_stderr',
                           sched_account='fake_account',
                           sched_partition='fake_partition',
                           sched_reservation='fake_reservation',
                           sched_nodelist="mynode",
                           sched_exclude_nodelist='fake_exclude_nodelist',
                           sched_exclusive_access='fake_exclude_access',
                           sched_options=['--fake'])
        self.job.options += ['--gres=gpu:4', '#DW jobdw anything']
        self.minimal_job = FakeJob(name='fake_job', launcher=self.launcher)

    @property
    @abc.abstractmethod
    def launcher(self):
        '''The launcher to be tested.'''

    @property
    @abc.abstractmethod
    def expected_command(self):
        '''The command expected to be emitted by the launcher.'''

    @property
    @abc.abstractmethod
    def expected_minimal_command(self):
        '''The command expected to be emitted by the launcher.'''

    def test_run_command(self):
        emitted_command = self.launcher.run_command(self.job)
        self.assertEqual(self.expected_command, emitted_command)

    def test_run_minimal_command(self):
        emitted_command = self.launcher.run_command(self.minimal_job)
        self.assertEqual(self.expected_minimal_command, emitted_command)


class TestSrunLauncher(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return getlauncher('srun')(options=['--foo'])

    @property
    def expected_command(self):
        return 'srun --foo'

    @property
    def expected_minimal_command(self):
        return 'srun --foo'


class TestSrunallocLauncher(_TestLauncher, unittest.TestCase):

    @property
    def launcher(self):
        return getlauncher('srunalloc')(options=['--foo'])

    @property
    def expected_command(self):
        return ('srun '
                '--job-name=fake_job '
                '--time=0:10:0 '
                '--output=fake_stdout '
                '--error=fake_stderr '
                '--ntasks=4 '
                '--ntasks-per-node=2 '
                '--ntasks-per-core=1 '
                '--ntasks-per-socket=1 '
                '--cpus-per-task=2 '
                '--partition=fake_partition '
                '--exclusive '
                '--hint=multithread '
                '--partition=fake_partition '
                '--account=fake_account '
                '--nodelist=mynode '
                '--exclude=fake_exclude_nodelist '
                '--fake '
                '--gres=gpu:4 '
                '--foo')

    @property
    def expected_minimal_command(self):
        return ('srun '
                '--job-name=fake_job '
                '--output=fake_job.out '
                '--error=fake_job.err '
                '--ntasks=1 '
                '--foo')


class TestAlpsLauncher(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return getlauncher('alps')(options=['--foo'])

    @property
    def expected_command(self):
        return 'aprun -B --foo'

    @property
    def expected_minimal_command(self):
        return 'aprun -B --foo'


class TestMpirunLauncher(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return getlauncher('mpirun')(options=['--foo'])

    @property
    def expected_command(self):
        return 'mpirun -np 4 --foo'

    @property
    def expected_minimal_command(self):
        return 'mpirun -np 1 --foo'


class TestMpiexecLauncher(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return getlauncher('mpiexec')(options=['--foo'])

    @property
    def expected_command(self):
        return 'mpiexec -n 4 --foo'

    @property
    def expected_minimal_command(self):
        return 'mpiexec -n 1 --foo'


class TestLauncherWrapperAlps(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return launchers.LauncherWrapper(
            getlauncher('alps')(options=['--foo']),
            'ddt', ['--offline']
        )

    @property
    def expected_command(self):
        return 'ddt --offline aprun -B --foo'

    @property
    def expected_minimal_command(self):
        return 'ddt --offline aprun -B --foo'


class TestLocalLauncher(_TestLauncher, unittest.TestCase):
    @property
    def launcher(self):
        return getlauncher('local')(['--foo'])

    @property
    def expected_command(self):
        return ''

    @property
    def expected_minimal_command(self):
        return ''


class TestSSHLauncher(_TestLauncher, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.job._sched_access = ['-l user', '-p 22222', 'host']
        self.minimal_job._sched_access = ['host']

    @property
    def launcher(self):
        return getlauncher('ssh')(['--foo'])

    @property
    def expected_command(self):
        return 'ssh -o BatchMode=yes -l user -p 22222 --foo host'

    @property
    def expected_minimal_command(self):
        return 'ssh -o BatchMode=yes --foo host'
