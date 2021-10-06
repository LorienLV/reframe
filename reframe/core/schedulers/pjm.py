# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import itertools
import re
import time

import reframe.core.runtime as rt
import reframe.core.schedulers as sched
import reframe.utility.osext as osext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import (JobError,
                                     JobSchedulerError)
from reframe.utility import seconds_to_hms

JOB_STATES = {
    'ACC': 'ACCEPTED',
    'RJT': 'REJECTED',
    'QUE': 'QUEUED',
    'RNA': 'ACQUIRING_RESOURCES',
    'RNP': 'EXECUTING_PROLOGE',
    'RUN': 'EXECUTING',
    'RNE': 'EXECUTING_EPILOGUE',
    'RNO': 'TERMINATING',
    'EXT': 'EXITED',
    'CCL': 'EXITED_BY_INTERRUPTION',
    'HLD': 'FIXED_STATE_DUE_TO_USER',
    'CCL': 'FIXED_STATE_DUE_TO_ERROR',
}

def pjm_state_completed(state):
    completion_states = {
        'REJECTED',
        'EXITED',
        'EXITED_BY_INTERRUPTION'
    }
    if state:
        return all(s in completion_states for s in state.split(','))

    return False

def pjm_state_pending(state):
    pending_states = {
        'ACCEPTED',
        'QUEUED',
        'ACQUIRING_RESOURCES',
        'FIXED_STATE_DUE_TO_USER',
        'FIXED_STATE_DUE_TO_ERROR',
    }
    if state:
        return any(s in pending_states for s in state.split(','))

_run_strict = functools.partial(osext.run_command, check=True)

class _PJMJob(sched.Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_cancelling = False

    @property
    def is_cancelling(self):
        return self._is_cancelling


@register_scheduler('pjm')
class PjmJobScheduler(sched.JobScheduler):
    def __init__(self):
        self._prefix = '#PJM'

        self._submit_timeout = rt.runtime().get_option(
            f'schedulers/@{self.registered_name}/job_submit_timeout'
        )
        self._use_nodes_opt = rt.runtime().get_option(
            f'schedulers/@{self.registered_name}/use_nodes_option'
        )

    def make_job(self, *args, **kwargs):
        return _PJMJob(*args, **kwargs)

    def _format_option(self, option):
        return self._prefix + ' ' + option

    def emit_preamble(self, job):
        num_tasks_per_node = job.num_tasks_per_node or 1
        num_cpus_per_task = job.num_cpus_per_task or 1

        preamble = [
            self._format_option('-N "%s"' % job.name),
            self._format_option('--mpi "proc=%d,max-proc-per-node=%d"' %
                                (job.num_tasks, num_tasks_per_node))
        ]

        outfile_fmt = '-o "%s"' % job.stdout
        errfile_fmt = '-e "%s"' % job.stderr
        preamble += [self._format_option(outfile_fmt),
                     self._format_option(errfile_fmt)]

        if job.time_limit is not None:
            h, m, s = seconds_to_hms(job.time_limit)
            preamble.append(
                self._format_option('-L elapse=%d:%d:%d' % (h, m, s))
            )

        # setting -L node >= 1 is the same as --exclusive in SLURM. 
        if self._use_nodes_opt:
            num_nodes = job.num_tasks // num_tasks_per_node
            preamble.append(self._format_option('-L node=%d' % num_nodes))

        for opt in job.options + job.cli_options:
            preamble.append(self._format_option(opt))

        # Filter out empty statements before returning
        return list(filter(None, preamble))

    def submit(self, job):
        cmd = f'pjsub {job.script_filename}'

        completed = _run_strict(cmd, timeout=self._submit_timeout)

        jobid_match = re.search(r'\[INFO\] PJM [0-9]+ pjsub Job (?P<jobid>\d+) submitted.',
                                completed.stdout)

        if not jobid_match:
            raise JobSchedulerError(
                'could not retrieve the job id of the submitted job'
            )

        job._jobid = jobid_match.group('jobid')
        job._submit_time = time.time()

    def allnodes(self):
        raise NotImplementedError('PJM backend does not support node listing')

    def filternodes(self, job, nodes):
        raise NotImplementedError('PJM backend does not support '
                                  'node filtering')

    def _cancel_if_pending_too_long(self, job):
        if not job.max_pending_time or not pjm_state_pending(job.state):
            return

        t_pending = time.time() - job.submit_time
        if t_pending >= job.max_pending_time:
            self.log(f'maximum pending time for job exceeded; cancelling it')
            self.cancel(job)
            job._exception = JobError('maximum pending time exceeded',
                                      job.jobid)

    def poll(self, *jobs):
        '''Update the status of the jobs.'''

        if jobs:
            # Filter out non-jobs
            jobs = [job for job in jobs if job is not None]

        if not jobs:
            return

        for job in jobs:
            jobinfo = osext.run_command(
                f'pjstat -H -S {job.jobid}'
            )

            state_match = re.search(
                r'^\s*STATE\s*:\s*(?P<state>[A-Z]+)', jobinfo.stdout, re.MULTILINE
            )

            # if not state_match:
            #     self.log(f'Job state not found (job info follows):\n{jobinfo}')
            #     continue

            state = state_match.group('state') if state_match else 'QUE'
            job._state = JOB_STATES[state]

            self._cancel_if_pending_too_long(job)
            if pjm_state_completed(state):
                exitcode_match = re.search(
                    r'^\s*EXIT CODE\s*:\s*(?P<code>\d+)', jobinfo, re.MULTILINE
                )
                if exitcode_match:
                    job._exitcode = int(exitcode_match.group('code'))

                completion_time_match = re.search(
                    r'^\s*JOB END DATE\s*:\s*(?P<date>'
                        '[0-9]{4}\/[0-9]{2}\/[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})', 
                        jobinfo, re.MULTILINE
                )
                if completion_time_match:
                    job._completion_time = completion_time_match.group('date')

    def wait(self, job):
        # Quickly return in case we have finished already
        if self.finished(job):
            return

        intervals = itertools.cycle([1, 2, 3])
        while not self.finished(job):
            self.poll(job)
            time.sleep(next(intervals))

    def cancel(self, job):
        _run_strict(f'pjdel {job.jobid}', timeout=self._submit_timeout)
        job._is_cancelling = True

    def finished(self, job):
        if job.exception:
            raise job.exception

        return pjm_state_completed(job.state)

