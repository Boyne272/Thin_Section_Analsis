import threading
import time
import sys
from queue import Queue


class Thread_Handler():

    def __init__(self, task, args):
        '''
        A basic wrapper around pythons threading module

        Parameters
        ----------

        tasks : callable
            The task to be parallelised
        
        arsgs : list of lists
            The arguments to be passed to task each time it is called
            (the length of this is the number of jobs to be done)

        '''

        # store passed params
        self.task = task
        self.args = args
        self.n_jobs = len(args)
        self.start = None

        # setup
        self._print_lock = threading.Lock()
        self._running = []
        self._counter = 0
        self._out_lock = threading.Lock()
        self.outputs = [f'Placeholder {i}' for i in range(self.n_jobs)]
        self.q = Queue()


    def run_serial(self):
        'Runs the given jobs in serial for stable comparison'
        # log the start time
        self.start = time.time()

        # create each job to be done
        for job in range(self.n_jobs):
            sys.stdout.write(f'\rStarting Job {job:3d}'
                             f'\tRuntime {time.time()-self.start:3.2f}s')

            # run job and store result
            output = self.task(*self.args[job])
            with self._out_lock:
                self.outputs[job] = output

        # print the final time
        sys.stdout.write(f'\rFinished'
                         f'\tRuntime {time.time()-self.start:3.2f}s')
        return self.outputs


    def run_parallel(self, n_threads):
        'Creates n_threads and runs all given jobs in parallel'
        # create the threads
        for n in range(n_threads):
            t = threading.Thread(target=self._targ, name=f'Thread-{n}')
            t.daemon = True # tell thread to die on script end
            t.start() # start the thread

        # log the start time
        self.start = time.time()

        # create each job to be done
        for job in range(self.n_jobs):
            self.q.put(job)
        self.q.join() # hold the code until all jobs are done
        
        return self.outputs


    def _targ(self):
        'Thread target function, gets a job from the queue and completes it'
        while True:
            job = self.q.get() # get job number
            thread_name = threading.current_thread().name

            # verbose
            with self._print_lock:
                self._running.append(job)
            if thread_name == 'Thread-0':
                self._sys_update()

            # run job and store result
            output = self.task(*self.args[job])
            with self._out_lock:
                self.outputs[job] = output

            # verbose
            with self._print_lock:
                self._running.remove(job)
                self._counter += 1
            if thread_name == 'Thread-0':
                self._sys_update()

            self.q.task_done() # inform queue that task is done


    def _sys_update(self):
        'Outputs the current state of affairs to user'
        sys.stdout.write(f'\rCompletion {self._counter:3d}/{self.n_jobs:3d}'
                         f'\tRuntime {time.time()-self.start:3.2f}s'
                         f'\tJobs Running: {self._running} \t')
        sys.stdout.flush()
