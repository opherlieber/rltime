""" SubProc vectorized ENV from OpenAI baselines """
import os
import numpy as np
import multiprocessing as mp
import cloudpickle
import contextlib

from .common import VecEnv, stack_vec_obs


@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default.  If the child process
    has MPI environment variables, MPI will think that the child process is an
    MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables
    temporarily such as when we are starting multiprocessing
    Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)


def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = cloudpickle.loads(env_fn)()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send(
                    (env.observation_space, env.action_space, env.spec))
            else:
                raise NotImplementedError(cmd)
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and
    communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None, context='spawn'):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments
            to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = \
            zip(*[ctx.Pipe() for _ in range(nenvs)])
        self.ps = [
            ctx.Process(
                target=worker,
                args=(work_remote, remote, cloudpickle.dumps(env_fn)))
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
      results = [remote.recv() for remote in self.remotes]
      self.waiting = False
      obs, rews, dones, infos = zip(*results)
      return stack_vec_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return stack_vec_obs([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, \
            "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            try:
                self.close()
            except BrokenPipeError:
                pass


def make_sub_proc_vec_env(creator, num_env, base_seed=0):
    def make_env(rank):  # pylint: disable=C0111
        def _thunk():
            env = creator()
            env.seed(base_seed + rank)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i) for i in range(num_env)])