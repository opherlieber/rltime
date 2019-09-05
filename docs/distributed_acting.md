# Distributed Acting
## Overview
RLtime supports distributed acting, to allow running high throughput decoupled training/acting. This is still a 'work in progress' so please use carefully. In particular there is still work to improve the throughput to match that reported in the [R2D2](https://openreview.net/pdf?id=r1lyTjAqYX) paper, and it has not been fully tested.

Currently only ['Ray'](https://ray.readthedocs.io) is supported for remote worker allocations, however custom worker implementations can be used by sub-classing 'ActorPool' and implementing the relevant methods.

## Ray Usage
For Ray installation and cluster setup instructions please see the [ray docs](https://ray.readthedocs.io/en/latest/installation.html).

Once the cluster is up and running you can configure the 'acting' section of your config file to use a 'Ray Pool' and configure the relevant arguments, for example:

```json
{
    "acting": {
        "actor_envs": 1,
        "pool": {
            "type": "@python('rltime.acting.ray_pool.RayPool')",
            "args": {
                "instances": 128,
                "cpus_per_worker": 1,
                "min_samples_per_request": 50,
            }
        }
    }
}
```
This configures each actor to use a single ENV, create a pool of 128 such actors using ray, and to fetch 50 transitions at a time from each actor. The 'cpus_per_worker' defines how many ray CPU resources to allocate for each such actor. If increasing 'actor_envs' then by default each ENV runs in a sub-process within the actor so you may need to increase cpus_per_worker accordingly.

If your ray redis-address is not the default "localhost:6379" then you need to add that as an argument as well with the correct address.

Additional arguments and their meanings can be found in the initializers for classes 'ActorPool' and 'RayPool'

You will likely need to increase the file descriptor limit on linux if using a replay buffer, due to shared memory optimizations done by default:
```bash
ulimit -n 65536
```
This is in addition to Ray shared memory settings (it's usually enough to initialize the ray stores to 2GB of memory).


You should also usually enable 'async_history' in the training arguments to improve throughput, and set 'train_frequency' to 0/null in the history buffer to allow acting and training to each run at their own rate.

**Important**: Ray is only used for acting, and not for the training and history buffer. You should therefore leave enough (~4-8) CPUs aside for training. For example if your head/GPU node has only 4-8 CPUs it's recommended to set '--num-cpus' for the ray head node to 0, so that actors are only allocated on the remote nodes.

There is currently no support yet for faulty/killed workers. In particular if a remote machine is killed (For example if it's a spot-instance) new actors will not be allocated to replace them, even if ray re-allocates replacement nodes.