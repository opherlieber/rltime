import numpy as np
from collections import deque
import random

from rltime.general.utils import anneal_value
from .replay_history import ReplayHistoryBuffer
from .data_structures.segment_tree import SumSegmentTree, MinSegmentTree


class PrioritizedReplayHistoryBuffer(ReplayHistoryBuffer):
    """A multi-step/multi-env prioritized replay buffer.

    This is tightly based on the openAI baselines prioritized-replay-buffer,
    but ehnanced to support weighted multi-step/multi-env replay with weighted
    overlapped sequence priorities.

    Priorities are tracked for nstep_train sequences (Per unique ENV ID),
    using a combination of mean+max of the td-errors/losses of all the
    transitions in the sequence, as described in the R2D2 paper. Sequences are
    overlapped according to the 'overlap' parameter. For example for
    nstep_train=80 and overlap=40, we will track weighted priorities for
    timesteps (0-79,40-119,80-159,etc...) for each unique ENV.
    There is no data/RAM duplication with the overlaps, but larger overlap
    reduces overall performance due to priority updating overheads.
    The parameters matching the R2D2 paper would be:
        nstep_train=80,overlap=40,max_weight_factor=0.9,alpha=0.9,beta=0.6

    Although there is a lot of code for handling multi-step replay, this class
    should be equivalent to existing prioritized replay implementations
    (e.g. openai baselines) when not doing multi-step replay (i.e. if
    nstep_train=1)

    NOTE: This is still missing the ability to initialize the priorities using
    the qvalues from acting time as done in APEX/R2D2. This is mainly important
    for high-throughput acting (i.e. when training frequency is close to the
    acting frequency)
    Also, there is no limiting of sequneces to not cross episode boundaries,
    except for the 'avoid_episode_crossing' option, though that isn't perfect
    as priorities are still tracked on the boundary-crossing sequence
    """
    def __init__(self, alpha=0.6, beta=0.4, beta_anneal=False, eps=1e-6,
                 overlap=None, max_weight_factor=0.9,
                 global_importance_scaling=False, **kwargs):
        """Initializes a prioritized replay buffer

        Args (In addition to ReplayHistoryBuffer args):
            alpha: Prioritized replay alpha value to use, i.e. the exponent to
                use on the weighted loss/td-error for the actual priority
            beta: Prioritized replay beta value to use for importance weights.
                A value of 0 means uniform/no importance weights while 1 would
                be the full importance weights (relative to the priorities)
            beta_anneal: Whether to anneal the beta value to 1.0 across the
                training period (Can also be a specific value to anneal to)
            eps: Epsilon value to add to all base td-errors to
                avoid 0-priorities causing INF importance weights
            overlap: The amount of overlap to use between subsequent
                nstep_train sequences. The default (None) sets this to
                nstep_train/2 rounded down.
                A value of 0 means no overlap while a (maximal) value of
                nstep_train-1 means a full overlap (i.e. every transition has
                its own weighted priority and can be chosen as the start of an
                nstep_train sequence for training). Larger values, especially
                nearing nstep_train, greatly increase the OH of managing this
                buffer. Some overlap is recommended in order to allow more
                diversity in choosing better sequences, and to allow more
                diversity in the positioning of independent transitions within
                the training sequence (For example to not force a transition
                to always be the last one in an LSTM sequence and not have
                any impact on learning the state representation)
                Can also be negative, for example -1 means a full overlap
            max_weight_factor: Factor to use when calculating the weighted
                priority of an nstep_train sequence. The weighted priority
                will be: <max_weight_factor*max+(1-max_weight_factor)*mean>
                where max and mean are the max and mean of the td-errors/losses
                of all transitions in the sequence.
            global_importance_scaling: Whether importance weights should be
                scaled using the global 'max weight', or the max weight in each
                specific training batch.
                Not clear what is more correct as it is not consistent across
                existing implementations. This changes how the 'beta' value
                causes (or doesn't) an annealing effect on the training rate.
                In particular when this value and beta_anneal are True
                there is a strong annealing of the importance weights across
                the training period which has a similar effect to annealing the
                LR. Setting this to True causes the general scale of the
                weights to be heavily affected by the transition with the
                minimal loss (Can be mitigated with the 'eps' parameter)
        """

        super().__init__(**kwargs)
        self._alpha = alpha
        self._beta = beta
        self._beta_anneal = beta_anneal
        self._eps = eps
        self._max_weight_factor = max_weight_factor
        self._global_importance_scaling = global_importance_scaling
        if overlap is None:
            overlap = int(self.nstep_train / 2)
        elif overlap < 0:
            overlap = self.nstep_train + overlap
            assert(overlap >= 0)
        assert(overlap < self.nstep_train), "Overlap must be < nstep_train"
        self._overlap = overlap
        # The gap (in timesteps) between subsequent overlapped sequences
        self._gap = self.nstep_train - overlap

        # Setup the sorted segment trees
        # The worst-case amount of priotity indices we will have:
        target_capacity = int(self.size / self._gap)
        it_capacity = 1
        while it_capacity < target_capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        # Min tree is only needed if global_importance_scaling is True
        if global_importance_scaling:
            self._it_min = MinSegmentTree(it_capacity)
        else:
            self._it_min = None
        self._max_loss = 1.0

        # segment tree indices for mapping sequences to the segment trees
        self._free_indexes = deque()
        for i in range(target_capacity):
            self._free_indexes.append(i)

        # Maps prioritization indexes to the actual sample data
        # (I.e. to the sample at the start of an overlapped sequence)
        self._index_data = [None] * target_capacity

        # Saves per env-id the running index of the first sample in that ENV.
        # (Allows mapping from a sample to it's offset in its ENV buffer,
        # see sample["env_buffer_offset"])
        self._env_sample_offsets = {}

    def _sample_added(self, sample):
        super()._sample_added(sample)
        assert(len(self._free_indexes) > 0)

        # Initialize initial priority for a new sample
        sample['loss'] = self._max_loss

        # Calculate the offset of this sample among all samples of the specific
        # ENV
        env_id = sample['env_id']
        if env_id not in self._env_sample_offsets:
            self._env_sample_offsets[env_id] = 0
        env_buffer_offset = self._env_sample_offsets[env_id]
        offset = env_buffer_offset+len(self.buffer[env_id])-1
        sample["env_buffer_offset"] = offset

        # Check if we can activate a new nstep_train sequence (Including
        # having enough 'target steps' for the last transition, and enough
        # prefix steps before the sequence)
        base_sequence_offset = \
            (offset - self.nstep_train + 1 - self.nstep_target + 1)
        if (base_sequence_offset % self._gap == 0) and \
                base_sequence_offset >= (env_buffer_offset+self.prefix_steps):
            # Get a free index to use as the prioritization index for this
            # new overlapped sequence
            idx = self._free_indexes.popleft()

            # The first sample in this overlapped nstep_train sequence
            base_sequence_index = base_sequence_offset - env_buffer_offset

            base_sequence_sample = self.buffer[env_id][base_sequence_index]
            base_sequence_sample['prioritization_index'] = idx
            self._index_data[idx] = base_sequence_sample

            # Calculate the initial weighted priority for this overalapped
            # sequence
            self._recalc_weighted_priority(idx)

    def _recalc_weighted_priority(self, idx):
        """Recalculates the weighted prioirty of a sequence"""
        base_sample = self._index_data[idx]
        assert(idx is not None)
        assert(base_sample['prioritization_index'] == idx)
        sequence_offset = base_sample['env_buffer_offset']
        assert(sequence_offset % self._gap == 0)
        env_id = base_sample['env_id']
        start_index = sequence_offset - self._env_sample_offsets[env_id]

        # The range of all nstep_train samples in this overlapped sequence
        sequence_range = \
            self.buffer[env_id][start_index:start_index+self.nstep_train]
        assert(sequence_range[0] is base_sample)
        if self.nstep_train == 1:
            # Optimization for the basic case of nstep_train=1, no need
            # to do all the sum/mean/gathering, just take the loss
            weighted_loss = sequence_range[0]['loss']
        else:
            all_losses = [sample['loss'] for sample in sequence_range]

            # The weighted priority of a sequence is a mixture of the max and
            # mean priorities of all samples in the sequence, controlled by
            # 'max_weight_factor'
            max_loss = np.max(all_losses)
            mean_loss = np.mean(all_losses)
            weighted_loss = self._max_weight_factor * max_loss + \
                (1-self._max_weight_factor)*mean_loss

        weighted_priority = weighted_loss ** self._alpha
        base_sample['weighted_priority'] = weighted_priority
        # Update the relevant sorting trees with the new weighted priority
        self._it_sum[idx] = weighted_priority
        if self._global_importance_scaling:
            self._it_min[idx] = weighted_priority

    def _sample_removed(self, sample):
        """Called when a sample is removed"""
        env_id = sample['env_id']
        # If this sample removal breaks the ability to select an overlapped
        # sequence we disable that sequence from being selected (We use the
        # assumption that it's always the first sample in the env removed)
        sequence_sample = self.buffer[env_id][self.prefix_steps]
        if (sequence_sample['env_buffer_offset'] % self._gap == 0) and \
                ('prioritization_index' in sequence_sample):
            idx = sequence_sample['prioritization_index']
            sequence_sample['prioritization_index'] = None
            # Ensure this idx can't be chosen or affect the min until it's
            # reused
            self._it_sum[idx] = 0
            if self._global_importance_scaling:
                self._it_min[idx] = np.inf
            self._free_indexes.append(idx)
            self._index_data[idx] = None

        assert(sample["env_buffer_offset"] == self._env_sample_offsets[env_id])
        self._env_sample_offsets[env_id] += 1

    def _sample_proportional(self, batch_size):
        """Sample proportionaly from the replay buffer"""
        res = []
        p_total = self._it_sum.sum()
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_losses(self, indices, losses):
        """Called during training to update the training-losses"""
        # Accumulate all prioritization indices affected by these updates
        # (Multiple prioritization indices can be affected by the trained
        # transitions, including indices which were not chosen for training,
        # due to overlapping sequences)
        affected_prio_idxs = {}
        nstep_train = self.nstep_train

        for (env_id, offset), loss in zip(indices, losses):
            env_start_offset = self._env_sample_offsets[env_id]
            if offset < env_start_offset:
                # This sample was removed from the buffer since the time
                # we generated this training data, so just skip
                continue
            env_buffer = self.buffer[env_id]
            # Update the loss for the sample this index refers to
            sample = env_buffer[offset - env_start_offset]
            # Priority is absolute loss/td-error plus fixed epsilon to avoid
            # 0-priority
            sample['loss'] = abs(loss) + self._eps

            # Mark all overlapped sequences affected by this sample,
            # starting from the nearest one
            base_offset = offset - (offset % self._gap)
            while base_offset + nstep_train > offset and \
                    base_offset >= env_start_offset:
                base_sample = env_buffer[base_offset - env_start_offset]
                idx = base_sample.get('prioritization_index', None)
                if idx is not None:
                    affected_prio_idxs[idx] = True
                base_offset -= self._gap

        # Recalc the weighted priority of each overlapped sequence affected
        # by this training update
        for affected_index in affected_prio_idxs:
            self._recalc_weighted_priority(affected_index)

    def _get_train_data(self, mbatch_size, train_progress):

        # Sample indexes according to priority
        idxes = self._sample_proportional(mbatch_size)

        # Anneal beta if requested
        beta = anneal_value(
            self._beta, train_progress, self._beta_anneal, 1.0)

        # The total amount of sequences we currently have in our pool
        total_items = len(self._index_data) - len(self._free_indexes)

        # The total sequence length to retrieve, including burnin/prefix steps
        total_seq_len = self.prefix_steps + self.nstep_train
        if total_items < mbatch_size:
            # We do not have enough nstep_train sequences to get the requested
            # mbatch size. Allow this as long as the buffer isn't full
            assert(len(self.linear_history) < self.size)
            return None

        sample_ranges = []
        extra_data = []
        p_sum = self._it_sum.sum()
        for idx in idxes:
            # The base sample for the overlapped sequence
            base_sample = self._index_data[idx]
            assert(base_sample is not None)

            # Figure out the current index of this sample within it's ENVs
            # buffer
            env_id = base_sample['env_id']
            base_offset = base_sample['env_buffer_offset']
            sample_index = base_offset - self._env_sample_offsets[env_id]
            # Include requested burn-in/prefix timesteps in the train data
            sample_index -= self.prefix_steps
            assert(sample_index >= 0)

            # Optionally shift the the start index to avoid episode crossing,
            # if configured
            sample_index = self._refine_sample_range(
                env_id, sample_index, total_seq_len)
            # Make the basic multistep sample data
            sample_ranges.append(self._make_sample_range(
                env_id, sample_index, total_seq_len))

            # Calculate the importance weight for this sequence
            weight = ((self._it_sum[idx]/p_sum) * total_items)**(-beta)
            # Use the same weight for all items in the sequence
            weights = [weight] * total_seq_len

            # Loss indices for updating the loss
            # Each loss index is a tuple containing the ENV ID and the global
            # offset of the relevant sample within that ENV ID
            # Prefix steps use -1 and should not be sent back to us
            loss_indices = [(-1, -1)]*self.prefix_steps + [
                (env_id, offset)
                for offset in range(base_offset, base_offset+self.nstep_train)
            ]

            extra_data.append({
                'importance_weights': np.array(weights),
                'loss_indices': np.array(loss_indices)
            })

        ret = self._make_train_batch(sample_ranges, extra_data)

        # Scale all importance weights, eighter by the global 'max weight',
        # or the max weight in this batch
        if self._global_importance_scaling:
            p_min = self._it_min.min() / p_sum
            max_weight = (p_min * total_items) ** (-beta)
        else:
            max_weight = np.max(ret['extra_data']['importance_weights'])
        ret['extra_data']['importance_weights'] /= max_weight

        return ret
