import numpy as np


class ValueLog():
    """Implemements a key/value aggregating dictionary log with optional
    grouping/precision and custom aggregation modes"""

    def __init__(self):
        self.log_values = {}

    def log(self, key, val, agg="mean", scope="get", group=None,
            precision=None):
        """Logs a value

        Args:
            key: The key for this value, this will be the key in the resulting
                log dictionary
            val: The value to log
            agg: How to aggregate all the values received, should be the name
                of a valid numpy operation like mean/max/sum etc...
            scope: Scope over which to aggregate/reset the values for this key.
                Valid values are:
                    get: Aggregate and reset each time get() is called
                    None: Never reset (Aggregate all values received from the
                        start)
                    <number>: Aggregate the last <number> values received
            group: Optionally place this key in a sub-key called 'group'. Can
                set a nested group using '->', e.g. "training->general"
            precision: Precision to round the final value to after aggregation

        Note: agg/scope/precision must be the same for each value logged with
            the same key+group
        """
        dest = self.log_values
        if group is not None:
            for subkey in group.split("->"):
                if subkey not in dest:
                    dest[subkey] = {}
                dest = dest[subkey]

        if key not in dest:
            dest[key] = {
                "data": [],
                "scope": scope,
                "agg": agg,
                "precision": precision
            }
        else:
            assert(dest[key]['agg'] == agg)
            assert(dest[key]['precision'] == precision)
            assert(dest[key]['scope'] == scope)
        dest[key]['data'].append(val)

        scope = dest[key]['scope']
        # If scope is a number, leave only that last amount in the history
        if isinstance(scope, int):
            dest[key]['data'] = dest[key]['data'][-int(scope):]

    def log_dict(self, source, agg="auto", group=None):
        """Logs values from a given dictionary in the same group/key structure
        """
        for key, val in source.items():
            if isinstance(val, dict):
                sub_group = key if group is None else group+"->"+key
                self.log_dict(val, agg=agg, group=sub_group)
            else:
                self.log(key, val, group=group, agg=agg)

    def _get_aggregator_for_key(self, key, agg_mode):
        if agg_mode == "auto":
            # 'auto' uses mean unless one of the supported modes is in
            # the key name (Example 'reward_max' will use max)
            supported_modes = ['min', 'mean', 'median', 'max', 'std', 'sum']
            # Example auto-keys might be 'reward_max', or just 'max'
            mode = key.split("_")[-1]
            if mode not in supported_modes:
                agg_mode = "mean"
            else:
                agg_mode = mode

        return getattr(np, agg_mode)

    def _aggregate_log_values(self, source, dest):
        """Aggregates the log values recursively from source->dest"""
        remove = []
        for key, item in source.items():
            if "data" not in item:
                # Assume it's a sub-group
                dest[key] = {}
                self._aggregate_log_values(item, dest[key])
            else:
                aggregator = self._get_aggregator_for_key(key, item['agg'])
                value = aggregator(item['data'])
                if item['precision'] is not None:
                    value = round(value, item['precision'])
                dest[key] = value
                if item['scope'] == 'get':
                    remove.append(key)
        for key in remove:
            del source[key]

    def get(self):
        res = {}
        self._aggregate_log_values(self.log_values, res)
        return res
