import warnings
import math
from torchprofile.handlers import handlers as original_handlers
from torchprofile.utils.trace import trace

__all__ = ['profile_macs', 'CustomMACProfiler']

def pow(node):
    return math.prod(node.outputs[0].shape)

def rsqrt(node):
    return math.prod(node.outputs[0].shape)

def abs(node):
    return math.prod(node.outputs[0].shape)

def log(node):
    return math.prod(node.outputs[0].shape)

def where(node):
    return math.prod(node.outputs[0].shape)

default_custom_handlers = [
    ('aten::pow', pow),
    ('aten::rsqrt', rsqrt),
    ('aten::abs', abs),
    ('aten::log', log),
    ('aten::where', where)
]

class CustomMACProfiler:
    def __init__(self, additional_handlers=None):
        # Merge original handlers with any additional custom handlers provided
        self.handlers = list(original_handlers)
        if additional_handlers is not None:
            self.handlers.extend(additional_handlers)

    def profile_macs(self, model, args=(), kwargs=None, reduction=sum, return_dict_format='all'):
        """
        Profile MACs (Multiply-Accumulate Operations) of a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model to be profiled.
            args (tuple): Arguments for tracing the model.
            kwargs (dict, optional): Keyword arguments for tracing the model.
            reduction (callable, optional): Function to reduce all the MAC counts (default: sum).
            return_dict_format (str): Dict output format ('all', 'individual', or 'both').

                - 'all': Return total MACs as per reduction, and a breakdown by function.
                - 'individual': Return only the breakdown by function.
                - 'both': Return both the reduced total and breakdown.

        Returns:
            Depending on `return_dict_format`:
                - 'all': Returns a single value (total MACs).
                - 'individual': Returns a dict of operations by function.
                - 'both': Returns a tuple (total MACs, dict of operations by function).
        """
        results = dict()
        ops_by_func = dict()

        # Trace the model to get computation graph
        graph = trace(model, args, kwargs)
        for node in graph.nodes:
            for operators, func in self.handlers:
                if isinstance(operators, str):
                    operators = [operators]
                if node.operator in operators:
                    if func is not None:
                        mac_count = func(node)
                        results[node] = mac_count
                        ops_by_func[func.__name__] = ops_by_func.get(func.__name__, 0) + mac_count
                    break
            else:
                warnings.warn(f'No handlers found: "{node.operator}". Skipped.')

        # Prepare the output based on return_dict_format
        reduced_result = reduction(results.values()) if reduction else None

        if return_dict_format == 'all':
            return reduced_result
        elif return_dict_format == 'individual':
            return ops_by_func
        elif return_dict_format == 'both':
            return reduced_result, ops_by_func
        else:
            raise ValueError(f'Invalid value for return_dict_format: {return_dict_format}')

def profile_macs(model, args=(), kwargs=None, reduction=sum, additional_handlers=None, return_dict_format='all'):
    """
    Profile MACs with an easy interface for adding custom handlers.

    Args:
        model (torch.nn.Module): The PyTorch model to be profiled.
        args (tuple): Arguments for tracing the model.
        kwargs (dict, optional): Keyword arguments for tracing the model.
        reduction (callable, optional): Function to reduce all the MAC counts (default: sum).
        additional_handlers (list, optional): Custom handlers to add.
        return_dict_format (str): Dict output format ('all', 'individual', or 'both').

    Returns:
        The MAC profiling results as per return_dict_format.
    """
    profiler = CustomMACProfiler(additional_handlers or default_custom_handlers)
    return profiler.profile_macs(model, args, kwargs, reduction, return_dict_format)

if __name__ == '__main__':
    print("Hi!")