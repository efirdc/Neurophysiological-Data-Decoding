from inspect import signature


def filter_kwargs(constructor, **kwargs):
    kwargs = {
        k: v for k, v in kwargs.items()
        if k in signature(constructor).parameters
    }
    return kwargs
