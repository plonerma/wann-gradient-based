import argparse
import toml
import re
from typing import Mapping, MutableMapping


def nested_update(d: MutableMapping, u: Mapping) -> MutableMapping:
    """ Update nested parameters.

    Source:
    https://stackoverflow.com/a/3233356
    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class ParamsAction(argparse.Action):
    def params(self, namespace, parser):
        if not hasattr(namespace, parser.params_attr):
            setattr(namespace, parser.params_attr, dict())

        return getattr(namespace, parser.params_attr)


class SetParams(ParamsAction):
    """
    Source:
    https://gist.github.com/fralau/061a4f6c13251367ef1d9a9a99fb3e8d
    """

    key_split_pattern = re.compile(r'(?<=[^\\])/')

    def parse_value(self, value):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def __call__(self, parser, namespace, values, option_string=None):
        params = self.params(namespace, parser)

        for item in values:
            key, value = item.split("=", 1)

            *head, tail = [
                # reverse escape slash and remove whitespaces around key
                k.replace(r'\/', '/').strip()
                for k in self.key_split_pattern.split(key)
            ]

            for k in head:
                params[k] = params = params.get(k, dict())

            params[tail] = self.parse_value(value)


class LoadParams(ParamsAction):
    read_params_file = toml.load

    def __call__(self, parser, namespace, values, option_string=None):
        params = self.params(namespace, parser)

        for fn in values:
            with open(fn, 'r') as f:
                nested_update(params, self.__class__.read_params_file(f))


class ParamParser(argparse.ArgumentParser):
    params_attr = 'params'

    def __init__(self, *args, **kwargs):
        self.params_attr = kwargs.pop('params_attr', self.params_attr)

        super().__init__(*args, **kwargs)

        self.add_argument(
            "--set",
            metavar="KEY=VALUE",
            nargs="+",
            help="Set a number of key-value pairs "
            "(do not put spaces before or after the = sign). "
            "If a value contains spaces, you should define "
            "it with double quotes: "
            'foo="this is a sentence". Note that '
            "values are always treated as strings.",
            action=SetParams,
        )

        self.add_argument(
            "--from",
            metavar="FILE",
            nargs="+",
            help="Load parameters from toml files (last file has precedence).",
            action=LoadParams
        )

    def parse_args(self, args=None, namespace=None):
        namespace = namespace or argparse.Namespace()
        if not hasattr(args, self.params_attr):
            setattr(namespace, self.params_attr, dict())

        return super().parse_args(args, namespace)

    def parse_params(self, *args, **kwargs):
        return getattr(self.parse_args(*args, **kwargs), self.params_attr)


if __name__ == "__main__":
    parser = ParamParser()
    params = parser.parse_params()

    print(params)
