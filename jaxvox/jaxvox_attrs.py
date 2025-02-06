import jax.numpy as jnp

class AttrManager:
    def __init__(self):
        self.attr2key = {}
        self.key2attr = {}
        self.default_value = (0,0,0)

    def get_default_value(self):
        return self.default_value

    def set_default_value(self, value):
        self.default_value = value

    def __len__(self):
        assert len(self.attr2key) == len(self.key2attr)
        return len(self.attr2key)

    def add(self, value):
        offset = 2
        key = len(self)+offset

        if value in self.attr2key:
            return self.attr2key[value]

        self.attr2key[value] = key
        self.key2attr[key] = value
        return key

    def add_attrvals_get_attrkeys(self, attrvals):
        ret = []
        for val in attrvals:
            try:
                len(val)
                val = tuple(val)
            except:
                pass
            ret.append(
                self.add(val)
            )

        return jnp.asarray(ret)

    def get_attrvals_for_attrkeys(self, attrkeys):
        ret = []
        for key in attrkeys:
            ret.append(self.key2attr.get(int(key), self.get_default_value()))
        return jnp.asarray(ret)

    def get_attrkeys_for_attrvas(self, attrkeys):
        ret = []
        for key in attrkeys:
            ret.append(self.attr2key[int(key)])
        return jnp.asarray(ret)
