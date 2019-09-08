class Normalization(object):

    def __init__(self, dic):
        self.obs_mean = dic['obs_mean'] if 'obs_mean' in dic.keys() else None
        self.obs_std = dic['obs_std'] if 'obs_mean' in dic.keys() else None
        self.acts_mean = dic['acts_mean'] if 'acts_mean' in dic.keys() else None
        self.acts_std = dic['acts_std'] if 'acts_std' in dic.keys() else None
        self.delta_mean = dic['delta_mean'] if 'delta_mean' in dic.keys() else None
        self.delta_std = duc['delta_std'] if 'delta_std' in dic.keys() else None
    
    def update(self, dic):

        self.obs_mean = dic['obs_mean'] if 'obs_mean' in dic.keys() else self.obs_mean
        self.obs_std = dic['obs_std'] if 'obs_mean' in dic.keys() else self.obs_std
        self.acts_mean = dic['acts_mean'] if 'acts_mean' in dic.keys() else self.acts_mean
        self.acts_std = dic['acts_std'] if 'acts_std' in dic.keys() else self.acts_std
        self.delta_mean = dic['delta_mean'] if 'delta_mean' in dic.keys() else self.delta_mean
        self.delta_std = duc['delta_std'] if 'delta_std' in dic.keys() else self.delta_std
