import tensorflow as tf
import numpy as np

class CoordHandler():

    def __init__(self, config):
        self.config = config

        self.batch_size = self.config["train_params"]["batch_size"]
        self.micro_patch_size = self.config["data_params"]["micro_patch_size"]
        self.macro_patch_size = self.config["data_params"]["macro_patch_size"]
        self.full_image_size = self.config["data_params"]["full_image_size"]
        self.coordinate_system = self.config["data_params"]["coordinate_system"]
        self.c_dim = self.config["data_params"]["c_dim"]

        self.ratio_macro_to_micro = self.config["data_params"]["ratio_macro_to_micro"]
        self.ratio_full_to_micro = self.config["data_params"]["ratio_full_to_micro"]
        self.num_micro_compose_macro = self.config["data_params"]["num_micro_compose_macro"]

        self.cache = {
            "const_centroid": {},
        }

    def sample_coord(self, num_extrap_steps=0):
        if self.coordinate_system == "euclidean":
            return self._euclidean_sample_coord(num_extrap_steps=num_extrap_steps)
        elif self.coordinate_system == "cylindrical":
            assert num_extrap_steps==0
            return self._cylindrical_sample_coord()
        else:
            raise NotImplementedError()


    def euclidean_coord_int_full_to_float_micro(self, i, ratio_full_to_micro, extrap_steps=0):
        if extrap_steps>0: # Extrapolation training
            ratio_original = ratio_full_to_micro - extrap_steps*2
            return -1 + (i-extrap_steps) * 2 / (ratio_original - 1)
        else:
            return -1 + i * 2 / (ratio_full_to_micro - 1)


    def hyperbolic_coord_int_full_to_float_micro(self, i, ratio_full_to_micro):
        return -1 + i * 2 / ratio_full_to_micro


    def hyperbolic_theta_to_euclidean(self, angle_ratio, proj_func):
        p = proj_func(pi * angle_ratio)
        if (type(p) is float) or (len(p.shape)==0):
            p = p if abs(p) > 1e-6 else 0
        else:
            p[np.abs(p)<1e-6] = 0
        return p    


    def _gen_const_centroids(self, target, dim, num_extrap_steps=0, is_hyperbolic=False):
        assert target in {"full", "macro"}

        # Check if in cache
        cache_key = (target, dim, num_extrap_steps, is_hyperbolic)
        if cache_key in self.cache["const_centroid"]:
            return self.cache["const_centroid"][cache_key]

        const_centroid = []
        if target=="full" and is_hyperbolic:
            assert dim==1
            num_patches = self.ratio_full_to_micro[dim] # Warn: -1 is the same location as 1 in 3D, so ignore 1 here
            ratio_over_micro = self.ratio_full_to_micro[dim]
            for i in range(num_patches):
                coord = -1 + i * 2 / ratio_over_micro
                const_centroid.append(coord)
        elif target=="full":
            num_pad_patch = (self.ratio_macro_to_micro[dim] - 1)
            num_patches = self.ratio_full_to_micro[dim] - num_pad_patch + num_extrap_steps*2
            ratio_over_micro = self.ratio_full_to_micro[dim]
            for i in range(num_patches):
                coord = -1 + (i-num_extrap_steps) * 2 / (ratio_over_micro - 1 - num_pad_patch)
                const_centroid.append(coord)
        else:
            num_patches = self.ratio_macro_to_micro[dim]
            ratio_over_micro = self.ratio_macro_to_micro[dim]
            for i in range(num_patches):
                coord = i / (ratio_over_micro - 1)
                const_centroid.append(coord)
        const_centroid = np.array(const_centroid)
        self.cache["const_centroid"][cache_key] = const_centroid
        return const_centroid


    def _euclidean_sample_coord(self, num_extrap_steps=0):

        const_centroid_x = self._gen_const_centroids(target="full", dim=0, num_extrap_steps=num_extrap_steps)
        const_centroid_y = self._gen_const_centroids(target="full", dim=1, num_extrap_steps=num_extrap_steps)
        const_micro_in_macro_x = self._gen_const_centroids(target="macro", dim=0)
        const_micro_in_macro_y = self._gen_const_centroids(target="macro", dim=1)

        # Random cropping position
        ps = self.micro_patch_size
        gs = self.macro_patch_size
        m_ratio = self.ratio_macro_to_micro
        valid_crop_size_x = self.full_image_size[0] - ps[0]
        valid_crop_size_y = self.full_image_size[1] - ps[1]
        macro_patch_center_range_x = self.full_image_size[0]-self.macro_patch_size[0]
        macro_patch_center_range_y = self.full_image_size[1]-self.macro_patch_size[1]

        num_pad_patch_x = (self.ratio_macro_to_micro[0] - 1)
        num_pad_patch_y = (self.ratio_macro_to_micro[1] - 1)
        d_macro_center_idx_x = np.random.randint(0, self.ratio_full_to_micro[0]-num_pad_patch_x+num_extrap_steps*2, self.batch_size)
        d_macro_center_idx_y = np.random.randint(0, self.ratio_full_to_micro[1]-num_pad_patch_y+num_extrap_steps*2, self.batch_size)

        d_macro_pos_x = np.array([const_centroid_x[i] for i in d_macro_center_idx_x]).reshape(-1, 1)
        d_macro_pos_y = np.array([const_centroid_y[i] for i in d_macro_center_idx_y]).reshape(-1, 1)

        # Wrap value to avoid numerical issue (e.g., 1.000001)
        if num_extrap_steps==0:
            d_macro_pos_x = np.clip(d_macro_pos_x, -1, 1)
            d_macro_pos_y = np.clip(d_macro_pos_y, -1, 1)

        d_macro_coord = np.concatenate([d_macro_pos_x, d_macro_pos_y], axis=1)

        # Transform d global position ([-1, 1]) to patch position ([-1, 1])
        g_micro_pos_x_proto = np.tile(np.expand_dims(d_macro_pos_x, 1), [1, self.num_micro_compose_macro, 1])
        g_micro_pos_y_proto = np.tile(np.expand_dims(d_macro_pos_y, 1), [1, self.num_micro_compose_macro, 1])
        g_micro_pos_x_l, g_micro_pos_y_l = [], []
        gpc_x = macro_patch_center_range_x
        gpc_y = macro_patch_center_range_y
        for yy in range(self.ratio_macro_to_micro[1]):
            for xx in range(self.ratio_macro_to_micro[0]):
                idx = yy*m_ratio[0] + xx
                T_x = const_micro_in_macro_x[xx]
                T_y = const_micro_in_macro_y[yy]
                g_micro_pos_x = ((g_micro_pos_x_proto[:,idx] + 1)/2 * gpc_x + (gs[0]/2) + (T_x*(gs[0]-ps[0])) - (m_ratio[0]/2)*ps[0]) / valid_crop_size_x * 2 - 1
                g_micro_pos_y = ((g_micro_pos_y_proto[:,idx] + 1)/2 * gpc_y + (gs[1]/2) + (T_y*(gs[1]-ps[1])) - (m_ratio[1]/2)*ps[1]) / valid_crop_size_y * 2 - 1
                g_micro_pos_x_l.append(g_micro_pos_x)
                g_micro_pos_y_l.append(g_micro_pos_y)
        g_micro_pos_x = np.concatenate(g_micro_pos_x_l, axis=1).reshape(-1, 1)
        g_micro_pos_y = np.concatenate(g_micro_pos_y_l, axis=1).reshape(-1, 1)
        g_micro_coord = np.concatenate([g_micro_pos_x, g_micro_pos_y], axis=1)

        # Unused, put some trash values
        c_angle_ratio = np.zeros_like(g_micro_pos_x)

        return d_macro_coord, g_micro_coord, c_angle_ratio


    def _cylindrical_sample_coord(self):

        const_centroid_x = self._gen_const_centroids(target="full", dim=0)
        const_centroid_t = self._gen_const_centroids(target="full", dim=1, is_hyperbolic=True)
        const_centroid_a = np.array([self.hyperbolic_theta_to_euclidean(t, proj_func=cos) for t in const_centroid_t])
        const_centroid_b = np.array([self.hyperbolic_theta_to_euclidean(t, proj_func=sin) for t in const_centroid_t])
        const_micro_in_macro_x = self._gen_const_centroids(target="macro", dim=0)
        const_micro_in_macro_t = self._gen_const_centroids(target="macro", dim=1)
        
        # Random cropping position
        num_pad_patch_x = (self.ratio_macro_to_micro[0] - 1)
        d_macro_center_idx_x = np.random.randint(0, self.ratio_full_to_micro[0]-num_pad_patch_x, self.batch_size)
        d_macro_pos_x = np.array([const_centroid_x[i] for i in d_macro_center_idx_x]).reshape(-1, 1)
        d_macro_center_idx_theta = np.random.randint(0, self.ratio_full_to_micro[1], self.batch_size)
        d_macro_pos_t = np.array([const_centroid_t[i] for i in d_macro_center_idx_theta]).reshape(-1, 1)
        d_macro_pos_a = np.array([const_centroid_a[i] for i in d_macro_center_idx_theta]).reshape(-1, 1) # This is for discriminator coord condition
        d_macro_pos_b = np.array([const_centroid_b[i] for i in d_macro_center_idx_theta]).reshape(-1, 1) # This is for discriminator coord condition

        # Wrap value to avoid numerical issue (e.g., 1.000001)
        d_macro_pos_x = np.clip(d_macro_pos_x, -1, 1)

        d_macro_coord = np.concatenate([d_macro_pos_x, d_macro_pos_a, d_macro_pos_b], axis=1)

        ps = self.micro_patch_size
        gs = self.macro_patch_size
        m_ratio = self.ratio_macro_to_micro
        valid_crop_size_x = self.full_image_size[0] - self.micro_patch_size[0]
        macro_patch_center_range_x = self.full_image_size[0]-self.macro_patch_size[0]
        g_micro_pos_x_proto = np.tile(np.expand_dims(d_macro_pos_x, 1), [1, self.num_micro_compose_macro, 1])

        g_micro_pos_x_l, g_micro_pos_a_l, g_micro_pos_b_l, y_angle_ratio_l = [], [], [], []
        g_micro_pos_t_l = [] # Used in real data cropping only
        single_patch_ratio = self.micro_patch_size[1] / self.full_image_size[1] * 2 # Rescale values [0, 1] -> [-1, 1]
        gpc_x = macro_patch_center_range_x
        for yy in range(self.ratio_macro_to_micro[1]):
            for xx in range(self.ratio_macro_to_micro[0]):
                idx = yy*self.ratio_macro_to_micro[0] + xx
                T_x = const_micro_in_macro_x[xx]
                T_y = const_micro_in_macro_t[yy]
                g_micro_pos_x_i = ((g_micro_pos_x_proto[:,idx] + 1)/2 * gpc_x + (gs[0]/2) + (T_x*(gs[0]-ps[0])) - (m_ratio[0]/2)*ps[0]) / valid_crop_size_x * 2 - 1
                g_micro_pos_ab_i = d_macro_pos_t + (T_y*single_patch_ratio)
                g_micro_pos_a_i  = self.hyperbolic_theta_to_euclidean(g_micro_pos_ab_i, proj_func=cos)
                g_micro_pos_b_i  = self.hyperbolic_theta_to_euclidean(g_micro_pos_ab_i, proj_func=sin)
                g_micro_pos_x_l.append(g_micro_pos_x_i)
                g_micro_pos_a_l.append(g_micro_pos_a_i)
                g_micro_pos_b_l.append(g_micro_pos_b_i)
                y_angle_ratio_l.append(g_micro_pos_ab_i)
        g_micro_pos_x = np.concatenate(g_micro_pos_x_l, axis=1).reshape(-1, 1)
        g_micro_pos_a = np.concatenate(g_micro_pos_a_l, axis=1).reshape(-1, 1)
        g_micro_pos_b = np.concatenate(g_micro_pos_b_l, axis=1).reshape(-1, 1)
        y_angle_ratio = np.concatenate(y_angle_ratio_l, axis=1).reshape(-1, 1)

        # Generator condition
        g_micro_coord = np.concatenate([g_micro_pos_x, g_micro_pos_a, g_micro_pos_b], axis=1)

        return d_macro_coord, g_micro_coord, y_angle_ratio

