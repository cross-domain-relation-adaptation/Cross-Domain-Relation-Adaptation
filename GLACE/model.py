import numpy as np


def linear(inp, in_c, out_c, name, vars_list, inp_sparse=False, use_relu=False):
    import tensorflow
    tf = tensorflow.compat.v1
    w_init = tf.initializers.glorot_uniform()
    W = tf.Variable(w_init(shape=[in_c, out_c]), name='W_{}'.format(name), dtype=tf.float32)
    b = tf.Variable(w_init(shape=[out_c]), name='b_{}'.format(name), dtype=tf.float32)
    vars_list.extend([W, b])
    matmul = tf.matmul if not inp_sparse else tf.sparse_tensor_dense_matmul
    x = matmul(inp, W) + b
    return tf.nn.relu(x) if use_relu else x


class GLACE:
    def __init__(self, args, ind=0):
        import tensorflow
        tf = tensorflow.compat.v1
        self.X = args.X_tf
        self.N, self.D = args.X.shape
        self.L = args.embedding_dim
        self.num_cls = len(np.unique(args.labels))
        if args.classes == -1:
            args.classes = [i for i in range(self.num_cls)]
        print(f'self.num_cls={self.num_cls},args.classes={args.classes},unique_labels={np.unique(args.labels)}')
        self.n_hidden = [512]
        self.labels = args.labels
        self.use_multihead = args.use_multihead
        self.labels_tf = tf.convert_to_tensor(self.labels.tolist(), dtype=tf.int32)
        self.u_i = tf.placeholder(name=f'u_i{ind}', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.u_j = tf.placeholder(name=f'u_j{ind}', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.label = tf.placeholder(name=f'label{ind}', dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])
        self.is_hom = tf.placeholder(name=f'is_hom{ind}', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.is_homf = tf.cast(self.is_hom, tf.float32)
        self.is_hetf = 1 - self.is_homf

        self.bb_vars, self.projs_vars, self.da_vars = [], [], []
        self.__create_model(args.proximity, ind)

        def get_split(edges, ground_truth):
            edges_is_legal = np.array([n1 in args.classes and n2 in args.classes for n1, n2 in
                                       zip(self.labels[edges[:, 0]], self.labels[edges[:, 1]])])
            edges_is_hom = self.labels[edges[:, 0]] == self.labels[edges[:, 1]]
            edges_is_het = ~edges_is_hom
            edges_is_hom = edges_is_hom & edges_is_legal
            edges_is_het = edges_is_het & edges_is_legal
            proximity = args.proximity
            return (edges, ground_truth[edges_is_hom], ground_truth[edges_is_het],
                    -self.energy_kl(edges[edges_is_hom, 0], edges[edges_is_hom, 1], proximity),
                    -self.ensemble_of_two_heads_energy_kl(edges[edges_is_het, 0], edges[edges_is_het, 1], proximity),
                    -self.ensemble_of_two_heads_energy_kl(edges[:, 0], edges[:, 1], proximity))

        (self.val_edges, self.val_ground_truth_hom, self.val_ground_truth_het, self.neg_val_energy_hom,
         self.neg_val_energy_het, self.neg_val_energy) = get_split(args.val_edges, args.val_ground_truth)

        (self.test_edges, self.test_ground_truth_hom, self.test_ground_truth_het, self.neg_test_energy_hom,
         self.neg_test_energy_het, self.neg_test_energy) = get_split(args.test_edges, args.test_ground_truth)

        def soft_max_custom_loss(label, energy, weights=None):
            log_sigmoid = tf.log_sigmoid(label * energy)
            if weights is None:
                return -tf.reduce_mean(log_sigmoid)
            else:
                total_weight = (tf.reduce_sum(self.is_homf) + 1e-3)
                return -tf.reduce_sum(log_sigmoid * weights) / total_weight

        # softmax loss
        self.loss_hom = soft_max_custom_loss(
            label=self.label,
            energy=-self.energy_kl(self.u_i, self.u_j, args.proximity),
            weights=None if args.supervised else self.is_homf)

        # da
        self.loss_da = soft_max_custom_loss(
            label=self.is_homf * 2 - 1,
            energy=-self.energy_kl(self.u_i, self.u_j, 'domain-adaptation'))

        # consistency
        total_weight_het = tf.reduce_sum(self.is_hetf) + 1e-3
        co_se = tf.square(self.diff_of_two_heads_energy_kl(self.u_i, self.u_j, args.proximity)) * self.is_hetf
        co_mse = tf.reduce_sum(co_se) / total_weight_het
        # co_rmse = tf.sqrt(co_mse)
        self.loss_co = co_mse

        self.optimizer_bb = tf.train.AdamOptimizer(learning_rate=args.learning_rate_bb)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        self.loss = self.loss_hom + self.loss_da * args.da_coef + self.loss_co * args.co_coef
        self.train_op = tf.group(
            self.optimizer_bb.minimize(self.loss, var_list=self.bb_vars),
            self.optimizer.minimize(self.loss, var_list=self.da_vars + self.projs_vars)
        )
        

    def __create_model(self, proximity, ind):
        import tensorflow
        tf = tensorflow.compat.v1
        sizes = [self.D] + self.n_hidden

        def get_bb_encoding(name, var_list):
            encoded = linear(inp=self.X, in_c=sizes[0], out_c=sizes[1], name=f'{name}_sparse',
                             vars_list=var_list, inp_sparse=True, use_relu=True)
            for i in range(2, len(sizes)):
                encoded = linear(inp=encoded, in_c=sizes[i - 1], out_c=sizes[i], name=f'{name}_{i}',
                                 vars_list=var_list, inp_sparse=False, use_relu=True)
            return encoded

        # bb
        encoding = get_bb_encoding(name=f'bb{ind}', var_list=self.bb_vars)
        self.encoding = encoding

        def get_projection(enc, in_c, out_c, num_proj, name, var_list):
            mu = linear(inp=enc, in_c=in_c, out_c=out_c * num_proj, name=f'mu_{name}',
                        vars_list=var_list, inp_sparse=False, use_relu=False)
            log_sigma = linear(inp=enc, in_c=in_c, out_c=out_c * num_proj, name=f'sigma_{name}',
                               vars_list=var_list, inp_sparse=False, use_relu=False)
            return (tf.reshape(mu, [self.N * num_proj, out_c]),
                    tf.reshape(tf.nn.elu(log_sigma) + 1 + 1e-14, [self.N * num_proj, out_c]))

        self.mu, self.sigma = get_projection(enc=encoding, in_c=sizes[-1], out_c=self.L, num_proj=self.num_cls,
                                             name=f'projections{ind}', var_list=self.projs_vars)

        mu1 = tf.reshape(self.mu, [self.N, self.num_cls, self.L])
        sigma1 = tf.reshape(self.sigma, [self.N, self.num_cls, self.L])
        self.pred1 = mu1[:, 0, :]

        if proximity == 'second-order':
            encoding_ctx = get_bb_encoding(name=f'ctx{ind}', var_list=self.bb_vars)
            self.ctx_mu, self.ctx_sigma = get_projection(enc=encoding_ctx, in_c=sizes[-1], out_c=self.L,
                                                         num_proj=self.num_cls,
                                                         name=f'ctx{ind}', var_list=self.projs_vars)

        @tf.custom_gradient
        def grad_reverse(x):
            return tf.identity(x), lambda dy: -dy

        encoding_rev = grad_reverse(encoding)
        self.da_mu, self.da_sigma = get_projection(enc=encoding_rev, in_c=sizes[-1], out_c=self.L, num_proj=1,
                                                   name=f'da{ind}', var_list=self.da_vars)

    def ensemble_of_two_heads_energy_kl(self, u_i, u_j, proximity):
        return (self.energy_kl(u_i, u_j, proximity) + self.energy_kl(u_j, u_i, proximity)) * 0.5

    def diff_of_two_heads_energy_kl(self, u_i, u_j, proximity):
        import tensorflow
        tf = tensorflow.compat.v1
        return tf.abs(self.energy_kl(u_i, u_j, proximity) - self.energy_kl(u_j, u_i, proximity))

    def energy_kl(self, u_i, u_j, proximity):
        import tensorflow
        tf = tensorflow.compat.v1
        assert proximity in ['first-order', 'second-order', 'domain-adaptation']
        if proximity in ['first-order', 'second-order']:
            if self.use_multihead:
                labels = tf.gather(self.labels_tf, u_i)
            else:
                labels = tf.zeros_like(u_i)
            u_i = u_i * self.num_cls + labels
            u_j = u_j * self.num_cls + labels
            mu_i = tf.gather(self.mu, u_i)
            sigma_i = tf.gather(self.sigma, u_i)
            mu, sigma = self.mu, self.sigma
            if proximity == 'second-order':
                mu, sigma = self.ctx_mu, self.ctx_sigma
            mu_j = tf.gather(mu, u_j)
            sigma_j = tf.gather(sigma, u_j)
        else:
            assert proximity == 'domain-adaptation'
            mu_i = tf.gather(self.da_mu, u_i)
            sigma_i = tf.gather(self.da_sigma, u_i)
            mu_j = tf.gather(self.da_mu, u_j)
            sigma_j = tf.gather(self.da_sigma, u_j)
        return self._sym_energy_kl(gauss_i=(mu_i, sigma_i), gauss_j=(mu_j, sigma_j))

    def _sym_energy_kl(self, gauss_i, gauss_j):
        import tensorflow
        tf = tensorflow.compat.v1

        def _energy_kl(mu_1, sigma_1, mu_2, sigma_2):
            sigma_ratio = sigma_2 / sigma_1
            trace_fac = tf.reduce_sum(sigma_ratio, 1)
            log_det = tf.reduce_sum(tf.log(sigma_ratio + 1e-14), 1)
            mu_diff_sq = tf.reduce_sum(tf.square(mu_1 - mu_2) / sigma_1, 1)
            res = trace_fac + mu_diff_sq - tf.constant(self.L, dtype=tf.float32) - log_det
            return res

        return (_energy_kl(*gauss_i, *gauss_j) + _energy_kl(*gauss_j, *gauss_i)) * .5

