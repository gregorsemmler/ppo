import numpy as np
import torch
from torch import nn

from atari_wrappers import LazyFrames


def get_output_shape(layer, shape):
    layer_training = layer.training
    if layer_training:
        layer.eval()
    out = layer(torch.zeros(1, *shape))
    before_flattening = tuple(out.size())[1:]
    after_flattening = int(np.prod(out.size()))
    if layer_training:
        layer.train()
    return before_flattening, after_flattening


class PreProcessor(object):

    def preprocess(self, x):
        raise NotImplementedError()


class SimpleCNNPreProcessor(PreProcessor):

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def preprocess(self, state_in):
        state = state_in.__array__() if isinstance(state_in, LazyFrames) else state_in
        return torch.from_numpy((state / 255.0).transpose(2, 0, 1)[np.newaxis, :, :]).type(self.dtype)


class NoopPreProcessor(PreProcessor):

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def preprocess(self, x):
        return torch.from_numpy(x[np.newaxis, :]).type(self.dtype)


class ActorCriticModel(nn.Module):

    @property
    def action_dimension(self) -> int:
        raise NotImplementedError()

    @property
    def is_discrete(self) -> bool:
        raise NotImplementedError()

    @property
    def is_shared(self) -> bool:
        raise NotImplementedError()

    def actor_parameters(self):
        raise NotImplementedError()

    def critic_parameters(self):
        raise NotImplementedError()


class MLPModel(ActorCriticModel):

    def __init__(self, input_size, action_dimension, discrete=True, fully_params=(64, 64), activation="relu",
                 fixed_std=False):
        super().__init__()
        self.action_dim = action_dimension
        self.input_size = input_size
        self.activation = activation
        self.discrete = discrete
        self.fixed_std = fixed_std

        policy_layers = []
        value_layers = []

        prev_full_n = self.input_size
        for full_n in fully_params:
            policy_layers.append(nn.Linear(prev_full_n, full_n))
            policy_layers.append(self.get_activation())
            value_layers.append(nn.Linear(prev_full_n, full_n))
            value_layers.append(self.get_activation())
            prev_full_n = full_n

        value_layers.append(nn.Linear(prev_full_n, 1))

        self.policy_shared = nn.Sequential(*policy_layers)
        self.policy_mean = nn.Linear(prev_full_n, action_dimension)
        if not self.discrete:
            if not self.fixed_std:
                self.policy_log_std = nn.Linear(prev_full_n, action_dimension)
            else:
                self.policy_log_std = nn.Parameter(torch.zeros(action_dimension))
        else:
            self.policy_log_std = None
        self.value = nn.Sequential(*value_layers)

    def get_activation(self):
        if self.activation == "relu":
            return nn.ReLU(inplace=True)
        elif self.activation == "elu":
            return nn.ELU(inplace=True)
        raise ValueError(f"Unknown Activation {self.activation}")

    @property
    def action_dimension(self) -> int:
        return self.action_dim

    @property
    def is_discrete(self) -> bool:
        return self.discrete

    @property
    def is_shared(self) -> bool:
        return False

    def actor_parameters(self):
        params = list(self.policy_shared.parameters()) + list(self.policy_mean.parameters())
        if isinstance(self.policy_log_std, nn.Module):
            params += list(self.policy_log_std.parameters())
        elif isinstance(self.policy_log_std, nn.Parameter):
            params += [self.policy_log_std]
        return params

    def critic_parameters(self):
        return self.value.parameters()

    def forward(self, x):
        if self.discrete:
            return self.policy_mean(self.policy_shared(x)), self.value(x)
        p_shared_out = self.policy_shared(x)
        if not self.fixed_std:
            return (self.policy_mean(p_shared_out), self.policy_log_std(p_shared_out)), self.value(x)
        return (self.policy_mean(p_shared_out), self.policy_log_std), self.value(x)


class SharedMLPModel(ActorCriticModel):

    def __init__(self, input_size, action_dimension, discrete=True, shared_params=(64, 64), head_params=(32,),
                 activation="elu", fixed_std=False):
        super().__init__()
        self.action_dim = action_dimension
        self.input_size = input_size
        self.activation = activation
        self.discrete = discrete
        self.fixed_std = fixed_std

        shared_layers = []
        policy_layers = []
        value_layers = []

        prev_full_n = self.input_size
        for full_n in shared_params:
            shared_layers.append(nn.Linear(prev_full_n, full_n))
            shared_layers.append(self.get_activation())
            prev_full_n = full_n

        for full_n in head_params:
            policy_layers.append(nn.Linear(prev_full_n, full_n))
            policy_layers.append(self.get_activation())
            value_layers.append(nn.Linear(prev_full_n, full_n))
            value_layers.append(self.get_activation())
            prev_full_n = full_n

        value_layers.append(nn.Linear(prev_full_n, 1))

        self.shared = nn.Sequential(*shared_layers)
        self.policy_shared = nn.Sequential(*policy_layers)
        self.policy_mean = nn.Linear(prev_full_n, action_dimension)
        if not self.discrete:
            if not self.fixed_std:
                self.policy_log_std = nn.Linear(prev_full_n, action_dimension)
            else:
                self.policy_log_std = nn.Parameter(torch.zeros(action_dimension))
        else:
            self.policy_log_std = None
        self.value = nn.Sequential(*value_layers)

    def get_activation(self):
        if self.activation == "relu":
            return nn.ReLU(inplace=True)
        elif self.activation == "elu":
            return nn.ELU(inplace=True)
        raise ValueError(f"Unknown Activation {self.activation}")

    @property
    def action_dimension(self) -> int:
        return self.action_dim

    @property
    def is_discrete(self) -> bool:
        return self.discrete

    @property
    def is_shared(self) -> bool:
        return True

    def actor_parameters(self):
        return self.parameters()

    def critic_parameters(self):
        return self.parameters()

    def forward(self, x):
        shared_out = self.shared(x)
        p_shared_out = self.policy_shared(shared_out)
        if self.discrete:
            return self.policy_mean(p_shared_out), self.value(shared_out)

        if not self.fixed_std:
            return (self.policy_mean(p_shared_out), self.policy_log_std(p_shared_out)), self.value(x)
        return (self.policy_mean(p_shared_out), self.policy_log_std), self.value(shared_out)


class CNNModel(ActorCriticModel):

    def __init__(self, input_shape, action_dimension, discrete=True, conv_params=((16, 8, 4, 0), (32, 4, 2, 0)),
                 fully_params=(256,), fixed_std=False):
        super().__init__()
        self.input_shapes = input_shape
        self.action_dim = action_dimension
        self.discrete = discrete
        self.fixed_std = fixed_std

        prev_n_filters = self.input_shapes[0]
        conv_layers = []
        for (n_filters, k_size, stride, padding) in conv_params:
            conv_layers.append(nn.Conv2d(prev_n_filters, n_filters, kernel_size=k_size, stride=stride, padding=padding))
            conv_layers.append(nn.ReLU(inplace=True))
            prev_n_filters = n_filters

        self.conv = nn.Sequential(*conv_layers)

        _, prev_full_n = get_output_shape(self.conv, input_shape)
        policy_head_layers = [nn.Flatten()]
        value_layers = [nn.Flatten()]
        for full_n in fully_params:
            policy_head_layers.append(nn.Linear(prev_full_n, full_n))
            policy_head_layers.append(nn.ReLU(inplace=True))
            value_layers.append(nn.Linear(prev_full_n, full_n))
            value_layers.append(nn.ReLU(inplace=True))
            prev_full_n = full_n

        value_layers.append(nn.Linear(prev_full_n, 1))

        self.policy_shared = nn.Sequential(*policy_head_layers)
        self.policy_mean = nn.Linear(prev_full_n, action_dimension)
        if not self.discrete:
            if not self.fixed_std:
                self.policy_log_std = nn.Linear(prev_full_n, action_dimension)
            else:
                self.policy_log_std = nn.Parameter(torch.zeros(action_dimension))
        else:
            self.policy_log_std = None
        self.value_head = nn.Sequential(*value_layers)

    def forward(self, x):
        conv_out = self.conv(x)
        p_shared_out = self.policy_shared(conv_out)
        if self.discrete:
            return self.policy_mean(p_shared_out), self.value_head(conv_out)
        if not self.fixed_std:
            return (self.policy_mean(p_shared_out), self.policy_log_std(p_shared_out)), self.value_head(conv_out)
        return (self.policy_mean(p_shared_out), self.policy_log_std), self.value_head(conv_out)

    @property
    def action_dimension(self) -> int:
        return self.action_dim

    @property
    def is_discrete(self) -> bool:
        return self.discrete

    @property
    def is_shared(self) -> bool:
        return True

    def actor_parameters(self):
        return self.parameters()

    def critic_parameters(self):
        return self.parameters()


class ResidualBlock(nn.Module):

    def __init__(self, in_planes, out_planes, bias=True, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias),
            nn.BatchNorm2d(out_planes)
        ) if stride != 1 or in_planes != out_planes else lambda x: x
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualModel(ActorCriticModel):

    def __init__(self, input_shape, num_filters, num_residual_blocks, val_hidden_size, action_dimension, discrete=True,
                 fixed_std=False):
        super().__init__()
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks
        self.val_hidden_size = val_hidden_size
        self.action_dim = action_dimension
        self.discrete = discrete
        self.fixed_std = fixed_std

        self.residual_tower = nn.Sequential(
            nn.Conv2d(self.input_shape[0], self.num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(inplace=True),
            *[ResidualBlock(self.num_filters, self.num_filters) for _ in range(num_residual_blocks)]
        )

        tower_out_shape = (self.num_filters,) + self.input_shape[1:]

        self.policy_conv = nn.Sequential(
            nn.Conv2d(self.num_filters, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        _, poly_conv_flat = get_output_shape(self.policy_conv, tower_out_shape)
        self.policy_shared = nn.Sequential(
            self.policy_conv,
            nn.Flatten(),
        )
        self.policy_mean = nn.Linear(poly_conv_flat, self.action_dim)
        if not self.discrete:
            if not self.fixed_std:
                self.policy_log_std = nn.Linear(poly_conv_flat, self.action_dim)
            else:
                self.policy_log_std = nn.Parameter(torch.zeros(self.action_dim))
        else:
            self.policy_log_std = None

        self.val_conv = nn.Sequential(
            nn.Conv2d(self.num_filters, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        _, val_conv_flat = get_output_shape(self.val_conv, tower_out_shape)
        self.val_head = nn.Sequential(
            self.val_conv,
            nn.Flatten(),
            nn.Linear(val_conv_flat, self.val_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.val_hidden_size, 1)
        )

    @property
    def action_dimension(self) -> int:
        return self.action_dim

    @property
    def is_discrete(self) -> bool:
        return self.discrete

    @property
    def is_shared(self) -> bool:
        return True

    def actor_parameters(self):
        return self.parameters()

    def critic_parameters(self):
        return self.parameters()

    def forward(self, x):
        tower_out = self.residual_tower(x)
        p_shared_out = self.policy_shared(tower_out)
        if self.discrete:
            return self.policy_mean(p_shared_out), self.val_head(tower_out)
        if not self.fixed_std:
            return (self.policy_mean(p_shared_out), self.policy_log_std(p_shared_out)), self.val_head(tower_out)
        return (self.policy_mean(p_shared_out), self.policy_log_std), self.val_head(tower_out)
