import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# torch is PyTorch's core library. It provides tensors — multi-dimensional
# arrays like numpy but with two superpowers:
#   1. They can live on GPU for massively parallel computation
#   2. PyTorch tracks all operations on them automatically so it can compute
#      gradients — the backbone of how neural networks learn (backpropagation)
#
# torch.nn contains all neural network building blocks: layers, activations,
# loss functions. We alias it as nn for brevity.
# ─────────────────────────────────────────────────────────────────────────────


class DQNNetwork(nn.Module):
    # ─────────────────────────────────────────────────────────────────────────
    # We inherit from nn.Module — PyTorch's base class for all neural networks.
    # Inheriting gives us:
    #   - Automatic tracking of all layers and their learnable parameters
    #   - .to(device) to move everything to GPU in one call
    #   - torch.save() / torch.load() for saving and loading weights
    #   - Gradient computation through our network automatically
    #
    # Think of nn.Module as the engine. DQNNetwork is the car body around it.
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, state_size, action_size):
        # ─────────────────────────────────────────────────────────────────────
        # super().__init__() calls nn.Module's own __init__ first.
        # This sets up all of PyTorch's internal bookkeeping.
        # If you forget this line PyTorch crashes — the machinery never starts.
        # ─────────────────────────────────────────────────────────────────────
        super(DQNNetwork, self).__init__()

        self.state_size  = state_size
        self.action_size = action_size

        # ─────────────────────────────────────────────────────────────────────
        # nn.Sequential chains layers in order. Data flows top to bottom.
        # Each layer's output becomes the next layer's input automatically.
        #
        # Architecture: 3 hidden layers shrinking from 256 → 256 → 128.
        # Why these sizes? Large enough to learn complex market patterns from
        # 162 inputs, small enough to train in reasonable time on a GPU.
        # This is an empirical choice — not derived from theory.
        # ─────────────────────────────────────────────────────────────────────
        self.network = nn.Sequential(

            # ── Layer 1 ───────────────────────────────────────────────────────
            # nn.Linear(in, out) is a fully connected layer.
            # Every input connects to every output neuron.
            # It learns a weight matrix of shape (state_size × 256) and a
            # bias vector of shape (256,). These weights are what training updates.
            nn.Linear(state_size, 256),

            # ── ReLU activation ───────────────────────────────────────────────
            # ReLU(x) = max(0, x). If negative → 0. If positive → unchanged.
            # Without non-linear activations, stacking linear layers is
            # mathematically identical to having just ONE linear layer —
            # they all collapse into a single matrix multiplication.
            # ReLU introduces non-linearity, letting the network approximate
            # any function — including "best action given this market state."
            nn.ReLU(),

            # ── Dropout ───────────────────────────────────────────────────────
            # Randomly zeros out 20% of neurons during each training step.
            # Forces the network to not over-rely on any single neuron.
            # Produces more robust, general features — less memorization.
            # Automatically disabled when you call model.eval() during testing.
            nn.Dropout(p=0.2),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # ── Output layer ──────────────────────────────────────────────────
            # Outputs exactly action_size values (3) — one Q-value per action.
            # Q-value[0] = expected future reward if we SELL now
            # Q-value[1] = expected future reward if we HOLD now
            # Q-value[2] = expected future reward if we BUY now
            #
            # NO activation function here. Q-values can be any real number —
            # negative, zero, large positive. Applying ReLU would cut off
            # negative Q-values. Applying softmax would force them to sum to 1
            # like probabilities. Both would destroy the meaning of Q-values.
            nn.Linear(128, action_size),
        )

        # ─────────────────────────────────────────────────────────────────────
        # Weight initialization.
        # PyTorch initializes weights randomly by default. We do better with
        # Xavier (Glorot) initialization — it sets weights in a range that
        # keeps signal strength roughly constant through each layer.
        # Without good initialization, signals can explode or vanish before
        # training even starts.
        # self.apply() walks every layer and calls _init_weights on each one.
        # ─────────────────────────────────────────────────────────────────────
        self.apply(self._init_weights)


    def _init_weights(self, layer):
        # ─────────────────────────────────────────────────────────────────────
        # isinstance(layer, nn.Linear) checks if this layer is a Linear layer.
        # We skip ReLU and Dropout — they have no weights to initialize.
        # xavier_uniform_ fills the weight matrix with values sampled from a
        # uniform distribution calibrated to the layer's input/output sizes.
        # constant_(bias, 0) sets all biases to zero — standard starting point.
        # The trailing underscore in PyTorch means "in-place operation" —
        # it modifies the tensor directly rather than returning a new one.
        # ─────────────────────────────────────────────────────────────────────
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)


    def forward(self, x):
        # ─────────────────────────────────────────────────────────────────────
        # forward() defines how data flows through the network.
        # PyTorch calls this automatically when you do: network(state_tensor)
        # You never call forward() directly.
        #
        # x: input tensor of shape (batch_size, state_size)
        # Returns: tensor of shape (batch_size, action_size) — the Q-values
        #
        # We just pass x straight through the Sequential container.
        # Sequential handles passing x through each layer in order.
        # ─────────────────────────────────────────────────────────────────────
        return self.network(x)