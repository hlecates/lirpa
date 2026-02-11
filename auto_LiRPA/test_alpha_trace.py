#!/usr/bin/env python3
"""
Test file to trace alpha-CROWN parameter creation on a simple multi-layer ReLU network.
This demonstrates when and how alpha parameters are created in auto_LiRPA.
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class SimpleReLUNetwork(nn.Module):
    """A simple 3-layer ReLU network: input(4) -> 8 -> 6 -> output(2)

    Weights are initialized to create many unstable neurons when input is near zero.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 6)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(6, 2)

        # Initialize weights to create unstable neurons
        # Small weights + zero bias means output will be near zero for inputs near zero
        # This creates the unstable region where lower < 0 < upper
        with torch.no_grad():
            # Layer 1: weights that produce outputs near zero
            self.fc1.weight.data = torch.randn(8, 4) * 0.5
            self.fc1.bias.data = torch.zeros(8)  # Zero bias keeps outputs centered

            # Layer 2: similar setup
            self.fc2.weight.data = torch.randn(6, 8) * 0.5
            self.fc2.bias.data = torch.zeros(6)

            # Layer 3: output layer
            self.fc3.weight.data = torch.randn(2, 6) * 0.5
            self.fc3.bias.data = torch.zeros(2)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def trace_alpha_creation():
    """Main function to trace alpha creation in alpha-CROWN."""

    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    print("=" * 80)
    print("ALPHA-CROWN PARAMETER CREATION TRACE")
    print("=" * 80)

    # Create model
    model = SimpleReLUNetwork()
    model.eval()
    print("\n[1] MODEL ARCHITECTURE:")
    print(f"    Input: 4 neurons")
    print(f"    fc1 -> relu1: 8 neurons (ReLU layer 1)")
    print(f"    fc2 -> relu2: 6 neurons (ReLU layer 2)")
    print(f"    fc3: 2 neurons (output)")

    # Create input with perturbation
    # Use input centered at zero with LARGE perturbation to create many unstable neurons
    x0 = torch.zeros(1, 4)  # batch=1, input_dim=4, centered at zero
    eps_val = 1.0  # Large perturbation!
    x_l = x0 - eps_val
    x_u = x0 + eps_val
    center = (x_l + x_u) / 2.0
    eps = (x_u - x_l) / 2.0

    print("\n[2] INPUT PERTURBATION:")
    print(f"    Input center shape: {center.shape}")
    print(f"    Perturbation eps: {eps[0].tolist()}")
    print(f"    Input range: [{-eps_val}, {eps_val}] for all dimensions")

    # Create BoundedModule with verbosity
    print("\n[3] CREATING BOUNDEDMODULE...")
    bounded_model = BoundedModule(model, x0, bound_opts={
        'verbosity': 1,  # Enable verbose output
        'optimize_bound_args': {
            'iteration': 5,
            'lr_alpha': 0.5,
        }
    })

    # Print nodes in the graph
    print("\n[4] BOUNDED GRAPH NODES:")
    for name, node in bounded_model.named_modules():
        if hasattr(node, '__class__'):
            node_type = node.__class__.__name__
            if 'Bound' in node_type:
                print(f"    {name}: {node_type}")

    # Identify optimizable activations
    print("\n[5] IDENTIFYING OPTIMIZABLE ACTIVATIONS:")
    print("    (These are ReLU nodes where alpha will be created)")

    # Create bounded input
    ptb = PerturbationLpNorm(norm=float('inf'), eps=eps)
    x = BoundedTensor(center, ptb)

    # First, run a forward pass to set perturbed flags
    print("\n[6] RUNNING FORWARD PASS TO SET PERTURBED FLAGS...")
    bounded_model(x)

    # Now check which activations are enabled
    optimizable_acts = bounded_model.get_enabled_opt_act()
    print(f"\n[7] ENABLED OPTIMIZABLE ACTIVATIONS: {len(optimizable_acts)}")
    for node in optimizable_acts:
        print(f"    - {node.name} ({node.__class__.__name__})")
        print(f"      used={node.used}, perturbed={node.perturbed}")

    # Now run alpha-CROWN computation
    print("\n" + "=" * 80)
    print("[8] RUNNING ALPHA-CROWN (method='CROWN-Optimized')")
    print("    This will trigger:")
    print("    a) init_alpha() -> calls compute_bounds with CROWN first")
    print("    b) For each ReLU: init_opt_parameters() creates alpha tensors")
    print("    c) Optimization iterations to refine alpha")
    print("=" * 80)

    # Run with CROWN-Optimized
    lb, ub = bounded_model.compute_bounds(x=(x,), method='CROWN-Optimized')

    print("\n" + "=" * 80)
    print("[9] EXAMINING CREATED ALPHA PARAMETERS")
    print("=" * 80)

    # Examine the alpha parameters for each ReLU layer
    for node in bounded_model.get_enabled_opt_act():
        print(f"\n    ReLU Layer: {node.name}")
        print(f"    alpha_size: {node.alpha_size} (2 for ReLU: lower_lb_slope, lower_ub_slope)")

        if hasattr(node, 'alpha') and node.alpha:
            print(f"    Number of start_nodes with alpha: {len(node.alpha)}")
            for start_name, alpha_tensor in node.alpha.items():
                print(f"\n      Start node: {start_name}")
                print(f"      Alpha shape: {list(alpha_tensor.shape)}")
                print(f"        - Dim 0 ({alpha_tensor.shape[0]}): alpha_size (2 for ReLU)")
                print(f"        - Dim 1 ({alpha_tensor.shape[1]}): spec dimension (output neurons of start_node)")
                print(f"        - Dim 2 ({alpha_tensor.shape[2]}): batch size")
                print(f"        - Remaining: this layer's neuron shape {list(alpha_tensor.shape[3:])}")
                print(f"      Total alpha params: {alpha_tensor.numel()}")
                print(f"      requires_grad: {alpha_tensor.requires_grad}")

                # Show some alpha values
                print(f"      Sample alpha[0] (for lb computation):")
                print(f"        min={alpha_tensor[0].min().item():.4f}, max={alpha_tensor[0].max().item():.4f}")
                print(f"      Sample alpha[1] (for ub computation):")
                print(f"        min={alpha_tensor[1].min().item():.4f}, max={alpha_tensor[1].max().item():.4f}")
        else:
            print(f"    No alpha parameters created (node may not be perturbed)")

    print("\n" + "=" * 80)
    print("[10] ALPHA APPLICATION IN RELU RELAXATION")
    print("=" * 80)
    explain_alpha_application()

    # Demonstrate the relaxation with a specific neuron
    print("\n" + "=" * 80)
    print("[11] DETAILED RELAXATION EXAMPLE FOR ONE NEURON")
    print("=" * 80)
    demonstrate_relu_relaxation(bounded_model)

    print("\n" + "=" * 80)
    print("[12] FINAL BOUNDS")
    print("=" * 80)
    print(f"    Lower bounds: {lb[0].tolist()}")
    print(f"    Upper bounds: {ub[0].tolist()}")

    return bounded_model, lb, ub


def explain_alpha_application():
    """Explain when and how alpha is applied in the backward relaxation."""
    explanation = """
    WHEN ALPHA IS APPLIED (vs. fixed CROWN slopes):
    ------------------------------------------------
    File: operators/relu.py, _backward_relaxation() lines 622-706

    The decision is based on `self.opt_stage`:

    1. opt_stage == 'init' (during init_alpha's CROWN pass):
       - Alpha is NOT used yet
       - Uses fixed slopes from _relu_lower_bound_init() (line 678)
       - Saves these as self.init_d for later alpha initialization (line 265)

    2. opt_stage == 'opt' (during optimization iterations):
       - Alpha IS used as the lower bound slope
       - alpha[0] -> lb_lower_d (slope for computing LOWER bounds)
       - alpha[1] -> ub_lower_d (slope for computing UPPER bounds)

    3. opt_stage == 'reuse' (reusing previously optimized alpha):
       - Same as 'opt', alpha is used

    CODE FLOW IN _backward_relaxation():
    ------------------------------------
    ```python
    # Line 643-652: Check opt_stage and select alpha
    if self.opt_stage in ['opt', 'reuse']:
        # Alpha-CROWN: use optimized slopes
        lower_d = None  # Not used, we use lb_lower_d/ub_lower_d instead
        selected_alpha, alpha_lookup_idx = self.select_alpha_by_idx(...)

        # alpha[0] for lower bound propagation, alpha[1] for upper bound
        if last_lA is not None:
            lb_lower_d = selected_alpha[0]  # <-- ALPHA APPLIED HERE
        if last_uA is not None:
            ub_lower_d = selected_alpha[1]  # <-- ALPHA APPLIED HERE
    else:
        # Regular CROWN: use fixed initialization
        lower_d = self._relu_lower_bound_init(upper_d)  # Fixed slope
    ```

    HOW ALPHA AFFECTS THE RELAXATION:
    ---------------------------------
    For an unstable ReLU neuron with pre-activation bounds [l, u] where l < 0 < u:

    UPPER BOUND (triangle relaxation - FIXED, not optimized):
        slope = u / (u - l)
        intercept = -l * u / (u - l)
        y_upper = slope * x + intercept

    LOWER BOUND (alpha determines the slope - OPTIMIZED):
        slope = alpha  (where 0 <= alpha <= 1)
        intercept = 0
        y_lower = alpha * x

    The optimization tries to find alpha values that maximize the final
    lower bounds (or minimize upper bounds) of the network output.

    MASKING FOR STABLE NEURONS:
    ---------------------------
    After selecting alpha, _relu_mask_alpha() is called (line 666):
    - For neurons where upper <= 0: force slope = 0 (always inactive)
    - For neurons where lower >= 0: force slope = 1 (always active)
    - Alpha only matters for unstable neurons where lower < 0 < upper
    """
    print(explanation)


def demonstrate_relu_relaxation(bounded_model):
    """Show the actual relaxation values for specific neurons."""
    import torch

    for node in bounded_model.get_enabled_opt_act():
        if not hasattr(node, 'inputs') or len(node.inputs) == 0:
            continue

        input_node = node.inputs[0]
        if not hasattr(input_node, 'lower') or input_node.lower is None:
            continue

        lower = input_node.lower  # Pre-activation lower bounds
        upper = input_node.upper  # Pre-activation upper bounds

        print(f"\n    ReLU Layer: {node.name}")
        print(f"    Pre-activation bounds shape: {list(lower.shape)}")

        # Compute stability status for each neuron
        always_active = (lower >= 0).flatten()
        always_inactive = (upper <= 0).flatten()
        unstable = (~always_active & ~always_inactive).flatten()

        print(f"\n    Neuron stability analysis:")
        print(f"      Always active (l >= 0):   {always_active.sum().item()} neurons")
        print(f"      Always inactive (u <= 0): {always_inactive.sum().item()} neurons")
        print(f"      Unstable (l < 0 < u):     {unstable.sum().item()} neurons")

        # Show details for first few unstable neurons
        unstable_indices = unstable.nonzero(as_tuple=True)[0]
        if len(unstable_indices) > 0:
            print(f"\n    Relaxation for first 3 unstable neurons:")
            print(f"    " + "-" * 60)

            lower_flat = lower.flatten()
            upper_flat = upper.flatten()

            for i, idx in enumerate(unstable_indices[:3]):
                l = lower_flat[idx].item()
                u = upper_flat[idx].item()

                # Upper bound (triangle relaxation)
                upper_slope = u / (u - l)
                upper_intercept = -l * u / (u - l)

                print(f"\n    Neuron {idx.item()}:")
                print(f"      Pre-activation bounds: [{l:.4f}, {u:.4f}]")
                print(f"      Upper bound (FIXED triangle relaxation):")
                print(f"        slope = u/(u-l) = {u:.4f}/({u:.4f}-({l:.4f})) = {upper_slope:.4f}")
                print(f"        intercept = -l*u/(u-l) = {upper_intercept:.4f}")
                print(f"        y_upper = {upper_slope:.4f} * x + {upper_intercept:.4f}")

                # Get alpha values for this neuron
                print(f"      Lower bound (OPTIMIZED alpha):")
                for start_name, alpha_tensor in node.alpha.items():
                    # Alpha shape: [2, spec_dim, batch, *layer_shape]
                    # We want the alpha for this specific neuron
                    if alpha_tensor.dim() == 4:  # [2, spec, batch, neurons]
                        # Get alphas for first spec, first batch
                        alpha_lb = alpha_tensor[0, 0, 0, idx % alpha_tensor.shape[3]].item()
                        alpha_ub = alpha_tensor[1, 0, 0, idx % alpha_tensor.shape[3]].item()
                        print(f"        For start_node '{start_name}':")
                        print(f"          alpha[0] (for lb computation) = {alpha_lb:.4f}")
                        print(f"          alpha[1] (for ub computation) = {alpha_ub:.4f}")
                        print(f"          y_lower = {alpha_lb:.4f} * x  (intercept always 0)")
                    break  # Just show first start_node for brevity

        # Show the opt_stage
        print(f"\n    Current opt_stage: {node.opt_stage}")
        print(f"    (alpha is used when opt_stage in ['opt', 'reuse'])")


def explain_alpha_flow():
    """Print a detailed explanation of the alpha creation flow."""

    explanation = """
================================================================================
ALPHA-CROWN PARAMETER CREATION FLOW (DEFAULT SETTINGS)
================================================================================

OVERVIEW:
---------
Alpha parameters are learnable slope parameters for the lower bound relaxation
of ReLU (and other piecewise-linear) activations. In the unstable region where
lower < 0 < upper, the lower bound of ReLU can have any slope between 0 and 1.
Alpha-CROWN optimizes this slope to get tighter bounds.

KEY FILES:
----------
- optimized_bounds.py:971-1088  -> init_alpha() main entry point
- backward_bound.py:856-946     -> get_alpha_crown_start_nodes()
- operators/relu.py:52-162      -> init_opt_parameters() creates tensors
- operators/relu.py:622-706     -> _backward_relaxation() uses alpha

STEP-BY-STEP FLOW:
------------------

1. ENTRY POINT: compute_bounds(method='CROWN-Optimized')
   File: bound_general.py

   When you call bounded_model.compute_bounds(method='CROWN-Optimized'),
   it triggers the optimization path in optimize_bounds().

2. OPTIMIZE_BOUNDS SETUP: optimize_bounds()
   File: optimized_bounds.py:430-444

   Default settings (from default_optimize_bound_args):
     - enable_alpha_crown: True   <- enables alpha optimization
     - init_alpha: True           <- triggers alpha initialization
     - iteration: 20              <- optimization iterations
     - lr_alpha: 0.5              <- learning rate for alpha
     - use_shared_alpha: False    <- separate alpha per start node

   If init_alpha=True, calls self.init_alpha() at line 443.

3. INIT_ALPHA: init_alpha()
   File: optimized_bounds.py:971-1088

   a) Forward pass: self(*x) sets perturbed flags on nodes (line 975)

   b) Get optimizable activations: get_enabled_opt_act() (line 980)
      - Must be BoundOptimizableActivation
      - Must have used=True (in computation graph)
      - Must have perturbed=True (affected by input perturbation)

   c) Set init stage: node.opt_init() (line 988)

   d) Run CROWN pass: compute_bounds(method='CROWN') (lines 1004-1007)
      - This computes bounds through the network
      - For each ReLU, _backward_relaxation() is called
      - init_d is saved with the initial lower bound slopes (relu.py:265)

   e) Get start nodes: get_alpha_crown_start_nodes() (lines 1036-1041)
      - Returns list of (node_name, output_shape, unstable_idx, is_final)
      - Determines which nodes need their bounds propagated through this ReLU

   f) Create alpha: node.init_opt_parameters(start_nodes) (line 1048)

4. GET_ALPHA_CROWN_START_NODES:
   File: backward_bound.py:856-946

   For each ReLU node, determines which downstream nodes propagate bounds
   through it. Returns start_nodes list with:
     - Final output node (if bounds propagate to output)
     - Intermediate nodes (pre-activation of later ReLUs)

   Each entry: (node_name, output_shape, unstable_idx, is_final_node)

5. INIT_OPT_PARAMETERS: BoundRelu.init_opt_parameters()
   File: operators/relu.py:52-162

   THIS IS WHERE ALPHA TENSORS ARE ACTUALLY CREATED:

   a) Create alpha dictionary: self.alpha = OrderedDict() (line 55)

   b) Determine alpha_shape (lines 62-103):
      - If sparse_features_alpha: only unstable neurons in this layer
      - Otherwise: full layer shape

   c) For each start_node (lines 107-162):
      - Determine spec dimension size (output shape of start node)
      - Check for sparse_spec_alpha option
      - Create alpha tensor with shape:
        [alpha_size, spec_dim, batch_size, *layer_shape]

      For ReLU: alpha_size=2:
        - alpha[0]: slope for computing LOWER bounds
        - alpha[1]: slope for computing UPPER bounds

   d) Initialize from init_d: alpha.data.copy_(alpha_init.data) (line 157)

6. ALPHA USAGE IN BACKWARD PASS:
   File: operators/relu.py:622-706 (_backward_relaxation)

   When opt_stage in ['opt', 'reuse']:
     - Select alpha by index (line 646-647)
     - Use alpha[0] as lb_lower_d for lower bound (line 650)
     - Use alpha[1] as ub_lower_d for upper bound (line 652)
     - Apply masking for stable neurons (line 666)

7. OPTIMIZATION LOOP:
   File: optimized_bounds.py:488-700

   - Creates optimizer with alpha parameters
   - Runs iteration steps of gradient descent
   - Clips alpha to valid range [0,1] via clip_alpha() (relu.py:417-419)
   - Keeps track of best bounds

ALPHA TENSOR SHAPE EXPLANATION:
-------------------------------
Shape: [alpha_size, spec_dim, batch_size, *layer_shape]

- alpha_size (dim 0): 2 for ReLU
  [0] = slope for computing lower bounds
  [1] = slope for computing upper bounds

- spec_dim (dim 1): Size of start node's output
  For final node: number of output neurons
  For intermediate: shape of that layer's pre-activation

- batch_size (dim 2): Number of inputs in batch

- layer_shape (dims 3+): Shape of THIS ReLU layer's neurons
  e.g., for 8-neuron layer: shape is [8]
  e.g., for conv layer 32x7x7: shape is [32, 7, 7]

EXAMPLE FOR SIMPLE NETWORK:
---------------------------
Network: input(4) -> fc1(8) -> relu1 -> fc2(6) -> relu2 -> fc3(2)

For relu1 (8 neurons):
  Start nodes: [fc2 (pre-relu2), fc3 (output)]
  Alpha shapes:
    - alpha['fc2']: [2, 6, batch, 8]  <- 6 neurons in fc2 output
    - alpha['fc3']: [2, 2, batch, 8]  <- 2 neurons in final output

For relu2 (6 neurons):
  Start nodes: [fc3 (output)]
  Alpha shapes:
    - alpha['fc3']: [2, 2, batch, 6]  <- 2 neurons in final output

Total alpha parameters = 2*(6*8 + 2*8) + 2*(2*6) = 2*(48+16) + 24 = 152
(assuming batch=1)

DEFAULT INITIALIZATION:
-----------------------
Alpha is initialized from init_d, which comes from _relu_lower_bound_init():

- 'adaptive' (default): alpha = 1 if upper_d > 0.5, else 0
  (upper_d is the upper bound slope from triangle relaxation)

- 'same-slope': alpha = upper_d (same slope for upper and lower)
- 'zero-lb': alpha = 0 for unstable, 1 for always-positive
- 'one-lb': alpha = 1 for unstable

================================================================================
"""
    print(explanation)


if __name__ == "__main__":
    # First print the explanation
    explain_alpha_flow()

    # Then run the trace
    print("\n\n")
    bounded_model, lb, ub = trace_alpha_creation()
