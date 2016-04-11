@defstruct ccSGDOptions <: AbstractOptimizerOptions (
  (lr                :: Real = 0.01, lr > 0),
  (momentum          :: Real = 0.0, momentum >= 0),
  (grad_clip         :: Real = 0, grad_clip >= 0),
  (weight_decay      :: Real = 0.0001, weight_decay >= 0),
  (rescale_grad      :: Real = 1.0, rescale_grad >= 0),
  lr_scheduler       :: Any = nothing
)

#=doc
.. class:: ccSGD

   Stochastic gradient descent optimizer. Implemented in C++.

   .. function:: ccSGD(; kwargs...)

      :param Real lr: default `0.01`, learning rate.
      :param AbstractLearningRateScheduler lr_scheduler: default `nothing`, a
             dynamic learning rate scheduler. If set, will overwrite the `lr`
             parameter.
      :param Real momentum: default `0.0`, the momentum.
      :param Real rescale_grad: default `1`, rescaling factor of the gradient
      :param Real grad_clip: default `0`, if positive, will clip the gradient
             into the bounded range `[-grad_clip, grad_clip]`.
      :param Real weight_decay: default `0.0001`, weight decay is equivalent to
             adding a global l2 regularizer to the parameters.
=#
type ccSGD <: AbstractOptimizer
  opts  :: ccSGDOptions
  handle :: MX_handle
  state :: OptimizationState

  function ccSGD(; kwargs...)
    opts = ccSGDOptions(;kwargs...)
    opts.lr_scheduler = get_lr_scheduler(opts.lr_scheduler, opts.lr)
    handle = init_cc_optimizer("ccsgd", momentum = opts.momentum,
                               rescale_grad = opts.rescale_grad,
                               clip_gradient = opts.grad_clip)

    new(opts, handle)
  end
end

function create_state(self :: ccSGD, index :: Int, weight :: NDArray)
    return nothing
end

function update(self :: ccSGD, index :: Int, weight :: NDArray, grad :: NDArray, state :: Union{Void, NDArray})
  lr = get_learning_rate(self.opts.lr_scheduler, self.state)
  wd = self.opts.weight_decay

  @mxcall(:MXOptimizerUpdate, (MX_handle, Cint, MX_NDArrayHandle, 
          MX_NDArrayHandle, MX_float, MX_float), self.handle, index,
          weight.handle, grad.handle, lr, wd)
end
