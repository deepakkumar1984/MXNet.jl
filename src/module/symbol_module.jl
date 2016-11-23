
"""
    Module

Module is a basic module that wraps a `SymbolicNode`. It is functionally the same
as the `FeedForward` model, except using the module API.

A current limitation is that it only supports one context.

# Parameters
* `symbol :: SymbolicNode`: The wrapped `SymbolicNode`
* `data_names :: Vector{Symbol}`:
"""
type Module <: AbstractModule
  symbol :: SymbolicNode
  data_names :: Vector{Symbol}
  label_names :: Vector{Symbol}
  aux_names :: Vector{Symbol}
  context :: Context

  binded :: Bool
  for_training :: Bool
  inputs_need_grad :: Bool
  params_initialized :: Bool
  optimizer_initialized :: Bool

  data_shapes :: Nullable{Vector{Tuple{Int}}}
  label_shapes :: Nullable{Vector{Tuple{Int}}}
  output_shapes :: Nullable{Vector{Tuple{Int}}}

  arg_arrays :: Nullable{Vector{NDArray}}
  aux_arrays :: Nullable{Vector{NDArray}}
  grad_arrays :: Nullable{Vector{NDArray}}
  params_dirty :: Bool

  executor :: Nullable{Executor}

  function Module(symbol::SymbolicNode, data_names::Vector{Symbol},
                  label_names::Vector{Symbol}, context :: Context)

    aux_names = list_auxiliary_states(symbol)
    return new(symbol, data_names, label_names, aux_names, context,
               false, false, false, false, false,
               Nullable{Vector{Tuple{Int}}}(),
               Nullable{Vector{Tuple{Int}}}(),
               Nullable{Vector{Tuple{Int}}}(),
               Nullable{Vector{NDArray}}(),
               Nullable{Vector{NDArray}}(),
               Nullable{Vector{NDArray}}(),
               false,
               Nullable{Executor}())
  end
end

function Module(symbol::SymbolicNode;
                data_names = [:data], label_names = [:softmax_label],
                context = mx.cpu())
  return Module(symbol, data_names, label_names, context)
end

### default API
isbinded(self::Module) = self.binded
allows_training(self::Module) = self.for_training
isinitialized(self::Module) = self.params_initialized
hasoptimizer(self::Module) = self.hasoptimizer

data_names(self::Module) = self.data_names
output_names(self::Module) = list_outputs(symbol)

function data_shapes(self::Module)
  !isbinded(self) && return Nullable{Vector{Tuple{Int}}}()
  return self.data_shapes
end

function label_shapes(self::Module)
  !isbinded(self) && return Nullable{Vector{Tuple{Int}}}()
  return self.label_shapes
end

function output_shapes(self::Module)
  !isbinded(self) && return Nullable{Vector{Tuple{Int}}}()
  return self.output_shapes
end

function get_params(self::Module)
  if !(isbinded(self) && isinitialized(self))
    return (Nullable{Dict{Symbol, NDArray}}(), Nullable{Dict{Symbol, NDArray}}())
  end
  if self.params_dirty
    sync_params_from_device(self)
  end
  return (Dict(name => data for (name, data) in zip()),
          Dict(name => data for (name, data) in zip()))
end

function init_params(self::Module; initializer=, arg_params=nothing,
                     aux_params=nothing, allow_missing=false, force_init=false)
  if isinitialized(self) && !force_init
    return
  end

  @assert isbinded(self) "Call `bind` before initialization"
end

function bind(self::Module, data_shapes, label_shapes = Vector{Vector{Int}}();
              for_training=true, inputs_need_grad=true, force_rebind=false,
              grad_req=GRAD_WRITE)
  if force_rebind
    reset_bind(self)
  end

  if isbinded(self)
    warn("Already bound, ignoring bind()")
    return
  end

  self.for_training = for_training
  self.inputs_need_grad = inputs_need_grad
  self.binded = true

  @assert !for_training && !inputs_need_grad

  @assert length(self.data_names)  == length(data_shapes)
  @assert length(self.label_names) == length(label_shapes)

  self.data_shapes = Nullable(data_shapes)
  self.label_shapes = Nullable(label_shapes)

  provided_shapes = merge(
      Dict(name => shape for zip(self.data_names,  data_shapes)),
      Dict(name => shape for zip(self.label_names, label_shapes)))

  arg_shapes, out_shapes, aux_shapes = infer_shape(self; provided_shapes...)
  @assert(!isa(arg_shapes, Void), "Information not enough to perform complete shape inference")

  # TODO: perform type inference

  arg_arrays = NDArray[mx.zeros(shape, ctx) for shape in arg_shapes]
  arg_names  = list_arguments(self.symbol)

  grad_arrays = Dict{Symbol,NDArray}()

  if grad_req != GRAD_NOP
    shapes = zip(arg_names, arg_shapes)

    # if not in provided data, should be parameters
    provided_data_names = [x[1] for x in keys(provided_shapes)]
    shapes = filter(x -> !in(x[1], provided_data_names), shapes)

    # Remove all gradients for nop params
    # if isa(grad_req, Dict{Symbol, GRAD_REQ})
    #  shapes = filter(x -> grad_req[x[1]] != GRAD_NOP,shapes)
    # end

    for (name, shape) in shapes
      grad_arrays[name] = mx.zeros(shape, ctx)
    end
  end

  aux_arrays = NDArray[mx.zeros(shape, ctx) for shape in aux_shapes]
  executor = mx.bind(self, ctx, arg_arrays, args_grad=grad_arrays, grad_req=grad_req, aux_states=aux_arrays)

  self.executor = Nullable{Executor}(executor)
end

##
# Internals
##

function sync_params_from_devices(self::Module)
  throw(MethodError(sync_params_from_devices, (self,)))
end
