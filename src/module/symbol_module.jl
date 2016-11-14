
"""
    Module

Module is a basic module that wraps a `SymbolicNode`. It is functionally the same
as the `FeedForward` model, except using the module API.

# Parameters
* `symbol :: SymbolicNode`: The wrapped `SymbolicNode`
* `data_names :: Vector{Symbol}`:
"""
type Module <: AbstractModule
  symbol :: SymbolicNode
  data_names :: Vector{Symbol}
  label_names :: Vector{Symbol}
  context :: Vector{Context}

  binded :: Bool
  for_training :: Bool
  inputs_need_grad :: Bool
  params_initialized :: Bool
  optimizer_initialized :: Bool

  data_shapes :: Nullable{Dict{Symbol, Vector{Int}}}
  label_shapes :: Nullable{Dict{Symbol, Vector{Int}}}
  output_shapes :: Nullable{Dict{Symbol, Vector{Int}}}

  arg_params :: Nullable{Dict{Symbol, NDArray}}
  aux_params :: Nullable{Dict{Symbol, NDArray}}
  params_dirty :: Bool

  executor :: Executor

  function Module(symbol::SymbolicNode, data_names::Vector{Symbol},
                  label_names::Vector{Symbol}, context :: Vector{Context})
    return new(symbol, data_names, label_names, context,
               false, false, false, false, false,
               Nullable{Dict{Symbol, Vector{Int}}}(),
               Nullable{Dict{Symbol, Vector{Int}}}(),
               Nullable{Dict{Symbol, Vector{Int}}}(), false)
  end
end

function Module(symbol::SymbolicNode, data_names::Vector{Symbol},
                label_names::Vector{Symbol}, context :: Context)
  return Module(symbol, data_names, label_names, [context])
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
  !isbinded(self) && return Nullable{Dict{Symbol, Vector{Int}}}()
  return self.data_shapes
end

function label_shapes(self::Module)
  !isbinded(self) && return Nullable{Dict{Symbol, Vector{Int}}}()
  return self.label_shapes
end

function output_shapes(self::Module)
  !isbinded(self) && return Nullable{Dict{Symbol, Vector{Int}}}()
  return self.output_shapes
end

function get_params(self::Module)
  if !(isbinded(self) && isinitialized(self))
    return (Nullable{Dict{Symbol, NDArray}}(), Nullable{Dict{Symbol, NDArray}}())
  end
  if self.params_dirty
    sync_params_from_device(self)
  end
  return (self.arg_params, self.aux_params)
end

function init_params(self::Module; initializer=, arg_params=nothing,
                     aux_params=nothing, allow_missing=false, force_init=false)
  if isinitialized(self) && !force_init
    return
  end

  @assert isbinded(self) "Call `bind` before initialization"
end

function bind(self::Module, data_shapes, label_shapes = nothing;
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

  self.data_shapes = Nullable(data_shapes)
  if label_shapes === nothing
    self.label_shapes = Nullable{Dict{Symbol, Vector{Int}}}()
  else
    self.label_shapes = Nullable(label_shapes)
  end


end



##
# Internals
##

function sync_params_from_devices(self::Module)
  throw(MethodError(sync_params_from_devices, (self,)))
end
