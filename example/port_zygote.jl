# # Porting Yao with Zygote
using Zygote, YaoExtensions, Yao
using Random; Random.seed!(2)

# first include this patch file in your working directory.
include("zygote_patch.jl")

# ## setup a circuit
# wrap differentiable nodes with `Diff`, please try `print(c)` to see the marked nodes with `[∂]`.
c = variational_circuit(5) |> autodiff(:BP)
dispatch!(c, :random)

# ## define a loss function
h = rand_hermitian(1<<5)
function loss2(reg::ArrayReg, params::Vector{T}, circuit) where T
    reg = dispatch_diff_apply!(copy(reg), circuit, params)
    v = vec(reg.state)
    v' * h *v |> real
end

# ## obtaining the gradient
reg = rand_state(5)
params = rand!(parameters_of_diff(c))
grad_input, grad_params, _ = Zygote.gradient(loss2, reg, params, c)

# ## Check gradients
using Test

# to check gradients of input wave function, we can introduce a one-hot vector
# perturbation to get numerical finite-difference gradient.
η = 1e-5
perturb_input_real = η*product_state(bit"00111")
perturb_input_imag = im*η*product_state(bit"00111")

num_gradient = (loss2(reg + perturb_input_real, params, c) - loss2(reg - perturb_input_real, params, c)) / 2η +
                im*(loss2(reg + perturb_input_imag, params, c) - loss2(reg - perturb_input_imag, params, c)) / 2η
exact_gradient = select(grad_input, bit"00111").state[]
@test num_gradient ≈ exact_gradient

# check the gradient of circuit parameters
index = 1
perturb_params = (p=zeros(length(params)); p[index]=1; p)
loss_pos = loss2(reg, params+η*perturb_params, c)
loss_neg = loss2(reg, params-η*perturb_params, c)
@test grad_params[index] ≈ (loss_pos - loss_neg) / 2η
