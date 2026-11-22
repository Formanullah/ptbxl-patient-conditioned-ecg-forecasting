# Equations and their code operations

Shapes are `[B,T,12]`, where B is batch size, T=700 observed time steps,
F=300 future steps, H is recurrent hidden width and S is static-feature width.
All waveform metrics below use inverse-scaled physical mV. The alternative
500/500 task has its own protocol and cannot share error rankings with 700/300.

## Static conditioning and scaling

For lead l, training-history mean μ_l and population standard deviation σ_l:

$$\mu_l=\frac{1}{N_{tr}T}\sum_{n,t}x_{ntl},\qquad
\sigma_l=\sqrt{\frac{1}{N_{tr}T}\sum_{n,t}(x_{ntl}-\mu_l)^2}.$$

Use $z=(x-\mu_l)/\sigma_l$ and inverse $x=z\sigma_l+\mu_l$; a constant lead
uses scale 1. `waveform_scaler` and `inverse_scale` in `src/preprocessing.py`
implement these operations. Both filters use the raw-history statistics.

For static vector s, embedding $e=\operatorname{ReLU}(W_s s+b_s)$ is repeated
at each observed step: $u_t=[z_t;e]$. `self.static`, `expand`, and `torch.cat`
in each independent model implement this. With F1 and embedding width 16, the
temporal input has 28 channels. Imputation/scaling is training-fitted; missingness
flags are not scaled. In CNN-LSTM, concatenation happens before any convolution.

## LSTM gates

For input $u_t\in\mathbb R^D$, previous hidden $h_{t-1}\in\mathbb R^H$ and
cell $C_{t-1}\in\mathbb R^H$:

$$i_t=\sigma(W_{ii}u_t+b_{ii}+W_{hi}h_{t-1}+b_{hi}),$$
$$f_t=\sigma(W_{if}u_t+b_{if}+W_{hf}h_{t-1}+b_{hf}),$$
$$\widetilde C_t=\tanh(W_{ig}u_t+b_{ig}+W_{hg}h_{t-1}+b_{hg}),$$
$$C_t=f_t\odot C_{t-1}+i_t\odot\widetilde C_t,$$
$$o_t=\sigma(W_{io}u_t+b_{io}+W_{ho}h_{t-1}+b_{ho}),\qquad
h_t=o_t\odot\tanh(C_t).$$

Here σ(a)=1/(1+exp(−a)); tanh(a)=(exp(a)−exp(−a))/(exp(a)+exp(−a)).
i, f, and o are input, forget, and output gates; the candidate is new cell content;
⊙ is elementwise multiplication. W matrices are learned input/hidden transforms
and b are learned biases. Initial hidden/cell states are zeros by the PyTorch
default. Stacked layers consume the preceding layer's hidden sequence.

`torch.nn.LSTM(..., batch_first=True)` executes these gates; it packs gates in
i/f/g/o order in `weight_ih_l*` and `weight_hh_l*`. `hidden[-1]` selects the final
state of the top layer. See the [PyTorch reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).
The head computes $\hat y=\operatorname{reshape}(W_2\operatorname{Dropout}(
\operatorname{ReLU}(W_1h_T+b_1))+b_2)$ into `[B,F,12]`.

## Conv1D and pooling

For input X with channels c and observed position t:

$$Y_{b,o,t}=b_o+\sum_c\sum_{r=0}^{K-1}W_{o,c,r}X_{b,c,t+r-P}.$$

o indexes output channels, K is kernel size, P is zero padding, and b_o is a
learned bias. PyTorch Conv1d is cross-correlation, without reversing the kernel.
For stride a and dilation d:

$$L_{out}=\left\lfloor\frac{L_{in}+2P-d(K-1)-1}{a}+1\right\rfloor.$$

`Conv1d(28 or 12,32,7,padding=3)` and `Conv1d(32,64,5,padding=2)` both preserve
700 positions (stride/dilation 1). `ReLU(v)=max(0,v)`. `AvgPool1d(2)` averages
neighboring pairs and reduces length to 350. The transpose then supplies
`[B,350,64]` to the CNN-LSTM recurrent layer. Both convolutions see only observed
history; same padding is not a claim of per-timestep causal convolution.

## Loss, gradients, BPTT and AdamW

For normalized training predictions and targets with M=B×F×12 elements:

$$\mathcal L=\frac1M\sum_j(\hat z_j-z_j)^2,\qquad
\frac{\partial\mathcal L}{\partial\hat z_j}=\frac{2}{M}(\hat z_j-z_j).$$

`F.mse_loss`, `loss.backward`, `clip_grad_norm_` and `optimizer.step` in
`src/train.py` implement loss, backpropagation, gradient clipping and updates.
BPTT applies the chain rule through the recurrent computation graph; shared
parameters receive the sum of their contributions at all observed time steps.
For the full recurrent state $s_t=(h_t,C_t)$, a contribution through time
contains the ordered product
$\frac{\partial s_T}{\partial s_t}=
\frac{\partial s_T}{\partial s_{T-1}}\cdots
\frac{\partial s_{t+1}}{\partial s_t}$.
Clipping limits the gradient
norm to 1.0 before updating; it does not guarantee removal of vanishing gradients.

For gradient g and parameter θ, AdamW maintains
$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$ and
$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$, bias-corrects them using
$\hat m_t=m_t/(1-\beta_1^t)$ and $\hat v_t=v_t/(1-\beta_2^t)$, and updates
$\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)$.
η is learning rate; λ=1e-4 is decoupled weight decay. PyTorch defaults β1=.9,
β2=.999 and ε=1e-8 are used. No scheduler or early stopping changes these controlled
budgets. Normalized training MSE weights leads according to their fixed scales;
selection is nevertheless physical-mV patient-macro MAE.

## Physical-unit evaluation

Let e_j = prediction_j − truth_j in mV, over N records, F future samples and
L=12 leads; M=NFL:

$$MAE=\frac1M\sum_j|e_j|\quad[\mathrm{mV}],\qquad
MSE=\frac1M\sum_je_j^2\quad[\mathrm{mV}^2],\qquad RMSE=\sqrt{MSE}\quad[\mathrm{mV}].$$

**RMSE applies to LSTM and CNN-LSTM regression forecasts. It is a model-independent
evaluation metric, not a recurrent layer, gate, or architectural operation.**

Hand example: truth `[0,2]` mV and prediction `[1,4]` mV give errors `[1,2]`,
MAE=1.5 mV, MSE=2.5 mV² and RMSE=√2.5≈1.5811 mV. This example is arithmetic,
not a PTB-XL experiment.

For global truth mean $\bar y$, $R^2=1-\sum_j e_j^2/\sum_j(y_j-\bar y)^2$.
The example above has R²=−1.5. A constant truth has undefined R² and is reported null.
Per-lead Pearson is
$r_l=\sum_j(p_{jl}-\bar p_l)(y_{jl}-\bar y_l)/\sqrt{\sum_j(p_{jl}-\bar p_l)^2
\sum_j(y_{jl}-\bar y_l)^2}$, aggregating records/time within that lead. A constant
prediction or truth gives null, not an invented zero. Correlation can be high
despite voltage bias and is not sufficient evidence of waveform accuracy.

For patient p with record set R_p, define each record's MAE over its F×12 values,
then $MAE_{patient}=\frac1P\sum_p\frac1{|R_p|}\sum_{r\in R_p}MAE_r$.
P is the number of patients. This gives equal weight to patients despite repeated
records and is the validation selection criterion. `metric_summary` implements
these definitions. Per-lead tables use the same target and physical units.

`bootstrap_pair` first pairs the same ECG/patient IDs, averages B−A differences
within patients, then resamples patients. Negative differences favor B.
Intervals are descriptive after validation selection, not a correction for
multiple comparisons, and do not measure training-seed variability.
