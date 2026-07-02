The new goal is to fit across a series of targets with each target having a different candidate library. 

I will explain how to calculate the new targets.

First calculate `J_rho`, `J_P`, and `J_Q` where `Q` is the nematic tensor defined earlier, I believe in `model_fitting/fitting/library.py`. However, this definition is not good because it depends on `P` which isn't accurate.

So define it as:
```
Q_{\alpha\beta} = 1/N \sum_{i}^N (p_i,\alpha p_i,\beta - delta_{\alpha\beta}/d)
```
where `d` is the dimensions `3`.

where `\alpha` and `\beta` are coordinate components. 

You will need to coares grain this as well, so account for that.

Now, you compute the `J` directly from these fields. You already know `J_rho`. But make sure its similar to the `J_m` present in the `model_fitting`, this is important as any discrepency would fail it. 

Calculate the other fluxes the same way. Its, of course, combining the `velocity` with the previous fields. If there is any difference between this and the method in `model_fitting`, raise it in the planning stage (i.e ask follow up or at least highlight it).

Now, here is what the fit targets.
```
Y_rho = gamma(J_rho - U0 P) -> P is polarization
Y_P = J_P/U0
Y_Q = J_Q
```

You will be working with tensors a lot for this, just a recommendation: use `tracel-ai/burn` in Rust for the tensor libraries if neccessary.

Here are the candidate libraries now:
Use these as the updated libraries.

Y_rho:

Candidates:
grad rho
rho grad(rho)
rho^2 grad rho
grad lap rho
(lap rho)grad rho
|grad rho|^2 grad rho
grad |grad rho|^2

Y_P:
Let `A = Q + rho/d * I` where `I` is the identity tensor
A
rho A
rho^2 A
rho^3 A

Y_Q:
Let `alpha_ijkl = delta_ij delta_kl + delta_ik delta_jl + delta_il delta_jk`
Do not coarse grain this.

Also let `F_rho = (THETA XI)_rho = the fit result for rho (vector)`

Candidates:
`P dot alpha`
`rho P dot alpha`
`rho ^2 P dot alpha`
`P dot II`
`F_rho I`




I'll also plan out some code-related things:
1. Make things modular and simple to add more `Y` in the case more targets are needed. 
2. Remove a lot of the rigorous validation of inputs, etc. I'm only using this, so don't raise errors that much as it makes the code not clean.
3. Try to use more dataclasses instead of raw `dict` to make the code easier.

I'll attach a paper with the formulas in the last section `Materials and Methods`, cross reference it with the plan. If there are any differences, raise it in the plan or include it in the plan at least (possible issues). Any possible uncertainty you should ask follow up questions. I will also manually edit the plan afters. 

You plan now, include assumptions and edge cases that you have to account for.
Be quite specific in the plan, and edit into PLAN.md (not codex planning mode).
