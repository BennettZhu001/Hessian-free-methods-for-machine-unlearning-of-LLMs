# Method, assumptions, and limitations

## Deletion objective

The full training multiset is split into retained indices $R$ and deletion
indices $D$. The desired reference model minimizes the retained objective

$$
F_R(\theta)=|R|^{-1}\sum_{i\in R}\ell_i(\theta)
             + (\lambda/2)\|\theta\|_2^2.
$$

At the full-data parameters $\hat\theta$, a second-order Taylor approximation
to the stationarity equation $\nabla F_R(\theta_R^*)=0$ yields

$$
\theta_R^* - \hat\theta \approx
-[\nabla^2F_R(\hat\theta)+\gamma I]^{-1}
  \nabla F_R(\hat\theta).
$$

The damping parameter $\gamma>0$ improves conditioning and makes the operator
positive definite in the default last-layer experiment.

## Matrix-free computation

CG only requires products with the linear operator. For a vector $v$, PyTorch
autograd evaluates

$$
v \mapsto \nabla^2F_R(\hat\theta)v + \gamma v
$$

by differentiating the inner product between $\nabla F_R$ and $v$. Therefore
storage is linear in the number of selected parameters rather than quadratic.

## Default: language-model head

The transformer is first trained end to end. During unlearning, its learned
hidden representation is fixed and only the output projection is updated. For
multinomial cross-entropy with L2 regularization, this conditional objective is
convex. The damped Hessian is positive definite, matching standard CG
assumptions.

## Experimental: all parameters

When `--unlearn-parameters all` is used, the retained objective is non-convex.
Its Hessian can be indefinite, ordinary CG may break down, and a one-step
quadratic model may be inaccurate for large deletions. Damping can help locally
but does not create an unlearning guarantee.

## What the experiment does not establish

- It does not certify that all information about deleted records is absent.
- Parameter closeness to one retrained model is neither necessary nor sufficient
  for distributional equivalence in a non-identifiable neural network.
- A small retain loss does not rule out membership inference or data extraction.
- A synthetic benchmark does not establish scalability to modern LLMs.
- From-scratch retraining is optimizer- and seed-dependent for a non-convex model.

The experiment is best read as a transparent numerical test of a local
second-order deletion update and the engineering primitives needed to scale it.
