import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    param = np.array(param, dtype=float)
    grad = np.array(grad, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)
    m = m*beta1+(1-beta1)*grad
    v = v*beta2+(1-beta2)*(grad**2)
    m1 = m/(1-(beta1**t))
    v1 = v/(1-(beta2**t))
    param -= lr*(m1/(v1**(0.5)+eps))

    return param, m, v
    