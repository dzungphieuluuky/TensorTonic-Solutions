import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    param, grad, m, v = np.array(param), np.array(grad), np.array(m), np.array(v)
    m_t = beta1 * m + (1 - beta1) * grad
    v_t = beta2 * v + (1 - beta2) * grad**2
    mt = m_t / (1 - beta1**t)
    vt = v_t / (1 - beta2**t)
    param = param - lr * (mt / (np.sqrt(vt) + eps))
    return param, m_t, v_t