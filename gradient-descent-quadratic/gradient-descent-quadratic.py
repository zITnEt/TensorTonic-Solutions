def grd(a, b, c, x):
    return 2*a*x+b

def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    
    ans=x0

    for _ in range(steps):
        ans -= lr*grd(a, b, c, ans)

    return ans