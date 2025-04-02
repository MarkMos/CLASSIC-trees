def locate(xx,n,x):
    jl = 0
    ju = n+1
    while ju-jl > 1:
        jm = (ju+jl)//2
        if ((xx[n-1] > xx[0]) == (x > xx[jm-1])):
            jl = jm
        else:
            ju = jm
    return int(jl-1)