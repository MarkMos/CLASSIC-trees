from numpy import log

n_switch = 60
tiny = 1e-5
aln2I = 1/log(2.0)

def indexxx(n, arr, indx):
    if n > n_switch:
        indx = indexx(n, arr, indx)
    else:
        indx = indexsh(n, arr, indx)
    return indx

def indexx(n, arr, indx):
    '''
    Indexing or array arr[n] using numerical recipes heapsort.
    '''
    if n <= 1:
        return indx
    l = n//2
    ir = n-1
    while True:
        if l > 0:
            l -= 1
            indxt = indx[l]
            q = arr[indxt]
        else:
            indxt = indx[ir]
            q = arr[indxt]
            indx[ir] = indx[0]
            ir -= 1
            if ir == 0:
                indx[0] = indxt
                break
        i = l
        j = 2*l + 1
        while j <= ir:
            if j < ir and arr[indx[j]] < arr[indx[j+1]]:
                j += 1
            if q < arr[indx[j]]:
                indx[i] = indx[j]
                i = j
                j = 2*j+1
            else:
                break
        indx[i] = indxt
    return indx

def indexsh(n, arr,indx):
    '''
    Indexing or array arr[n] using shell sort.
    '''
    if n >= 2:
        log_nb2 = int(log(n)*aln2I+tiny)
        m = n
        for n_n in range(log_nb2):
            m = m//2
            k = n - m
            for j in range(int(k)):
                i=j
                done3 = False
                while done3 is False:
                    l = i + m
                    if arr[indx[l]] < arr[indx[i]]:
                        t_index = indx[i]
                        indx[i] = indx[l]
                        indx[l] = t_index
                        i = i - m
                        if i < 0:
                            done3 = True
                    else:
                        done3 = True
    return indx