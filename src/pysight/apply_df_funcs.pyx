cpdef unsigned long long b22(str x) except? -2:
    return int(x, 2)

cpdef unsigned long long b16(str x) except? -2:
    return int(x, 16)

cpdef str hextobin(str h):
    return bin(int(h, 16))[2:].zfill(len(h) * 4)
