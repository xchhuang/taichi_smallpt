from math import pow

M_PI = 3.14159265358979323846


def clamp(x, low=0.0, high=1.0):
    if x > high:
        return high
    if x < low:
        return low
    return x


def to_byte(x, gamma=2.2):
    return int(clamp(255.0 * pow(x, 1.0 / gamma), 0.0, 255.0))


def write_ppm(w, h, Ls, fname="image.ppm"):
    with open(fname, 'w') as outfile:
        outfile.write('P3\n{0} {1}\n{2}\n'.format(w, h, 255))
        for L in Ls:
            outfile.write('{0} {1} {2} '.format(to_byte(L[0]), to_byte(L[1]), to_byte(L[2])))
