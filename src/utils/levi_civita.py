def levi_civita_symbol(i, j, k):
    denominator = abs(j - i) * abs(k - i) * abs(k - j)
    numerator = (j - i) * (k - i) * (k - j)

    if denominator == 0:
        return 0
    return numerator / denominator
