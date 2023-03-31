def string_to_list(x, leng):
    if ',' in x:
        x = x.split(',')
        res = [int(i) for i in x]
    else:
        res = [int(x) for i in range(leng-1)]

    return res
