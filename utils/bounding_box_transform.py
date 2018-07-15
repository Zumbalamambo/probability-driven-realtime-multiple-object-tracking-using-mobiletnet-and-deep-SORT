def to_tlwh(boundingbox):
    """Get current position in bounding box format `(top left x, top left y,
    width, height)`.

    Returns
    -------
    ndarray
        The bounding box.

    """
    ret = boundingbox
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    return ret

def to_tlbr(boundingbox):
    """Get current position in bounding box format `(min x, miny, max x,
    max y)`.

    Returns
    -------
    ndarray
        The bounding box.

    """
    ret = to_tlwh(boundingbox)
    print(ret)
    ret[2:] = ret[:2] + ret[2:]
    return ret