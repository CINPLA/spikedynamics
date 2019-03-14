import quantities as pq
import numpy as np

def poisson_continuity_correction(n, observed):
    """
    n : array
        Likelihood to observe n or more events
    observed : array
        Rate of Poisson process
    References
    ----------
    Stark, E., & Abeles, M. (2009). Unbiased estimation of precise temporal
    correlations between spike trains. Journal of neuroscience methods, 179(1),
    90-100.
    """
    if n.ndim == 0:
        n = np.array([n])
    assert n.ndim == 1
    from scipy.stats import poisson
    assert np.all(n >= 0)
    result = np.zeros(n.shape)
    if n.shape != observed.shape:
        observed = np.repeat(observed, n.size)
    for i, (n_i, rate) in enumerate(zip(n, observed)):
        if n_i == 0:
            result[i] = 1.
        else:
            rates = [poisson.pmf(j, rate) for j in range(n_i)]
            result[i] = 1 - np.sum(rates) - 0.5 * poisson.pmf(n_i, rate)
    return result


def hollow_kernel(kernlen, width, hollow_fraction=0.6, kerntype='gaussian'):
    '''
    Returns a hollow kernel normalized to it's sum
    Parameters
    ----------
    kernlen : int
        Length of kernel, must be uneven (kernlen % 2 == 1)
    width : float
        Width of kernel (std if gaussian)
    hollow_fraction : float
        Fractoin of the central bin to removed.
    Returns
    -------
    kernel : array
    '''
    if kerntype == 'gaussian':
        from scipy.signal import gaussian
        assert kernlen % 2 == 1
        kernel = gaussian(kernlen, width)
        kernel[int(kernlen / 2.)] *= (1 - hollow_fraction)
    else:
        raise NotImplementedError
    return kernel / sum(kernel)


def cch_convolve(cch, width, hollow_fraction, kerntype):
    import scipy.signal as scs
    kernlen = len(cch) - 1
    kernel = hollow_kernel(kernlen, width, hollow_fraction, kerntype)
    # padd edges
    len_padd = int(kernlen / 2.)
    cch_padded = np.zeros(len(cch) + 2 * len_padd)
    # "firstW/2 bins (excluding the very first bin) are duplicated,
    # reversed in time, and prepended to the cch prior to convolving"
    cch_padded[0:len_padd] = cch[1:len_padd+1][::-1]
    cch_padded[len_padd: - len_padd] = cch
    # # "Likewise, the lastW/2 bins aresymmetrically appended to the cch."
    cch_padded[-len_padd:] = cch[-len_padd-1:-1][::-1]
    # convolve cch with kernel
    result = scs.fftconvolve(cch_padded, kernel, mode='valid')
    assert len(cch) == len(result)
    return result


def cch_significance(t1, t2, binsize, limit, hollow_fraction, width,
                     kerntype='gaussian'):
    """
    Parameters
    ---------
    t1 : np.array, or neo.SpikeTrain
        First spiketrain, raw spike times in seconds.
    t2 : np.array, or neo.SpikeTrain
        Second spiketrain, raw spike times in seconds.
    binsize : float, or quantities.Quantity
        Width of each bar in histogram in seconds.
    limit : float, or quantities.Quantity
        Positive and negative extent of histogram, in seconds.
    kernlen : int
        Length of kernel, must be uneven (kernlen % 2 == 1)
    width : float
        Width of kernel (std if gaussian)
    hollow_fraction : float
        Fraction of the central bin to removed.
    References
    ----------
    Stark, E., & Abeles, M. (2009). Unbiased estimation of precise temporal
    correlations between spike trains. Journal of neuroscience methods, 179(1),
    90-100.
    English et al. 2017, Neuron, Pyramidal Cell-Interneuron Circuit Architecture
    and Dynamics in Hippocampal Networks
    """
    cch, bins = correlogram(t1, t2, binsize=binsize, limit=limit,
                            density=False)
    pfast = np.zeros(cch.shape)
    cch_smooth = cch_convolve(cch=cch, width=width,
                              hollow_fraction=hollow_fraction,
                              kerntype=kerntype)
    pfast = poisson_continuity_correction(cch, cch_smooth)
    # ppeak describes the probability of obtaining a peak with positive lag
    # of the histogram, that is signficantly larger than the largest peak
    # in the negative lag direction.
    ppeak = np.zeros(cch.shape)
    max_vals = np.zeros(cch.shape)
    cch_half_len = int(np.floor(len(cch) / 2.))
    max_vals[cch_half_len:] = np.max(cch[:cch_half_len])
    max_vals[:cch_half_len] = np.max(cch[cch_half_len:])
    ppeak = poisson_continuity_correction(cch, max_vals)
    return ppeak, pfast, bins, cch, cch_smooth


def transfer_probability(t1, t2, binsize, limit, hollow_fraction, width,
                         latency, winsize, kerntype='gaussian'):
    cch, bins = correlogram(t1, t2, binsize=binsize, limit=limit,
                            density=False)
    cch_s = cch_convolve(cch=cch, width=width,
                              hollow_fraction=hollow_fraction,
                              kerntype=kerntype)

    mask = (bins >= latency) & (bins <= latency + winsize)
    cmax = np.max(cch[mask])
    idx, = np.where(cch==cmax * mask)
    idx = idx if len(idx) == 1 else idx[0]
    pfast, = poisson_continuity_correction(cmax, cch_s[idx])
    cch_half_len = int(np.floor(len(cch) / 2.))
    max_pre = np.max(cch[:cch_half_len])
    ppeak, = poisson_continuity_correction(cmax, max_pre)
    ptime = float(bins[idx])
    trans_prob = sum(cch[mask] - cch_s[mask]) / len(t1)
    return trans_prob, ppeak, pfast, ptime, cmax


def correlogram(t1, t2=None, binsize=.001, limit=.02, auto=False,
                density=False):
    """Return crosscorrelogram of two spike trains.
    Essentially, this algorithm subtracts each spike time in `t1`
    from all of `t2` and bins the results with np.histogram, though
    several tweaks were made for efficiency.
    Originally authored by Chris Rodger, copied from OpenElectrophy, licenced
    with CeCill-B. Examples and testing written by exana team.
    Parameters
    ---------
    t1 : np.array, or neo.SpikeTrain
        First spiketrain, raw spike times in seconds.
    t2 : np.array, or neo.SpikeTrain
        Second spiketrain, raw spike times in seconds.
    binsize : float, or quantities.Quantity
        Width of each bar in histogram in seconds.
    limit : float, or quantities.Quantity
        Positive and negative extent of histogram, in seconds.
    auto : bool
        If True, then returns autocorrelogram of `t1` and in
        this case `t2` can be None. Default is False.
    density : bool
        If True, then returns the probability density function.
    See also
    --------
    :func:`numpy.histogram` : The histogram function in use.
    Returns
    -------
    (count, bins) : tuple
        A tuple containing the bin right edges and the
        count/density of spikes in each bin.
    Note
    ----
    `bins` are relative to `t1`. That is, if `t1` leads `t2`, then
    `count` will peak in a positive time bin.
    Examples
    --------
    >>> t1 = np.arange(0, .5, .1)
    >>> t2 = np.arange(0.1, .6, .1)
    >>> limit = 1
    >>> binsize = .1
    >>> counts, bins = correlogram(t1=t1, t2=t2, binsize=binsize,
    ...                            limit=limit, auto=False)
    >>> counts
    array([0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0])
    The interpretation of this result is that there are 5 occurences where
    in the bin 0 to 0.1, i.e.
    >>> idx = np.argmax(counts)
    >>> '%.1f, %.1f' % (abs(bins[idx - 1]), bins[idx])
    '0.0, 0.1'
    The correlogram algorithm is identical to, but computationally faster than
    the histogram of differences of each timepoint, i.e.
    >>> diff = [t2 - t for t in t1]
    >>> counts2, bins = np.histogram(diff, bins=bins)
    >>> np.array_equal(counts2, counts)
    True
    """
    if auto: t2 = t1
    lot = [t1, t2, limit, binsize]
    if any(isinstance(a, pq.Quantity) for a in lot):
        if not all(isinstance(a, pq.Quantity) for a in lot):
            raise ValueError('If any is quantity all must be ' +
                             '{}'.format([type(d) for d in lot]))
        dim = t1.dimensionality
        t1, t2, limit, binsize = [a.rescale(dim).magnitude for a in lot]
    # For auto-CCGs, make sure we use the same exact values
    # Otherwise numerical issues may arise when we compensate for zeros later
    if not int(limit * 1e10) % int(binsize * 1e10) == 0:
        raise ValueError('Time limit {} must be a '.format(limit) +
                         'multiple of binsize {}'.format(binsize) +
                         ' remainder = {}'.format(limit % binsize))
    # For efficiency, `t1` should be no longer than `t2`
    swap_args = False
    if len(t1) > len(t2):
        swap_args = True
        t1, t2 = t2, t1

    # Sort both arguments (this takes negligible time)
    t1 = np.sort(t1)
    t2 = np.sort(t2)

    # Determine the bin edges for the histogram
    # Later we will rely on the symmetry of `bins` for undoing `swap_args`
    limit = float(limit)

    # The numpy.arange method overshoots slightly the edges i.e. binsize + epsilon
    # which leads to inclusion of spikes falling on edges.
    bins = np.arange(-limit, limit + binsize, binsize)

    # Determine the indexes into `t2` that are relevant for each spike in `t1`
    ii2 = np.searchsorted(t2, t1 - limit)
    jj2 = np.searchsorted(t2, t1 + limit)

    # Concatenate the recentered spike times into a big array
    # We have excluded spikes outside of the histogram range to limit
    # memory use here.
    big = np.concatenate([t2[i:j] - t for t, i, j in zip(t1, ii2, jj2)])

    # Actually do the histogram. Note that calls to np.histogram are
    # expensive because it does not assume sorted data.
    count, bins = np.histogram(big, bins=bins, density=density)

    if auto:
        # Compensate for the peak at time zero that results in autocorrelations
        # by subtracting the total number of spikes from that bin. Note
        # possible numerical issue here because 0.0 may fall at a bin edge.
        c_temp, bins_temp = np.histogram([0.], bins=bins)
        bin_containing_zero = np.nonzero(c_temp)[0][0]
        count[bin_containing_zero] = 0#-= len(t1)

    # Finally compensate for the swapping of t1 and t2
    if swap_args:
        # Here we rely on being able to simply reverse `counts`. This is only
        # possible because of the way `bins` was defined (bins = -bins[::-1])
        count = count[::-1]

    return count, bins[1:]
