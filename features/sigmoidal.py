import numpy as np

class SigmoidalFeatures(object):
    """
    Sigmoidal features

    1 / (1 + exp(m - x) @ c)
    """

    def __init__(self, mean, coef = 1):
        """
        construct sigmoidal features

        Parameters
        ----------
        mean : (n_features, n_dims) or (n_features, ) ndarray
              center of sigmoid function
        coef : (n_dim, ) ndarray or int or float
              coefficient to be multiplied with the distance
        """
        if mean.ndim == 1:
            mean = mean[:, None]
        else:
            assert mean.ndim == 2
        if isinstance(coef, (int, float)):
            if np.size(mean, 1) == 1:
                coef = np.array([coef])
            else:
                raise ValueError('mismatch of dimension')
        else:
            # in this case coef is a ndarray
            assert coef.ndim == 1
            assert np.size(mean, 1) == len(coef)
        self.__mean = mean
        self.__coef = coef

    def __sigmoid(self, x, mean):
        return 0

    def transform(self, x):
        """
        transform the input array with sigmoidal features

        Parameters
        ----------
        x: (sample_size, ndim) or (sample_size,) ndarray
            input array

        Returns
        -------
        output: (sample_size, n_features) ndarray
            sigmoidal features
        """

        if x.ndim == 1:
            x = x[:, None]
        else:
            assert x.ndim == 2
        assert np.size(x, axis = 1) == np.size(self.__mean, axis = 1)
        basis = [np.ones(len(x))]
        for m in self.__mean:
            basis.append(self.__sigmoid(x, m))
        return np.asarray(basis).transpose()