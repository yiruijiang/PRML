import numpy as np

#python 3 自动继承object
class GaussianFeatures(object):
    """
    Gaussian Features

    Parameters
    ----------
    mean: (n_features, ndim) or (n_features,) ndarray
    places to locate gaussian function at

    var: float
        variance of the gaussian function
    """

    # python 中 __init__ 方法第一参数永远是self， 代表实例
    def __init__(self, mean, var):

        # be sure that mean is ndarray and not scalar
        if mean.ndim == 1:
            mean = mean[:, None]
        else:
            assert mean.ndim == 2
        
        # be sure that var has the right type
        assert isinstance(var, (int,float))

        #python 中 实例的参数想加就加
        self.__mean = mean
        self.__var = var

    def _gauss(self, x, mean):
        """
        compute the gaussian function

        Parameters
        ----------
        x: (sample_size, ndim) or (sample_size, )
            input array
        
        """
        result = np.exp(-0.5 * np.sum(np.square(x - mean), axis = -1) / self.__var)
        print("result shape",result.shape)
        return result

    def transform(self, x):
        """
        transform input array with gaussian features

        Parameters
        ----------
        x: (sample_size, ndim) or (sample_size, )
            input array
        
        Returns
        -------
        output : (sample_size, n_features)
            gaussian features
        """
        if x.ndim == 1:
            x = x[:, None]
        else:
            assert x.ndim == 2
        assert np.size(x, axis = 1) == np.size(self.__mean, axis = 1)

        basis = [np.ones(len(x))]

        for m in self.__mean:
            basis.append(self._gauss(x, m))
        
        return np.asarray(basis).transpose()
