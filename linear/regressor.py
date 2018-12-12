import numpy as np

# object is the base class
class Regressor(object):
    """
    Base class for regressors
    """

    def fit(self, X, t, **kwargs):
        """
        estimates parameters given training dateset

        Parameters
        ----------
        X: (sample_size, n_fatures) np.ndarray
            training data input
        
        t: (sample_size, ) np.ndarray
            training data target
        """
        self.__check_input(X)
        self.__check_target(t)
        if hasattr(self, "__fit"):
            self.__fit(X, t, **kwargs)
        else:
            raise NotImplementedError

    def __check_input(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError("X(input) is not a np.ndarray")
        if X.ndim != 2:
            raise ValueError("X(input) is not two dimentional array")
        # The hasattr() method returns true if an object has the given named attribute 
        # and false if it does not.
        if hasattr(self, "n_features") and self.n_features != np.size(X, 1):
            raise ValueError(
                "mismatch in dimension 1 of X(input) "
                "(size {} is different from {})"
                .format(np.size(X, axis = 1), self.n_features))
    
    def __check_target(self, t):
        if not isinstance(t, np.ndarray):
            raise TypeError("t(target) must be np.ndarray")
        if t.ndim != 1:
            raise ValueError("t(target) mst be one dimensional array")

