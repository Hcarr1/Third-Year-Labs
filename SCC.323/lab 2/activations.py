import math

def ReLU(v) -> float:
    """
    function to apply rectified Linear Unit (ReLU) activation function
    :param v:
    :return: float
    """
    if v <= 0:
        return 0
    else:
        return v

def Heaviside(v) -> float:
    """
    function to apply heaviside activation function
    :param v:
    :return: float
    """
    if v >= 0:
        return 1
    else:
        return 0

def Sigmoid(v) -> float:
    """
    function to apply sigmoid activation function
    :param v:
    :return: float
    """
    return 1 / (1 + math.exp(-v))


def Tanh(v) -> float:
    """
    function to apply tanh activation function
    :param v:
    :return: float
    """
    return math.tanh(v)

def PReLU(v, a=0.01) -> float:
    """
    function to apply parametric ReLU activation function
    :param v:
    :param a: Parameter, slope for negative values, defaulted as LeakyRelu slope
    :return: float
    """
    if v <= 0:
        return a * v
    else:
        return v

def LeakyReLU(v) -> float:
    """
    function to apply leaky ReLU activation function
    :param v:
    :return: float
    """
    return PReLU(v, a=0.01)

def ParametrisedSigmoid(v, a=1) -> float:
    """
    function to apply parametrised sigmoid activation function
    :param v:
    :param a: Parameter, controls the slope, defaulted to 1
    :return: float
    """
    return 1 / (1 + math.exp(-v / a))

def SoftSign(v) -> float:
    """
    function to apply softsign activation function
    :param v:
    :return: float
    """
    return v / (1+ abs(v))

def SiLU(v) -> float:
    """
    function to apply Sigmoid Linear Unit (SiLU) activation function
    :param v:
    :return: float
    """
    return v / (1 + Sigmoid(v))

def SoftPlus(v) -> float:
    """
    function to apply softplus activation function
    :param v:
    :return: float
    """
    return math.log(1 + math.exp(v))

def SELU(v, a=1, scale=1) -> float:
    """
    function to apply Scaled Exponential Linear Unit (SELU) activation function
    :param v:
    :param a: Parameter, defaulted to 1
    :param scale: Parameter, defaulted to 1
    :return: float
    """
    if v > 0:
        return scale * v
    else:
        return scale * (a * (math.exp(v) - 1))

def ELiSH(v) -> float:
    """
    function to apply Exponential Linear Sigmoid Squashing (ELISH) activation function
    :param v:
    :return: float
    """
    if v >= 0:
        return v / (1 + math.exp(-v))
    else:
        return (math.exp(v) - 1) / (1 + math.exp(-v))

def SobolevaModTanh(v, alphas) -> float:
    """
    function to apply Soboleva Modified Tanh activation function
    :param v:
    :param alphas: list of parameters [alpha1, alpha2, alpha3, alpha4]
    :return: float
    """
    return (math.exp(alphas[0] * v) - math.exp(-alphas[1] * v)) / (math.exp(alphas[2] * v) + math.exp(-alphas[3] * v))
