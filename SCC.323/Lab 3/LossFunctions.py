
def MSE_loss(y, y_hat, n):
    """
    Mean Squared Error Loss Function
    :param y: true labels
    :param y_hat: predicted labels
    :param n: number of samples
    :return: Mean squared error loss
    """
    total = sum((y_true - y_pred) ** 2 for y_true, y_pred in zip(y, y_hat))

    return total / n