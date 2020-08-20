import tensorflow.keras.backend as K

# dice coefficient 
def diceCoefficient(y_true, y_pred, epsilon = 1):
    
    numerator = 2. * K.sum(y_true * y_pred) + epsilon
    denominator = K.sum(y_true) + K.sum(y_pred) + epsilon
    dice_coefficient = numerator / denominator
    return dice_coefficient

    