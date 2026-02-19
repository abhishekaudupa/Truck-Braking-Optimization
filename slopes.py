import math

def get_slope_angle(x, slope_index, data_set_index):  

    if (data_set_index == 1): # Training
        if (slope_index == 1):
            alpha = 4 + math.sin(x/100) + math.cos(math.sqrt(2)*x/50) 

        if (slope_index == 2):
            alpha = 2.5 + math.sin(x/150) + math.cos(math.sqrt(5)*x/70)

        if (slope_index == 3):
            alpha = 3 - math.sin(x/187) + math.cos(math.sqrt(8)*x/100)

        if (slope_index == 4):
            alpha = 3 + math.sin(x/50) + math.cos(math.sqrt(3)*x/50)

        if (slope_index == 5):
            alpha = 6 + math.sin(x/80) + math.cos(math.sqrt(4)*x/70)

        if (slope_index == 6):
            alpha = 5 + math.sin(x/120) + 0.2*math.cos(math.sqrt(8)*x/50)

        if (slope_index == 7):
            alpha = 1.6 + math.sin(x/53) - math.cos(math.sqrt(12)*x/90)

        if (slope_index == 8):
            alpha = 3.2 - math.sin(x/100) + 2 * math.cos(math.sqrt(1)*x/200)

        if (slope_index == 9):
            alpha = 7 + math.sin(x/150) + math.cos(math.sqrt(22)*x/200)

        if (slope_index == 10):
            alpha = 7 + 2*math.sin(x/80) + math.cos(math.sqrt(2)*x/100);

    elif (data_set_index == 2): # Validation
        if (slope_index == 1):
            alpha = 6 - math.sin(x/100) + math.cos(math.sqrt(3)*x/50)

        if (slope_index == 2):
            alpha = math.exp(math.sin(x/30) + math.cos(x/500))

        if (slope_index == 3):
            alpha = 9 - math.log(x+1.5) + math.sin(x/100)

        if (slope_index == 4):
            alpha = 1 + x**2/130000 - math.sin(x/100) + math.cos(x/1000)

        if (slope_index == 5):
            alpha = 5 + math.sin(x/50) + math.cos(math.sqrt(5)*x/50) 


    elif (data_set_index == 3): # Test
        if (slope_index == 1):
            alpha = 6 - math.sin(x/100) + math.cos(math.sqrt(7)*x/50) 

        if (slope_index == 2):
            alpha = 8 - x/150 - math.sin(x/100) - math.cos(x/250)

        if (slope_index == 3):
            alpha = 1 + math.exp(math.cos(x/100) + math.sin(math.sqrt(x)/50))

        if (slope_index == 4):
            alpha = 5 - x**2/200000 + math.sin(x/70)

        if (slope_index == 5):
            alpha = 4 + (x/1000) + math.sin(x/70) + math.cos(math.sqrt(7)*x/100)

    return alpha
