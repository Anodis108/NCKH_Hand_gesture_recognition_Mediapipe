def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57: # 0 ~ 9
        number = key - 48    
    if key == 110:   # n
        mode = 0
    elif key == 107: # k
        mode = 1 
    elif key == 104: # h
        mode = 2
    return number, mode