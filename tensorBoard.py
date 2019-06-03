def launchTensorBoard():
    import os
    tensorBoardPath = "C:\\Users\\anto\\Documents\\SelfDrivingCar\\logs"
    os.system('tensorboard --logdir=' + tensorBoardPath)
    return

import threading
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()