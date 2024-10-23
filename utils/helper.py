import os
import logging
class ModelSaver(object):
    """handle the number of saved models"""
    def __init__(self, max_count):
        self.save_list = []
        self.max_count = max_count

    def append(self, save_path):
        if len(self.save_list) < self.max_count:
            self.save_list.append(save_path)
        else:
            remove_path = self.save_list[0]
            del self.save_list[0]
            self.save_list.append(save_path)
            if os.path.exists(remove_path):
                os.remove(remove_path)

class RunningAverageTracker(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.average = 1.0 * self.sum / self.count

    def get_average(self):
        return self.average

    def get_count(self):
        return self.count
    
def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s",
                                     "%m-%d %H:%M:%S")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)