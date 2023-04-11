import time


class LossMetric:
    """ A class for logging the losses of each module during training.

    Methods
    -------
    update():
        Update the average of a given loss at the given epoch.
    """

    def __init__(self):
        self.sum = None
        self.avg = None
        self.count = None
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_dict():
    """ Initializes the module losses as LossMetric() objects.

    Returns
    -------
    A dictionary containing the loss objects
    """

    loss_D_fake = LossMetric()
    loss_D_real = LossMetric()
    loss_D = LossMetric()
    loss_G = LossMetric()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G': loss_G}


def update_losses(model, loss_meter_dict, count):
    """ Iterates through the loss dictionary and updates the losses.

    Returns
    -------
    The updated dictionary
    """

    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

    return loss_meter_dict


def get_timestamp():
    """ Returns current timestamp. """

    # Get the current time in seconds since the epoch
    current_time = time.time()
    # Convert the current time to a struct_time object
    time_struct = time.localtime(current_time)
    # Extract the day, month, year, hours, and minutes from the time_struct object
    day = time_struct.tm_mday
    month = time_struct.tm_mon
    year = time_struct.tm_year
    # Construct the timestamp string in the format DAY:MONTH:YEAR
    timestamp = f"{day:02d}:{month:02d}:{year}"

    return timestamp