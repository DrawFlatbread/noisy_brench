from .custom import TaskDataset


class Task(TaskDataset):
    def __init__(self, transform, mode, pred=..., probability=...):
        super(Task, self).__init__(transform, mode, pred, probability)
        self.annotation_path = 'data/task2/5/data/'


   