# Provided by the FeTS Initiative (www.fets.ai) as part of the FeTS Challenge 2021

# Contributing Authors (alphabetical):
# Micah Sheller (Intel)

class FeTSChallengeAssigner:
    def __init__(self, tasks, authorized_cols, training_tasks, validation_tasks, **kwargs):
        """Initialize."""
        self.training_collaborators = []
        self.tasks = tasks
        self.training_tasks = training_tasks
        self.validation_tasks = validation_tasks
        self.collaborators = authorized_cols

    def set_training_collaborators(self, training_collaborators):
        self.training_collaborators = training_collaborators


    def get_tasks_for_collaborator(self, collaborator_name, round_number):
        """Get tasks for the collaborator specified."""
        if collaborator_name in self.training_collaborators:
            return self.training_tasks
        else:
            return self.validation_tasks

    def get_collaborators_for_task(self, task_name, round_number):
        """Get collaborators for the task specified."""
        if task_name in self.validation_tasks:
            return self.collaborators
        else:
            return self.training_collaborators

    def get_all_tasks_for_round(self, round_number):
        return self.training_tasks

    def get_aggregation_type_for_task(self, task_name):
        """Extract aggregation type from self.tasks."""
        if 'aggregation_type' not in self.tasks[task_name]:
            return None
        return self.tasks[task_name]['aggregation_type']
