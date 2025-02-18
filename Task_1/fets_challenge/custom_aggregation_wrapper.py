from openfl.interface.aggregation_functions.experimental import PrivilegedAggregationFunction


# extends the openfl agg func interface to include challenge-relevant information
class CustomAggregationWrapper(PrivilegedAggregationFunction):
    def __init__(self, func):
        super().__init__()
        self.func = func
        self.collaborators_chosen_each_round = None
        self.collaborator_times_per_round = None
    
    def set_state_data_for_round(self,
                                 collaborators_chosen_each_round,
                                 collaborator_times_per_round):
        self.collaborators_chosen_each_round = collaborators_chosen_each_round
        self.collaborator_times_per_round = collaborator_times_per_round

    # pass-through that includes additional information from the challenge experiment wrapper
    def call(self,
             local_tensors,
             tensor_db,
             tensor_name,
             fl_round,
             *__):
        return self.func(local_tensors,
                         tensor_db,
                         tensor_name,
                         fl_round,
                         self.collaborators_chosen_each_round,
                         self.collaborator_times_per_round)
