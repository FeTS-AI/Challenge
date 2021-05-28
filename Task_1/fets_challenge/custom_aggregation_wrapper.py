from openfl.component.aggregation_functions import AggregationFunctionInterface


# extends the openfl agg func interface to include challenge-relevant information
class CustomAggregationWrapper(AggregationFunctionInterface):
    def __init__(self, func):
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
             db_iterator,
             tensor_name,
             fl_round,
             *__):
        return self.func(local_tensors,
                         db_iterator,
                         tensor_name,
                         fl_round,
                         self.collaborators_chosen_each_round,
                         self.collaborator_times_per_round)
