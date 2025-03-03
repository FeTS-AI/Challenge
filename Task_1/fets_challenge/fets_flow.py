import os
from copy import deepcopy
from typing import Union

import numpy as np
import torch as pt
import yaml

from sys import path
from openfl.federated import Plan
from pathlib import Path

from openfl.experimental.workflow.interface import FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator

from GANDLF.compute.generic import create_pytorch_objects
from GANDLF.config_manager import ConfigManager

#from .fets_challenge_model import inference, fedavg

class FeTSFederatedFlow(FLSpec):
    def __init__(self, model, rounds=3, **kwargs):
        super().__init__(**kwargs)
        self.fets_model = model
        self.n_rounds = rounds
        self.current_round = 1

    @aggregator
    def start(self):
        self.collaborators = self.runtime.collaborators
        self.next(self.initialize_collaborators, foreach='collaborators')

    @collaborator
    def initialize_collaborators(self):
        if isinstance(self.gandlf_config, str) and os.path.exists(self.gandlf_config):
            gandlf_conf = yaml.safe_load(open(self.gandlf_config, "r"))

        print(gandlf_conf)

        #gandlf_config_path = "/home/ad_tbanda/code/fedAI/Challenge/Task_1/gandlf_config.yaml"
        gandlf_config = Plan.load(Path(self.gandlf_config))
        print(gandlf_config)
        print(gandlf_config['weighted_loss'])

        gandlf_conf = ConfigManager(self.gandlf_config)

        (
            model,
            optimizer,
            train_loader,
            val_loader,
            scheduler,
            params,
        ) = create_pytorch_objects(
            gandlf_conf, train_csv=self.train_csv, val_csv=self.val_csv, device=self.device
        )
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.params = params
        self.device = self.device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = 1
        self.next(self.aggregated_model_validation)

    @collaborator
    def aggregated_model_validation(self):
        print(f'Performing aggregated model validation for collaborator {self.input}')
        print(f'Val Loader: {self.val_loader}')
        self.agg_validation_score = self.fets_model.validate(self.model, self.input, self.current_round, self.val_loader, self.params, self.scheduler)
        print(f'{self.input} value of {self.agg_validation_score}')
        self.next(self.train)

    @collaborator
    def train(self):
        print(f'Performing training for collaborator {self.input}')
        self.fets_model.train(self.model, self.input, self.current_round, self.train_loader, self.params, self.optimizer, self.epochs)
        self.metric = "Test"
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        self.local_validation_score = self.fets_model.validate(self.model, self.input, self.current_round, self.val_loader, self.params, self.scheduler)
        print(f'Doing local model validation for collaborator {self.input}:'
              + f' {self.local_validation_score}')
        self.next(self.join)

    @aggregator
    def join(self, inputs): 
        self.average_loss = sum(input.loss for input in inputs) / len(inputs)
        self.aggregated_model_accuracy = sum(
            input.agg_validation_score for input in inputs) / len(inputs)
        self.local_model_accuracy = sum(
            input.local_validation_score for input in inputs) / len(inputs)
        print(f'Average aggregated model validation values = {self.aggregated_model_accuracy}')
        print(f'Average training loss = {self.average_loss}')
        print(f'Average local model validation values = {self.local_model_accuracy}')
        print("Taking FedAvg of models of all collaborators")
        self.model = fedavg([input.model for input in inputs])

        self.next(self.internal_loop)

    @aggregator
    def internal_loop(self):
        if self.current_round == self.n_rounds:
            self.next(self.end)
        else:
            self.current_round += 1
            self.next(self.aggregated_model_validation, foreach='collaborators')

    @aggregator
    def end(self):
        print('This is the end of the flow')