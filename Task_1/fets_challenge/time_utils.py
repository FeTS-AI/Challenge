from collections import namedtuple
from logging import getLogger
import warnings

import numpy as np
import pandas as pd

## COLLABORATOR TIMING DISTRIBUTIONS
# These data are derived from the actual timing information in the real-world FeTS information
# They reflect a subset of the institutions involved.
# Tuples are (mean, stddev) in seconds

# time to train one patient
TRAINING_TIMES = [(6.710741331207654, 0.8726112813698301),
                  (2.7343911917098445, 0.023976155580152165),
                  (3.173076923076923, 0.04154320960517865),
                  (6.580379746835443, 0.22461890673025595),
                  (3.452046783625731, 0.47136389322749656),
                  (6.090788461700995, 0.08541499003440205),
                  (3.206933911159263, 0.1927067498514361),
                  (3.3358208955223883, 0.2950567549663471),
                  (4.391304347826087, 0.37464538999161057),
                  (6.324805129494594, 0.1413885448869165),
                  (7.415133477633478, 1.1198881747151301),
                  (5.806410256410255, 0.029926699295169234),
                  (6.300204918032787, 0.24932319729777577),
                  (5.886317567567567, 0.018627858809133223),
                  (5.478184991273998, 0.04902740607167421),
                  (6.32440159574468, 0.15838847558954935),
                  (20.661918328585003, 6.085405543890793),
                  (3.197901325478645, 0.07049966132127056),
                  (6.523963730569948, 0.2533266757118492),
                  (2.6540077569489338, 0.025503099659276184),
                  (1.8025746183640918, 0.06805805332403576)]

# time to validate one patient
VALIDATION_TIMES = [(23.129135113591072, 2.5975116854269507),
                    (12.965544041450777, 0.3476297824941513),
                    (14.782051282051283, 0.5262660449172765),
                    (16.444936708860762, 0.42613177203005187),
                    (15.728654970760235, 4.327559980390658),
                    (12.946098012884802, 0.2449927822869217),
                    (15.335950126991456, 1.1587597276712558),
                    (24.024875621890544, 3.087348297794285),
                    (38.361702127659576, 2.240113332190875),
                    (16.320970580839827, 0.4995108101783225),
                    (30.805555555555554, 3.1836337269688237),
                    (12.100899742930592, 0.41122386959584895),
                    (13.099897540983607, 0.6693132795197584),
                    (9.690202702702702, 0.17513593019922968),
                    (10.06980802792321, 0.7947848617875114),
                    (14.605333333333334, 0.6012305898922827),
                    (36.30294396961064, 9.24123672148819),
                    (16.9130060292851, 0.7452868131028928),
                    (40.244078460399706, 3.7700993678269037),
                    (13.161603102779575, 0.1975347910041472),
                    (11.222161868549701, 0.7021223062972527)]

# time to download the model
DOWNLOAD_TIMES = [(112.42869743589742, 14.456734719659513),
                  (117.26870618556701, 12.549951446132013),
                  (13.059666666666667, 4.8700489616521185),
                  (47.50220338983051, 14.92128656898884),
                  (162.27864210526315, 32.562113378948396),
                  (99.46072058823529, 13.808785580783224),
                  (33.6347090909091, 25.00299299660141),
                  (216.25489393939392, 19.176465340447848),
                  (217.4117230769231, 20.757673955585453),
                  (98.38857297297298, 13.205048376808929),
                  (88.87509473684209, 23.152936862511545),
                  (66.96994262295081, 16.682497150763503),
                  (36.668852040816326, 13.759109844677598),
                  (149.31716326530614, 26.018185409516104),
                  (139.847, 80.04755583050091),
                  (54.97624444444445, 16.645170929316794)]

# time to upload the model
UPLOAD_TIMES = [(192.28497409326425, 21.537450985376967),
                (194.60103626943004, 24.194406902237056),
                (20.0, 0.0),
                (52.43859649122807, 5.047207127169352),
                (182.82417582417582, 14.793519078918195),
                (143.38059701492537, 7.910690646792151),
                (30.695652173913043, 9.668122350904568),
                (430.95360824742266, 54.97790476867727),
                (348.3174603174603, 30.14347985347738),
                (141.43715846994536, 5.271340868190727),
                (158.7433155080214, 64.87526819391198),
                (81.06086956521739, 7.003461202082419),
                (32.60621761658031, 5.0418315093016615),
                (281.5388601036269, 90.60338778706557),
                (194.34065934065933, 36.6519776778435),
                (66.53787878787878, 16.456280602190606)]

logger = getLogger(__name__)
# This catches PyTorch UserWarnings for CPU
warnings.filterwarnings("ignore", category=UserWarning)

# one week
# MINUTE = 60
# HOUR = 60 * MINUTE
# DAY = 24 * HOUR
# WEEK = 7 * DAY
MAX_SIMULATION_TIME = 7 * 24 * 60 * 60  #TODO check if this can be move to time_utils.py file

CollaboratorTimeStats = namedtuple('CollaboratorTimeStats',
                                    [
                                        'validation_mean',
                                        'training_mean',
                                        'download_speed_mean',
                                        'upload_speed_mean',
                                        'validation_std',
                                        'training_std',
                                        'download_speed_std',
                                        'upload_speed_std',
                                    ]
                                    )

def gen_collaborator_time_stats(collaborator_names, seed=0xFEEDFACE):

    np.random.seed(seed)

    stats = {}    
    for col in collaborator_names:
        ml_index    = np.random.randint(len(VALIDATION_TIMES))
        validation  = VALIDATION_TIMES[ml_index]
        training    = TRAINING_TIMES[ml_index]
        net_index   = np.random.randint(len(DOWNLOAD_TIMES))
        download    = DOWNLOAD_TIMES[net_index]
        upload      = UPLOAD_TIMES[net_index]

        stats[col] = CollaboratorTimeStats(validation_mean=validation[0],
                                           training_mean=training[0],
                                           download_speed_mean=download[0],
                                           upload_speed_mean=upload[0],
                                           validation_std=validation[1],
                                           training_std=training[1],
                                           download_speed_std=download[1],
                                           upload_speed_std=upload[1])
    return stats

def compute_times_per_collaborator(collaborator_name,
                                   training_collaborators,
                                   epochs_per_round,
                                   collaborator_data,
                                   collaborator_time_stats,
                                   round_num):
    np.random.seed(round_num)
    #times = {}
    #for col in collaborator_names:
    time = 0

    print(f'Computing time for collaborator {collaborator_name}')
    # stats
    stats = collaborator_time_stats[collaborator_name]

    # download time
    download_time = np.random.normal(loc=stats.download_speed_mean,
                                        scale=stats.download_speed_std)
    download_time = max(1, download_time)
    time += download_time

    # data loader
    #data = collaborator_data[collaborator_name]

    # validation time
    data_size = collaborator_data.get_valid_data_size()
    validation_time_per = np.random.normal(loc=stats.validation_mean,
                                            scale=stats.validation_std)
    validation_time_per = max(1, validation_time_per)
    time += data_size * validation_time_per

    # only if training
    if collaborator_name in training_collaborators:
        # training time
        data_size = collaborator_data.get_train_data_size()
        training_time_per = np.random.normal(loc=stats.training_mean,
                                                scale=stats.training_std)
        training_time_per = max(1, training_time_per)

        # training data size depends on the hparams
        data_size *= epochs_per_round
        time += data_size * training_time_per
        
        # if training, we also validate the locally updated model 
        data_size = collaborator_data.get_valid_data_size()
        validation_time_per = np.random.normal(loc=stats.validation_mean,
                                                scale=stats.validation_std)
        validation_time_per = max(1, validation_time_per)
        time += data_size * validation_time_per

        # upload time
        upload_time = np.random.normal(loc=stats.upload_speed_mean,
                                        scale=stats.upload_speed_std)
        upload_time = max(1, upload_time)
        time += upload_time
        
        #times[col] = time
    return time