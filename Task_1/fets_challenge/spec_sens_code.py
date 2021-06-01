import numpy as np

def check_shapes_same(output, target):
    if output.shape != target.shape:
        raise ValueError('Shapes of output {} and target {} do not match.'.format(output.shape, target.shape))


def check_classes_enumerated_along_correct_axis(array, axis, num_classes):
    if array.shape[axis] != num_classes:
        raise ValueError('The idx={} channel of array should enumerate classes, but its shape is {} and there are {} classes.'.format(axis, array.shape, num_classes))


def check_is_binary_single(array):
    unique_values = np.unique(array)
    if not set(unique_values).issubset(set([1.0, 0.0])):
        raise ValueError('The provided array is not binary as unique values are: {}'.format(unique_values))

def apply_threshold(array, threshold=0.5):
    
    over_threshold = array >= threshold
    
    bin_tensor = np.zeros_like(array)
    bin_tensor[over_threshold] = 1
    
    return bin_tensor


def binarize_output(output, class_list, modality, threshold=0.5, class_axis=1):
    if class_list == ['4', '1||4', '1||2||4']:
        check_classes_enumerated_along_correct_axis(array=output, axis=class_axis, num_classes=len(class_list))

        slices = [slice(None) for _ in output.shape]

        # select appropriate channel for modality, and convert floats to binary using threshold
        if modality == 'ET':
            slices[class_axis] = 0 
            binarized_output = apply_threshold(output[tuple(slices)], threshold=threshold)
        elif modality == 'TC':
            slices[class_axis] = 1
            binarized_output = apply_threshold(output[tuple(slices)], threshold=threshold)
        elif modality == 'WT':
            slices[class_axis] = 2
            binarized_output = apply_threshold(output[tuple(slices)], threshold=threshold)
        else:
            raise ValueError('Modality {} is not currently supported.'.format(modality))
          
    else:
        raise ValueError("Class list {} is not currently supported.".format(class_list))

    check_is_binary_single(binarized_output)
    
    return binarized_output


def brats_labels(output, 
                 target, 
                 class_list= ['4', '1||4', '1||2||4'], 
                 binarized=True, 
                 **kwargs):
    # take output and target and create: (output_<task>, lable_<task>)
    # for tasks in ['enhancing', 'core', 'whole']
    # these can be binary (per-voxel) decisions (if binarized==True) or float valued
    if binarized:
        output_enhancing = binarize_output(output=output, 
                                           class_list=class_list, 
                                           modality='ET')
        
        output_core = binarize_output(output=output, 
                                      class_list=class_list, 
                                      modality='TC')
        
        output_whole = binarize_output(output=output, 
                                       class_list=class_list, 
                                       modality='WT')
       
    # We detect specific use_cases here, and force a change in the code when another is wanted.
    # In all cases, we rely on the order of class_list !!!
    if list(class_list) == ['4', '1||4', '1||2||4']:
        # In this case we track only enhancing tumor, tumor core, and whole tumor (no background class).
    
        if not binarized:

            # enhancing signal is channel 0 because of known class_list with fused labels
            output_enhancing = output[:,0,:,:,:]

            # core signal is channel 1 because of known class_list with fused labels
            output_core = output[:,1,:,:,:]

            # whole signal is channel 2 because of known class_list with fused labels
            output_whole = output[:,2,:,:,:]
        
        
        # enhancing signal is channel 0 because of known class_list with fused labels
        target_enhancing = target[:,0,:,:,:]
        
        # core signal is channel 1 because of known class_list with fused labels
        target_core = target[:,1,:,:,:]
        
        # whole signal is channel 2 because of known class_list with fused labels
        target_whole = target[:,2,:,:,:]
    else:
        raise ValueError('No implementation for this model class_list: ', class_list)

    check_shapes_same(output=output_enhancing, target=target_enhancing)
    check_shapes_same(output=output_core, target=target_core)
    check_shapes_same(output=output_whole, target=target_whole)

    return {'outputs': {'ET': output_enhancing, 
                        'TC': output_core,
                        'WT': output_whole},
            'targets': {'ET': target_enhancing, 
                        'TC': target_core, 
                        'WT': target_whole}}
