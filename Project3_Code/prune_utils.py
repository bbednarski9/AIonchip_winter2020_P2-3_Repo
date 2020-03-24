# Not using this function. Will be replaced by prune_weights_in_training_perc 
# and prune_weights_in_training_thresh
def pre_prune_weights(self):
    # get weights in dict {name: torch.Tensor}
    state_dict = self.net.state_dict()
    # ================================================================ #
    # YOUR CODE HERE:
    #   1.find prunable variables i.e. kernel weight/bias
    #   2.prune parameters based on your threshold, calculated based on input argument percentage
    #   example pseudo code for step 2-3:
    #       for name, var in enumerate(state_dict):
    #           # construct pruning mask
    #           mask = var < threshold
    #           new_var = var[var < threshold]
    #           state_dict[name] = new_var
    # ================================================================ #
    pass

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

# this function only defines pruning functionality for percentage based pruning
# verify that percentage based pruning is uncommented in mode.py (directions in repo.)
def prune_weights_in_training_perc(self):
# get weights in dict {name: torch.Tensor}
    perc_prune = 0.10 # percentage of weights, biases and feature maps values to prune in layer
    if perc_prune > 1 or perc_prune < 0:
        raise ValueError('Error- pass a value between 0 and 1')
    nonzero_sum = 0
    total_sum = 0
    state_dict = self.net.state_dict()
    for key, value in state_dict.items():
        if ("weight" in key or "bias" in key):
            tester = abs(value.flatten())
            values, indices = tester.sort()
            thresh_index = int(values.shape[0]*perc_prune)
            thresh_value = values.data[thresh_index].item()
            mask = abs(value) > thresh_value
            new_value =  mask*value
            state_dict[key] = new_value
            nonzero_sum += mask.sum().item()
            total_sum += value.numel()
    self.net.load_state_dict(state_dict)
    #print('Percentage remaining:', nonzero_sum/total_sum)
    return
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

# this function only defines pruning functionality for threshold based pruning
# verify that threshold based pruning is uncommented in mode.py (directions in repo.)
def prune_weights_in_training_thresh(self):
# get weights in dict {name: torch.Tensor}
    #print("Made it into thresh-based pruning")
    state_dict = self.net.state_dict()
    threshold = 0.0000025
    # ================================================================ #
    # YOUR CODE HERE:
    #   you can reuse code for pre_prune_weights here
    #       -> make sure pruned weights not recovered
    #   or reselect threshold dynamically
    #       -> make sure pruned percentage same
    # ================================================================ #
    nonzero_sum = 0
    total_sum = 0
    for key, value in state_dict.items():
        if ("weight" in key or "bias" in key):  
            mask = abs(value) > threshold 
            new_value =  mask*value
            state_dict[key] = new_value
            nonzero_sum += mask.sum().item()
            total_sum += value.numel()
    self.net.load_state_dict(state_dict)
    #print('Percentage remaining:', nonzero_sum/total_sum)
    return

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

