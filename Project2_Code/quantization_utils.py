import numpy as np
import tensorflow as tf
import math
'''
utility function
with with block.suppress_stdout_stderr():
    your code
To hide stdout/stderr output i.e. from Tensorflow initialzation    
'''
from . import suppress_stdout_stderr as block

# helper function to do smart exponential math
def pow2(val):
    if val > 0:
        return 2**val
    elif val < 0:
        return 1 / (2**abs(val))
    else:
        return 1

# this is our core quantization function
# takes an array||Tensor, total word legnth and fractional/mantissa legnth
# returns a Tensorflow.Tensor struct with quantized values in place of orig.
# 8/16 BIT FLOATING POINT CONVERSION
def tf_symbolic_convert(value, wl, fl):

    # if we're passed a numpy array, convert to a tensor
    if isinstance(value, np.ndarray):
        value = tf.convert_to_tensor(value, dtype=tf.dtypes.float32)

    # some math for constants in conversion to floating point
    ml = wl - fl - 1
    bias = int(pow2(fl-1) -1)
    #print("bias:", bias)
    max_val = pow2(bias)
    for bit in range(ml):
        max_val += pow2(bias-bit-1)
    min_abs_val = pow2(-ml-2)
    min_exp_int = -bias + 1

    # keep track so we can make sure we return the same shape
    orig_shape = value.get_shape()
    orig_shape_list = orig_shape.as_list()
    dim = np.prod(orig_shape_list[0:], dtype='int32')
    value = tf.reshape(value, [1,dim])
    new_shape = tf.shape(value)

    # Build some placeholder Tensors for math opperations
    sign = tf.sign(value)
    value_abs = tf.abs(value)
    value_abs_clipped = tf.clip_by_value(value_abs, min_abs_val, max_val, name='value_abs_clipped')
    zeros = tf.fill(new_shape, 0.0, name='zeros')
    ones = tf.fill(new_shape, 1.0, name='ones')
    twos = tf.fill(new_shape, 2.0, name='twos')
    exponent = tf.clip_by_value(tf.math.floor(tf.math.divide(tf.math.log(value_abs_clipped), tf.math.log(twos))), min_exp_int, bias)
    
    msb_possible_value = tf.math.pow(twos, exponent, name='total')
    mask = tf.cast(tf.math.greater_equal(tf.math.subtract(value_abs, msb_possible_value), zeros), dtype=tf.dtypes.float32)
    #mask = tf.math.ceil(tf.clip_by_value(tf.math.subtract(value, msb_possible_value), 0, 1, name='mask_clipped'), name='mask')
    #total = tf.math.pow(twos, exponent, name='total')
    total = tf.math.multiply(msb_possible_value, mask, name='bit_value_masked')

    # Buillding a list of Tensors to add potential conversion values to
    value_stack = []
    value_stack.append(value_abs)
    value_stack.append(value_abs)
    check_stack = []
    check_stack.append(total)
    check_stack.append(zeros)

    # we have 2^mantissa_length possibilities after the exponent has been calculated
    # loop over these possibiltiies and add them to the list
    for i in reversed(range(ml**2)):
        rem = i
        temp_total = total
        temp_exponent = exponent
        for j in reversed(range(ml)):
            bit_val = pow2(j)
            temp_exponent = tf.math.subtract(temp_exponent, ones)
            if rem - bit_val >= 0:
                rem -= bit_val
                temp_total = tf.math.add(temp_total, tf.math.pow(twos, temp_exponent))
                check_stack.append(temp_total)
                value_stack.append(value_abs)

    # stack the values in the list, and prepare to search for index of lowest difference
    value_stack = tf.stack(value_stack,axis=2, name='value_stack')
    check_stack = tf.stack(check_stack,axis=2,name='check_stack')
    diff = tf.math.abs(tf.subtract(check_stack, value_stack))
    indicies = tf.math.argmin(diff, axis=2, output_type=tf.dtypes.int32)

    # core command: returns the value from the stack that has to lowest difference from the original value
    val_fp = tf.gather_nd(check_stack, tf.stack([tf.tile(tf.expand_dims(tf.range(tf.shape(indicies)[0]), 1), 
        [1, tf.shape(indicies)[1]]), tf.transpose(tf.tile(tf.expand_dims(tf.range(tf.shape(indicies)[1]), 1), 
            [1, tf.shape(indicies)[0]])), indicies], 2))

    # clip and reshape
    val_fp = tf.clip_by_value(tf.math.multiply(val_fp, sign), -max_val, max_val, name='val_fp_signed_clipped')
    val_fp = tf.reshape(val_fp,orig_shape)

    return val_fp


# Alternative quantization method: built-in tensorflow function used for quantization.
# comment out function above and uncomment this one to run it
'''
def tf_symbolic_convert(value, wl, fl):
    if isinstance(value, np.ndarray):
        value = tf.convert_to_tensor(value, dtype=tf.float32)
        return value

    bias = pow2(fl-1)-1
    mant_length = wl-fl-1
    max_mant_val = 1 - 1 / pow2(mant_length)
    max_exp_val = pow2(fl) - 1 - bias
    max_val = (1 + max_mant_val)*pow2(max_exp_val)
    val_fp = tf.quantization.fake_quant_with_min_max_args(inputs = value, min = -max_val, max=max_val, num_bits = wl, narrow_range = False, name = None)

    return val_fp
'''

class Qnn:
    def __init__(self):
        print('Instantiates Qnn')
        pass

    # dtype convertion: basic functions           
    def to_fixedpoint(self, data_i, word_len, frac_len):
        return tf_symbolic_convert(data_i, word_len, frac_len)

    # utility function to convert symbolically or numerically
    def convert(self, data_i, word_len, frac_len, symbolic=False):
        if symbolic is True:
            data_q = self.to_fixedpoint(data_i, word_len, frac_len)
        else:
            with tf.Graph().as_default():
                data_q = self.to_fixedpoint(data_i, word_len, frac_len)
                with block.suppress_stdout_stderr():
                    with tf.Session() as sess:
                        data_q = sess.run(data_q)
        return data_q 

    # error measurement
    # difference function is used to search 
    def difference(self, data_q, data_origin):    
        '''
        Compute the difference before and after quantization
        Inputs：
        - data_q: a numpy array of quantized data
        - data_origin: a numpy array of original data
        Returns:
        - dif : numerical value of quantization error 
        '''
        # ================================================================ #
        # YOUR CODE HERE:
        #   implement mean squared error between data_q and data_origin
        # ================================================================ #
        assert (data_q.shape == data_origin.shape), "data_q and data_origin dimensions do not match."
        data_q_flat = data_q.flatten()
        data_origin_flat = data_origin.flatten()
        entry_count = data_q_flat.size
        dif = np.sum(np.square(np.subtract(data_q_flat, data_origin_flat))) / entry_count 
        #print("means square difference: ", dif)
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return dif

    # search policy
    def search(self, data_i, word_len):
        '''
        Search for the optimal fraction length that leads to minimal quantization error for data_i
        Inputs：
        - data_i : a numpy array of original data
        - word_len : word length of quantized data
        Returns:
        - fl_opt : fraction length (python built-in int data type) that leads to minimal quantization error
        '''
        # ================================================================ #
        # YOUR CODE HERE:
        #   compute quantization error for different fraction lengths
        #   and determine fl_opt
        # ================================================================ #
        # define limits of iteration
        print("starting a search.")
        exp_min = 2
        if (word_len == 16):
            # max exponent length is 7 if 16 bit conversion
            exp_max = 7
        else:
            # max exponent length is 6 if 8-bit conversion
            exp_max = 6

        # loop over all possible fraction lengths
        n_fl = word_len-1
        results = np.zeros(exp_max-exp_min+1)
        for fl in range(exp_min, exp_max+1):
            with block.suppress_stdout_stderr():
                with tf.Graph().as_default(), tf.Session() as sess:
                    data_q_tf = tf_symbolic_convert(data_i, word_len, fl)
                    data_q = sess.run(data_q_tf)
                    results[fl - 2] = self.difference(data_q, data_i)
        fl_opt = int(np.argmin(results)+exp_min)

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return fl_opt

    # granularity
    def apply(self, data_i, word_len):
        fl_opt = self.search(data_i, word_len)
        data_q = self.convert(data_i, word_len, fl_opt)
        return data_q, fl_opt








# Old code down here to keep track of our development
'''
# Depricated function - used for iterative quantization saving this as a record.
def find_best_exponent(value, max_exp_val, min_exp_val, max_mant_val):
    best_exp = min_exp_val
    min_diff = value*2
    for current_exp in range(min_exp_val, max_exp_val+1):
        current_diff = value - pow2(current_exp)
        if current_diff > 0 and current_diff <= min_diff:
            best_exp = current_exp
            min_diff = current_diff


    if best_exp < max_exp_val:
        max_best_exp_diff = abs(value - (pow2(best_exp) + max_mant_val*pow2(best_exp)))
        max_next_exp_diff = abs(value-pow2(best_exp+1))
        if max_best_exp_diff < max_next_exp_diff:
            return best_exp
        else:
            return best_exp + 1

    return best_exp

# Depricated function - used for iterative quantization saving this as a record.
def find_closest_value(value, exp_to_check, mant_length):
    base_value = pow2(exp_to_check)
    best_value = pow2(exp_to_check)
    min_diff = abs(value - best_value)
    over_val_check = []
    if best_value >= value:
        #print('best val', best_value)
        return best_value

    for x in range(mant_length):
        val_add = pow2(exp_to_check-(x+1))
        current_val = base_value + val_add
        current_diff = abs(value - current_val)
        #print('current_val', current_val)
        if current_val >= value and current_diff < min_diff:
            #print('upper:', current_val, current_diff)
            over_val_check.append(current_val)
            # best_value = current_val
            # min_diff = current_diff
        elif current_diff <= min_diff and current_val < value:
            #print('lower: ', current_val, current_diff)
            best_value = current_val
            min_diff = current_diff
            base_value = current_val

    if len(over_val_check) > 0:
        if (abs(over_val_check[-1] - value) < min_diff):
            return over_val_check[-1]
    return best_value

# Imperfect method: always approaches from the bottom
def tf_symbolic_convert(value, wl, fl):
    if isinstance(value, np.ndarray):
        value = tf.convert_to_tensor(value, out_type=tf.dtypes.float32)

    shape = tf.shape(value, out_type=tf.dtypes.int32)
    bias = pow2(fl-1)-1
    ml = wl - fl - 1
    max_val = pow2(bias)
    for bit in range(ml):
        max_val += pow2(bias-bit-1)

    sign_t = tf.sign(value)
    value_abs = tf.abs(value)
    value_abs_clipped = tf.clip_by_value(value_abs, 1, max_val, name='value_abs_clipped')
    exponent = tf.math.floor(tf.math.divide(tf.math.log(value_abs_clipped), tf.math.log(twos), name='exponent'))
    ones = tf.constant(1, dtype=tf.dtypes.float32, shape=shape, name='ones')
    twos = tf.constant(2, dtype=tf.dtypes.float32, shape=shape, name='twos')
    total = tf.constant(0, dtype=tf.dtypes.float32, shape=shape, name='total')


    for bit in ml:
        # returns 0 if negative, 1 if positive when subtracting the next mask value
        bit_value = tf.math.pow(twos, exponent, name='bit_value')
        mask = tf.math.ceil(tf.clip_by_value(tf.math.subtract(value_abs, bit_value), 0, 1, name='mask_clipped'), name='mask')
        bit_value_masked = tf.math.multiply(bit_value, mask, name='bit_value_masked')
        total = tf.math.add(total, bit_value_masked, name='total')
        value_abs = tf.math.subtract(value_abs, bit_value_masked, name='value_abs')
        exponent = tf.math.subtract(exponent, ones, name='exponent')

    val_fp = tf.multiply(total, sign, name='val_fp_signed')

    return val_fp


'''


'''
    Convert float numpy array to wl-bit low precision data with Tensorflow AP
    Inputs：
    - value : a numpy array of input data
    - wl : word length of the data format to convert
    - fl : fraction length (exponent length for floating-point)
    Returns:
    - val_fp : tf.Tensor as the symbolic expression for quantization 

    # ================================================================ #
    # YOUR CODE HERE: [note: tf.clip_by_value could be helpful]
    # ================================================================ #
'''
'''
    # old, original method
    if not isinstance(value, np.ndarray):
        print('here1')
        with block.suppress_stdout_stderr():
            shape_tmp = tf.shape(value)
            ones_tmp = tf.ones(shape_tmp, dtype=tf.dtypes.float32)
            value = tf.multiply(value, ones_tmp)
            with tf.Graph().as_default(), tf.Session() as sess:
                value_np = sess.run(value)
        print('here2')
    else:
        value_np = value
        bias = pow2(fl-1)-1
        mant_length = wl-fl-1
        max_mant_val = 1 - 1 / pow2(mant_length)
        max_exp_val = pow2(fl) - 1 - bias
        min_exp_val = -bias
        max_val = (1 + max_mant_val)*pow2(max_exp_val)
        original_dims = value_np.shape
        value_np = value_np.flatten()
        quantized_vals = []
        for val in value_np:
            if val >= max_val:
                #print('best val', max_val)
                quantized_vals.append(max_val)
                continue

            elif val <= -max_val:
                #print('best val', -max_val)
                quantized_vals.append(-max_val)
                continue

            if val >= 0:
                sign = 0
            else:
                sign = 1

            val = abs(val)
            #find the proper exponent for floating point
            exp_to_check = find_best_exponent(val, max_exp_val, min_exp_val, max_mant_val)
            #print('exp:',exp_to_check)
            #find the proper mantisa bits that 
            val = find_closest_value(val, exp_to_check, mant_length)
            quantized_vals.append(val)
        quantized_vals = np.array(quantized_vals)
        val_fp = tf.convert_to_tensor(quantized_vals.reshape(original_dims), dtype=tf.float32)
        print("Exiting symbolic convert.")
    return val_fp

'''

'''
    # Pseudocode if using tensorflow built-in functions

    if isinstance(value, np.ndarray):
        val_fp = tf.convert_to_tensor(value, dtype=tf.float32)
    - below here we would have a tensor regardless of what we are passed
    - calculate the min and max
    - pass to a quantization function
        - make sure the output represents a floating point value
    - return

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
'''
