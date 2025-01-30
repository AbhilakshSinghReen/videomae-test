def batchify(array, batch_size):
    num_batches = len(array) // batch_size
    return [array[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
